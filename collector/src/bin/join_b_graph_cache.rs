use gpu_observer_collector::join_b_trace::{
    load_device_events, load_semantic, OwnershipSlice, SemanticTrace,
};
use std::{
    env,
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

const GO_SAN_BLOCK_EVENT: u16 = 1;
const MAX_GRAPH_BYTES: u64 = 64 * 1024 * 1024;
const MAX_LAUNCH_BYTES: u64 = 64 * 1024 * 1024;
const MAX_SUMMARY_BYTES: u64 = 1024 * 1024;
const MAX_GRAPH_NODES: usize = 65_536;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GraphNode {
    graph_exec: u64,
    replay_id: u32,
    node: u64,
    grid_id: u64,
    host_timestamp_ns: u64,
    context_slot: u32,
    kernel_slot: u32,
    stream: u64,
    api_stream: u64,
    grid: [u32; 3],
    block: [u32; 3],
    expected_blocks: u64,
}

#[derive(Clone, Copy, Debug)]
struct HostLaunch {
    host_timestamp_ns: u64,
}

#[derive(Clone, Copy, Debug)]
struct Replay {
    graph_exec: u64,
    replay_id: u32,
    node_start: usize,
    node_count: usize,
    step_index: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
struct Topology {
    graph_exec: u64,
    node_start: usize,
    node_count: usize,
}

#[derive(Debug)]
struct ProbeQuality {
    callback_data_scope_function: bool,
    launch_identity: u64,
    attributed_launches: u64,
    passive_launch_records: u64,
    passive_launches_seen: u64,
    launch_records: u64,
    launch_table_drops: u64,
    callback_data_failures: u64,
    setup_missed_launches: u64,
    graph_node_identity: u64,
    graph_nodes_seen: u64,
    graph_nodes_recorded: u64,
    graph_node_table_drops: u64,
}

#[derive(Debug)]
struct Config {
    expected_nodes: usize,
    device_tail: bool,
    decode_suffix: bool,
    require_reordering: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os();
    let program = PathBuf::from(arguments.next().unwrap_or_default());
    let usage = || {
        format!(
            "usage: {} SEMANTIC.bin GRAPH-NODES.tsv DEVICE-EVENTS.bin --ordinary-launches LAUNCHES.tsv --probe-summary SUMMARY.tsv --expected-nodes N --device-tail --decode-suffix [--require-reordering]",
            program.display()
        )
    };

    let semantic_path = PathBuf::from(arguments.next().ok_or_else(&usage)?);
    let graph_path = PathBuf::from(arguments.next().ok_or_else(&usage)?);
    let device_path = PathBuf::from(arguments.next().ok_or_else(&usage)?);
    let mut ordinary_launch_path = None;
    let mut summary_path = None;
    let mut expected_nodes = None;
    let mut device_tail = false;
    let mut decode_suffix = false;
    let mut require_reordering = false;
    while let Some(argument) = arguments.next() {
        if argument == "--ordinary-launches" {
            ordinary_launch_path = Some(PathBuf::from(arguments.next().ok_or_else(&usage)?));
        } else if argument == "--probe-summary" {
            summary_path = Some(PathBuf::from(arguments.next().ok_or_else(&usage)?));
        } else if argument == "--expected-nodes" {
            let value = arguments.next().ok_or_else(&usage)?;
            expected_nodes = Some(
                value
                    .to_str()
                    .ok_or("expected node count is not UTF-8")?
                    .parse::<usize>()?,
            );
        } else if argument == "--device-tail" {
            device_tail = true;
        } else if argument == "--decode-suffix" {
            decode_suffix = true;
        } else if argument == "--require-reordering" {
            require_reordering = true;
        } else {
            return Err(usage().into());
        }
    }
    let config = Config {
        expected_nodes: expected_nodes.ok_or_else(&usage)?,
        device_tail,
        decode_suffix,
        require_reordering,
    };
    if config.expected_nodes == 0 || config.expected_nodes > MAX_GRAPH_NODES {
        return Err("expected node count is outside the bounded graph table".into());
    }
    if !config.device_tail || !config.decode_suffix {
        return Err("graph-safe attribution currently requires explicit --device-tail and --decode-suffix assumptions".into());
    }

    let semantic = load_semantic(&semantic_path)?;
    let nodes = load_graph_nodes(&graph_path)?;
    let ordinary_launches = load_ordinary_launches(&ordinary_launch_path.ok_or_else(&usage)?)?;
    let quality = load_probe_quality(&summary_path.ok_or_else(&usage)?)?;
    let (events, device_quality) = load_device_events(&device_path)?;

    if semantic.steps.is_empty()
        || semantic.loss_markers != 0
        || semantic.packed_steps != semantic.steps.len() as u64
    {
        return Err("semantic packed-layout gate failed".into());
    }
    if nodes.is_empty() {
        return Err("graph-node trace is empty".into());
    }
    if device_quality.headers != 1
        || device_quality.drops != 0
        || device_quality.sequence_errors != 0
        || device_quality.retained != events.len() as u64
    {
        return Err("device event quality gate failed".into());
    }
    if !quality.callback_data_scope_function
        || quality.launch_identity != 0
        || quality.attributed_launches != 0
        || quality.passive_launch_records != 1
        || quality.passive_launches_seen != quality.launch_records
        || quality.launch_records != ordinary_launches.len() as u64
        || quality.launch_table_drops != 0
        || quality.callback_data_failures != 0
        || quality.setup_missed_launches != 0
        || quality.graph_node_identity != 1
        || quality.graph_nodes_seen != quality.graph_nodes_recorded
        || quality.graph_nodes_recorded != nodes.len() as u64
        || quality.graph_node_table_drops != 0
    {
        return Err(format!("probe quality gate failed: {quality:?}").into());
    }

    let mut replays = group_replays(&nodes, config.expected_nodes)?;
    validate_topologies(&nodes, &replays)?;
    let packed_timeline = packed_timeline(&semantic)?;
    let mut replay_crosses_packed_boundary = 0_u64;
    for replay in &mut replays {
        let first = nodes[replay.node_start].host_timestamp_ns;
        let last = nodes[replay.node_start + replay.node_count - 1].host_timestamp_ns;
        let (step_index, crosses) = step_for_replay(first, last, &semantic, &packed_timeline);
        replay.step_index = step_index;
        replay_crosses_packed_boundary += u64::from(crosses);
    }
    if replay_crosses_packed_boundary != 0 {
        return Err("a graph replay crossed two packed-layout windows".into());
    }

    let last_prefill_packed_ns = semantic
        .steps
        .iter()
        .filter(|step| step.begin.prefill_tokens != 0)
        .filter_map(|step| step.packed_begin.map(|packed| packed.timestamp_ns))
        .max()
        .ok_or("semantic trace contains no prefill step")?;

    let mut eligible_steps = Vec::new();
    eligible_steps.try_reserve(semantic.steps.len())?;
    for (step_index, step) in semantic.steps.iter().enumerate() {
        let packed = step.packed_begin.ok_or("step is missing packed layout")?;
        if packed.timestamp_ns > last_prefill_packed_ns
            && step.begin.prefill_tokens == 0
            && step.begin.decode_tokens != 0
        {
            eligible_steps.push(step_index);
        }
    }
    if eligible_steps.is_empty() {
        return Err("decode suffix after the final prefill is empty".into());
    }

    let mut ordinary_launches_in_suffix = 0_u64;
    for launch in &ordinary_launches {
        let (step_index, crosses) = step_for_replay(
            launch.host_timestamp_ns,
            launch.host_timestamp_ns,
            &semantic,
            &packed_timeline,
        );
        if crosses {
            return Err("ordinary target launch crossed a packed-layout boundary".into());
        }
        if step_index.is_some_and(|index| eligible_steps.binary_search(&index).is_ok()) {
            ordinary_launches_in_suffix += 1;
        }
    }
    if ordinary_launches_in_suffix != 0 {
        return Err(format!(
            "decode suffix contains {ordinary_launches_in_suffix} ordinary target launches"
        )
        .into());
    }

    let mut selected_replays = Vec::new();
    selected_replays.try_reserve(eligible_steps.len())?;
    let mut step_replay_counts = zeroed_vec::<u32>(semantic.steps.len())?;
    let mut ignored_replays = 0_u64;
    for (replay_index, replay) in replays.iter().enumerate() {
        if let Some(step_index) = replay.step_index {
            if eligible_steps.binary_search(&step_index).is_ok() {
                selected_replays.push(replay_index);
                step_replay_counts[step_index] += 1;
            } else {
                ignored_replays += 1;
            }
        } else {
            ignored_replays += 1;
        }
    }
    for &step_index in &eligible_steps {
        if step_replay_counts[step_index] != 1 {
            return Err(format!(
                "eligible step {} has {} graph replays, expected exactly one",
                semantic.steps[step_index].begin.step_id, step_replay_counts[step_index]
            )
            .into());
        }
    }
    if selected_replays.len() != eligible_steps.len() {
        return Err("decode suffix and selected replay counts disagree".into());
    }

    let expected_selected_events = selected_replays.iter().try_fold(0_usize, |sum, &index| {
        let replay = replays[index];
        nodes[replay.node_start..replay.node_start + replay.node_count]
            .iter()
            .try_fold(sum, |inner, node| {
                let blocks = usize::try_from(node.expected_blocks)
                    .map_err(|_| "graph node block count does not fit usize")?;
                inner
                    .checked_add(blocks)
                    .ok_or("selected event count overflow")
            })
    })?;
    if expected_selected_events == 0 || events.len() < expected_selected_events {
        return Err(format!(
            "device trace has {} events but selected graph suffix requires {}",
            events.len(),
            expected_selected_events
        )
        .into());
    }
    let device_start = events.len() - expected_selected_events;
    let selected_events = &events[device_start..];
    if selected_events.iter().any(|event| event.launch_id != 0) {
        return Err("function-scope graph events must carry launch_id=0".into());
    }

    let mut request_event_counts = zeroed_vec::<u64>(semantic.owners.len())?;
    let mut step_node_counts = zeroed_vec::<u64>(semantic.steps.len())?;
    let mut bitset = Vec::<u64>::new();
    let max_grid_x = selected_replays
        .iter()
        .flat_map(|&index| {
            let replay = replays[index];
            nodes[replay.node_start..replay.node_start + replay.node_count]
                .iter()
                .map(|node| node.grid[0])
        })
        .max()
        .unwrap_or(0);
    let bitset_words = usize::try_from(u64::from(max_grid_x).div_ceil(64))?;
    bitset.try_reserve_exact(bitset_words)?;
    bitset.resize(bitset_words, 0);

    let mut event_cursor = 0_usize;
    let mut padding_blocks = 0_u64;
    let mut distinct_graph_execs = Vec::<u64>::new();
    let mut selected_reordered_steps = 0_u64;
    let mut stream_identity = None;

    for &replay_index in &selected_replays {
        let replay = replays[replay_index];
        let step_index = replay.step_index.ok_or("selected replay has no step")?;
        let step = &semantic.steps[step_index];
        if step.scheduler_order_mismatches != 0 {
            selected_reordered_steps += 1;
        }
        if !distinct_graph_execs.contains(&replay.graph_exec) {
            distinct_graph_execs.try_reserve(1)?;
            distinct_graph_execs.push(replay.graph_exec);
        }
        let step_owners = &semantic.owners[step.owner_start..step.owner_start + step.owner_count];
        for node in &nodes[replay.node_start..replay.node_start + replay.node_count] {
            let identity = (node.context_slot, node.stream, node.api_stream);
            if stream_identity.is_some_and(|expected| expected != identity) {
                return Err(
                    "selected graph nodes are not confined to one context/stream identity".into(),
                );
            }
            stream_identity = Some(identity);
            if node.grid[1] != 1
                || node.grid[2] != 1
                || node.expected_blocks != u64::from(node.grid[0])
                || node.grid[0] < step.begin.scheduled_tokens
            {
                return Err(
                    "selected graph node does not have the required 1-D token-row geometry".into(),
                );
            }
            let count = usize::try_from(node.expected_blocks)?;
            let node_events = selected_events
                .get(event_cursor..event_cursor + count)
                .ok_or("device event partition exceeds selected tail")?;
            event_cursor += count;
            let words = usize::try_from(node.expected_blocks.div_ceil(64))?;
            bitset[..words].fill(0);
            let mut unique_blocks = 0_u64;
            for event in node_events {
                if event.kind != GO_SAN_BLOCK_EVENT
                    || event.kernel_slot != node.kernel_slot
                    || event.block[1] != 0
                    || event.block[2] != 0
                    || event.block[0] >= node.grid[0]
                {
                    return Err(
                        "device event disagrees with graph-node kind, kernel, or geometry".into(),
                    );
                }
                let word = event.block[0] as usize / 64;
                let mask = 1_u64 << (event.block[0] % 64);
                if bitset[word] & mask != 0 {
                    return Err("duplicate block coordinate inside a graph node".into());
                }
                bitset[word] |= mask;
                unique_blocks += 1;

                if event.block[0] < step.begin.scheduled_tokens {
                    let owner_offset = owner_for_row(step_owners, event.block[0])
                        .ok_or("scheduled token row has no authoritative owner")?;
                    request_event_counts[step.owner_start + owner_offset] += 1;
                } else {
                    padding_blocks += 1;
                }
            }
            if unique_blocks != node.expected_blocks {
                return Err("graph node is missing one or more block coordinates".into());
            }
            step_node_counts[step_index] += 1;
        }
    }
    if event_cursor != selected_events.len() {
        return Err("selected device tail was not consumed exactly".into());
    }

    let mut request_count_mismatches = 0_u64;
    for &step_index in &eligible_steps {
        let step = &semantic.steps[step_index];
        let replay = selected_replays
            .iter()
            .map(|&index| replays[index])
            .find(|replay| replay.step_index == Some(step_index))
            .ok_or("eligible step has no selected replay")?;
        println!(
            "step={} packing_generation={} scheduled={} decode={} requests={} graph_exec=0x{:x} replay={} nodes={} reordered_positions={}",
            step.begin.step_id,
            step.packed_begin.map_or(0, |packed| packed.sequence_id),
            step.begin.scheduled_tokens,
            step.begin.decode_tokens,
            step.owner_count,
            replay.graph_exec,
            replay.replay_id,
            step_node_counts[step_index],
            step.scheduler_order_mismatches,
        );
        let owner_end = step.owner_start + step.owner_count;
        for (owner, &observed) in semantic.owners[step.owner_start..owner_end]
            .iter()
            .zip(&request_event_counts[step.owner_start..owner_end])
        {
            let expected = u64::from(owner.scheduled_tokens) * step_node_counts[step_index];
            request_count_mismatches += u64::from(expected != observed);
            println!(
                "  request=0x{:016x} phase={} rows=[{},{}) blocks={} expected={}",
                owner.request_id, owner.phase, owner.row_begin, owner.row_end, observed, expected,
            );
        }
    }

    println!(
        "summary status={} semantic_records={} semantic_loss={} packed_steps={} max_steps_in_flight={} scheduler_order_mismatch_steps={} ordinary_launches={} ordinary_launches_in_suffix={} graph_nodes={} graph_replays={} ignored_replays={} eligible_decode_steps={} selected_replays={} expected_nodes_per_replay={} distinct_graph_execs={} selected_reordered_steps={} device_headers={} device_retained={} device_skipped_prefix={} selected_device_events={} device_drops={} device_sequence_errors={} padding_blocks={} request_count_mismatches={} assignment=packed_window event_partition=single_stream_ordered_count device_window=explicit_tail semantic_clock=CLOCK_MONOTONIC graph_clock=CLOCK_MONOTONIC device_clock=raw_globaltimer_order_only ownership=authoritative_packed_layout",
        if request_count_mismatches == 0
            && (!config.require_reordering || selected_reordered_steps != 0)
        {
            "PASS"
        } else {
            "FAIL"
        },
        semantic.records,
        semantic.loss_markers,
        semantic.packed_steps,
        semantic.max_steps_in_flight,
        semantic.scheduler_order_mismatch_steps,
        ordinary_launches.len(),
        ordinary_launches_in_suffix,
        nodes.len(),
        replays.len(),
        ignored_replays,
        eligible_steps.len(),
        selected_replays.len(),
        config.expected_nodes,
        distinct_graph_execs.len(),
        selected_reordered_steps,
        device_quality.headers,
        device_quality.retained,
        device_start,
        selected_events.len(),
        device_quality.drops,
        device_quality.sequence_errors,
        padding_blocks,
        request_count_mismatches,
    );

    if request_count_mismatches != 0 {
        return Err("per-request block-count gate failed".into());
    }
    if config.require_reordering && selected_reordered_steps == 0 {
        return Err("decode suffix did not exercise packed-row reordering".into());
    }
    Ok(())
}

fn load_graph_nodes(path: &Path) -> Result<Vec<GraphNode>, Box<dyn Error>> {
    let file = File::open(path)?;
    let bytes = file.metadata()?.len();
    if bytes == 0 || bytes > MAX_GRAPH_BYTES {
        return Err(format!("invalid graph-node table size: {bytes}").into());
    }
    let mut nodes = Vec::new();
    nodes.try_reserve(
        (bytes as usize / 192)
            .saturating_add(1)
            .min(MAX_GRAPH_NODES),
    )?;
    let mut format_seen = false;
    let mut header_seen = false;
    for (line_number, line) in BufReader::with_capacity(1024 * 1024, file)
        .lines()
        .enumerate()
    {
        let line = line?;
        if line.starts_with("# format=GOSAN_GRAPH01 ") {
            format_seen = true;
            continue;
        }
        if line.starts_with("graph_exec\t") {
            header_seen = true;
            continue;
        }
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if nodes.len() == MAX_GRAPH_NODES {
            return Err("graph-node table exceeds the fixed subscriber capacity".into());
        }
        let mut fields = line.split('\t');
        let mut next = || {
            fields
                .next()
                .ok_or_else(|| format!("graph-node row {} is truncated", line_number + 1))
        };
        let graph_exec = parse_hex(next()?)?;
        let replay_id = next()?.parse()?;
        let node = parse_hex(next()?)?;
        let grid_id = next()?.parse()?;
        let host_timestamp_ns = next()?.parse()?;
        let context_slot = next()?.parse()?;
        let kernel_slot = next()?.parse()?;
        let stream = parse_hex(next()?)?;
        let api_stream = parse_hex(next()?)?;
        let grid = [next()?.parse()?, next()?.parse()?, next()?.parse()?];
        let block = [next()?.parse()?, next()?.parse()?, next()?.parse()?];
        let expected_blocks = next()?.parse()?;
        let function = next()?;
        if fields.next().is_some()
            || graph_exec == 0
            || replay_id == 0
            || node == 0
            || grid_id == 0
            || host_timestamp_ns == 0
            || expected_blocks == 0
            || function.is_empty()
        {
            return Err(format!("graph-node row {} is invalid", line_number + 1).into());
        }
        nodes.try_reserve(1)?;
        nodes.push(GraphNode {
            graph_exec,
            replay_id,
            node,
            grid_id,
            host_timestamp_ns,
            context_slot,
            kernel_slot,
            stream,
            api_stream,
            grid,
            block,
            expected_blocks,
        });
    }
    if !format_seen || !header_seen {
        return Err("graph-node table is missing its versioned header".into());
    }
    Ok(nodes)
}

fn load_ordinary_launches(path: &Path) -> Result<Vec<HostLaunch>, Box<dyn Error>> {
    let file = File::open(path)?;
    let bytes = file.metadata()?.len();
    if bytes == 0 || bytes > MAX_LAUNCH_BYTES {
        return Err(format!("invalid ordinary-launch table size: {bytes}").into());
    }
    let mut launches = Vec::new();
    launches.try_reserve(
        (bytes as usize / 160)
            .saturating_add(1)
            .min(MAX_GRAPH_NODES),
    )?;
    let mut format_seen = false;
    let mut header_seen = false;
    let mut expected_launch_id = 1_u64;
    for (line_number, line) in BufReader::with_capacity(1024 * 1024, file)
        .lines()
        .enumerate()
    {
        let line = line?;
        if line.starts_with("# format=GOSAN02 ") {
            format_seen = true;
            continue;
        }
        if line.starts_with("launch_id\t") {
            header_seen = true;
            continue;
        }
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if launches.len() == MAX_GRAPH_NODES {
            return Err("ordinary-launch table exceeds the fixed subscriber capacity".into());
        }
        let mut fields = line.split('\t');
        let launch_id: u64 = fields
            .next()
            .ok_or("ordinary launch is truncated")?
            .parse()?;
        let grid_id: u64 = fields
            .next()
            .ok_or("ordinary launch is truncated")?
            .parse()?;
        let host_timestamp_ns: u64 = fields
            .next()
            .ok_or("ordinary launch is truncated")?
            .parse()?;
        let context_slot: u32 = fields
            .next()
            .ok_or("ordinary launch is truncated")?
            .parse()?;
        let kernel_slot: u32 = fields
            .next()
            .ok_or("ordinary launch is truncated")?
            .parse()?;
        let stream = parse_hex(fields.next().ok_or("ordinary launch is truncated")?)?;
        let grid = [
            fields
                .next()
                .ok_or("ordinary launch is truncated")?
                .parse::<u32>()?,
            fields
                .next()
                .ok_or("ordinary launch is truncated")?
                .parse::<u32>()?,
            fields
                .next()
                .ok_or("ordinary launch is truncated")?
                .parse::<u32>()?,
        ];
        let block = [
            fields
                .next()
                .ok_or("ordinary launch is truncated")?
                .parse::<u32>()?,
            fields
                .next()
                .ok_or("ordinary launch is truncated")?
                .parse::<u32>()?,
            fields
                .next()
                .ok_or("ordinary launch is truncated")?
                .parse::<u32>()?,
        ];
        let expected_blocks: u64 = fields
            .next()
            .ok_or("ordinary launch is truncated")?
            .parse()?;
        let function = fields.next().ok_or("ordinary launch is truncated")?;
        if fields.next().is_some()
            || launch_id != expected_launch_id
            || grid_id == 0
            || host_timestamp_ns == 0
            || context_slot >= 4
            || kernel_slot >= 8192
            || stream == u64::MAX
            || grid.contains(&0)
            || block.contains(&0)
            || expected_blocks != u64::from(grid[0]) * u64::from(grid[1]) * u64::from(grid[2])
            || function.is_empty()
        {
            return Err(format!("ordinary-launch row {} is invalid", line_number + 1).into());
        }
        expected_launch_id += 1;
        launches.try_reserve(1)?;
        launches.push(HostLaunch { host_timestamp_ns });
    }
    if !format_seen || !header_seen || launches.is_empty() {
        return Err("ordinary-launch table is empty or missing its versioned header".into());
    }
    Ok(launches)
}

fn load_probe_quality(path: &Path) -> Result<ProbeQuality, Box<dyn Error>> {
    let file = File::open(path)?;
    let bytes = file.metadata()?.len();
    if bytes == 0 || bytes > MAX_SUMMARY_BYTES {
        return Err(format!("invalid probe summary size: {bytes}").into());
    }
    let first = BufReader::new(file)
        .lines()
        .next()
        .ok_or("probe summary is empty")??;
    let value = |key: &str| -> Result<&str, Box<dyn Error>> {
        first
            .split_ascii_whitespace()
            .find_map(|field| {
                field
                    .strip_prefix(key)
                    .and_then(|rest| rest.strip_prefix('='))
            })
            .ok_or_else(|| format!("probe summary is missing {key}").into())
    };
    Ok(ProbeQuality {
        callback_data_scope_function: value("callback_data_scope")? == "function",
        launch_identity: value("launch_identity")?.parse()?,
        attributed_launches: value("attributed_launches")?.parse()?,
        passive_launch_records: value("passive_launch_records")?.parse()?,
        passive_launches_seen: value("passive_launches_seen")?.parse()?,
        launch_records: value("launch_records")?.parse()?,
        launch_table_drops: value("launch_table_drops")?.parse()?,
        callback_data_failures: value("callback_data_failures")?.parse()?,
        setup_missed_launches: value("setup_missed_launches")?.parse()?,
        graph_node_identity: value("graph_node_identity")?.parse()?,
        graph_nodes_seen: value("graph_nodes_seen")?.parse()?,
        graph_nodes_recorded: value("graph_nodes_recorded")?.parse()?,
        graph_node_table_drops: value("graph_node_table_drops")?.parse()?,
    })
}

fn group_replays(
    nodes: &[GraphNode],
    expected_nodes: usize,
) -> Result<Vec<Replay>, Box<dyn Error>> {
    let mut replays = Vec::new();
    replays.try_reserve(nodes.len() / expected_nodes + 1)?;
    let mut cursor = 0_usize;
    while cursor < nodes.len() {
        let identity = (nodes[cursor].graph_exec, nodes[cursor].replay_id);
        let start = cursor;
        while cursor < nodes.len()
            && (nodes[cursor].graph_exec, nodes[cursor].replay_id) == identity
        {
            if cursor > start
                && nodes[cursor].host_timestamp_ns < nodes[cursor - 1].host_timestamp_ns
            {
                return Err("graph-node timestamps decrease within one replay".into());
            }
            cursor += 1;
        }
        let count = cursor - start;
        if count != expected_nodes {
            return Err(format!(
                "graph replay (0x{:x},{}) has {} nodes, expected {}",
                identity.0, identity.1, count, expected_nodes
            )
            .into());
        }
        if replays
            .iter()
            .any(|replay: &Replay| (replay.graph_exec, replay.replay_id) == identity)
        {
            return Err("graph replay identity appears in multiple non-contiguous ranges".into());
        }
        replays.try_reserve(1)?;
        replays.push(Replay {
            graph_exec: identity.0,
            replay_id: identity.1,
            node_start: start,
            node_count: count,
            step_index: None,
        });
    }
    Ok(replays)
}

fn validate_topologies(nodes: &[GraphNode], replays: &[Replay]) -> Result<(), Box<dyn Error>> {
    let mut topologies = Vec::<Topology>::new();
    topologies.try_reserve(replays.len().min(16))?;
    for replay in replays {
        if let Some(topology) = topologies
            .iter()
            .find(|topology| topology.graph_exec == replay.graph_exec)
        {
            if topology.node_count != replay.node_count {
                return Err("node count changed for a stable graph executable".into());
            }
            for offset in 0..replay.node_count {
                let expected = nodes[topology.node_start + offset];
                let observed = nodes[replay.node_start + offset];
                if expected.node != observed.node
                    || expected.kernel_slot != observed.kernel_slot
                    || expected.context_slot != observed.context_slot
                    || expected.stream != observed.stream
                    || expected.api_stream != observed.api_stream
                    || expected.grid != observed.grid
                    || expected.block != observed.block
                    || expected.expected_blocks != observed.expected_blocks
                {
                    return Err("node topology changed for a stable graph executable".into());
                }
            }
        } else {
            topologies.try_reserve(1)?;
            topologies.push(Topology {
                graph_exec: replay.graph_exec,
                node_start: replay.node_start,
                node_count: replay.node_count,
            });
        }
    }
    Ok(())
}

fn packed_timeline(semantic: &SemanticTrace) -> Result<Vec<(u64, usize)>, Box<dyn Error>> {
    let mut timeline = Vec::new();
    timeline.try_reserve(semantic.steps.len())?;
    for (step_index, step) in semantic.steps.iter().enumerate() {
        let packed = step.packed_begin.ok_or("step is missing packed layout")?;
        timeline.push((packed.timestamp_ns, step_index));
    }
    timeline.sort_unstable_by_key(|entry| entry.0);
    if timeline.windows(2).any(|pair| pair[0].0 >= pair[1].0) {
        return Err("packed-layout timestamps are not strictly increasing".into());
    }
    Ok(timeline)
}

fn step_for_replay(
    first_timestamp_ns: u64,
    last_timestamp_ns: u64,
    semantic: &SemanticTrace,
    timeline: &[(u64, usize)],
) -> (Option<usize>, bool) {
    let position = timeline.partition_point(|entry| entry.0 <= first_timestamp_ns);
    if position == 0 {
        return (None, false);
    }
    if position < timeline.len() && timeline[position].0 <= last_timestamp_ns {
        return (None, true);
    }
    let step_index = timeline[position - 1].1;
    (
        (last_timestamp_ns <= semantic.steps[step_index].end_timestamp_ns).then_some(step_index),
        false,
    )
}

fn owner_for_row(owners: &[OwnershipSlice], row: u32) -> Option<usize> {
    let position = owners.partition_point(|owner| owner.row_end <= row);
    owners
        .get(position)
        .filter(|owner| owner.authoritative && row >= owner.row_begin && row < owner.row_end)
        .map(|_| position)
}

fn zeroed_vec<T: Clone + Default>(len: usize) -> Result<Vec<T>, std::collections::TryReserveError> {
    let mut values = Vec::new();
    values.try_reserve_exact(len)?;
    values.resize(len, T::default());
    Ok(values)
}

fn parse_hex(value: &str) -> Result<u64, Box<dyn Error>> {
    Ok(u64::from_str_radix(
        value.strip_prefix("0x").unwrap_or(value),
        16,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(exec: u64, replay: u32, node_id: u64, timestamp: u64) -> GraphNode {
        GraphNode {
            graph_exec: exec,
            replay_id: replay,
            node: node_id,
            grid_id: node_id + 100,
            host_timestamp_ns: timestamp,
            context_slot: 0,
            kernel_slot: 7,
            stream: 0,
            api_stream: 0,
            grid: [2, 1, 1],
            block: [512, 1, 1],
            expected_blocks: 2,
        }
    }

    #[test]
    fn groups_replays_and_accepts_stable_topology() {
        let nodes = [
            node(9, 1, 11, 100),
            node(9, 1, 12, 101),
            node(9, 2, 11, 200),
            node(9, 2, 12, 201),
        ];
        let replays = group_replays(&nodes, 2).unwrap();
        assert_eq!(replays.len(), 2);
        validate_topologies(&nodes, &replays).unwrap();
    }

    #[test]
    fn rejects_changed_topology_for_same_graph_exec() {
        let nodes = [
            node(9, 1, 11, 100),
            node(9, 1, 12, 101),
            node(9, 2, 11, 200),
            node(9, 2, 13, 201),
        ];
        let replays = group_replays(&nodes, 2).unwrap();
        assert!(validate_topologies(&nodes, &replays).is_err());
    }

    #[test]
    fn maps_only_authoritative_packed_rows() {
        let owners = [
            OwnershipSlice {
                request_id: 1,
                phase: 2,
                scheduled_tokens: 2,
                row_begin: 0,
                row_end: 2,
                authoritative: true,
            },
            OwnershipSlice {
                request_id: 2,
                phase: 2,
                scheduled_tokens: 1,
                row_begin: 2,
                row_end: 3,
                authoritative: true,
            },
        ];
        assert_eq!(owner_for_row(&owners, 0), Some(0));
        assert_eq!(owner_for_row(&owners, 2), Some(1));
        assert_eq!(owner_for_row(&owners, 3), None);
    }

    #[test]
    fn event_type_remains_compact_copy_data() {
        assert_eq!(
            std::mem::size_of::<gpu_observer_collector::join_b_trace::DeviceEvent>(),
            40
        );
    }
}

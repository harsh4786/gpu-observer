//! Offline, bounded Join-B instruction sampler for EXP-0025.
//!
//! The inference hot path remains fixed-record/no_std. This cold analyzer joins
//! sampled Compute Sanitizer callbacks after capture, so its bounded maps and
//! strings cannot perturb request latency.

use gpu_observer_collector::join_b_trace::{load_launches, load_semantic, OwnershipSlice, Step};
use std::{
    collections::BTreeMap,
    env,
    error::Error,
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

const MAX_TEXT_BYTES: u64 = 256 * 1024 * 1024;
const EVENT_HEADER_BYTES: usize = 64;
const EVENT_BYTES: usize = 64;
const EVENT_FORMAT_VERSION: u32 = 2;
const MAX_EVENT_BYTES: u64 = 128 * 1024 * 1024;

#[derive(Clone)]
struct FunctionInfo {
    slot: u32,
    pc: u64,
    size: u64,
    callbacks: u64,
    emitted: u64,
    dropped: u64,
    name: String,
}

#[derive(Clone)]
struct Instruction {
    offset: u64,
    text: String,
}

#[derive(Clone, Copy)]
struct SampleEvent {
    device_timestamp_raw: u64,
    pc: u64,
    address: u64,
    launch_id: u64,
    kernel_slot: u32,
    block: [u32; 3],
    kind: u16,
    access_size: u16,
}

#[derive(Default)]
struct EventQuality {
    retained: u64,
    write_attempts: u64,
    drops: u64,
    sequence_errors: u64,
}

#[derive(Default)]
struct RequestSamples {
    events: u64,
    sampled_bytes: u64,
    kinds: [u64; 9],
    first_device_timestamp: u64,
    last_device_timestamp: u64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os();
    let program = PathBuf::from(args.next().unwrap_or_default());
    let usage = || {
        format!(
        "usage: {} SEMANTIC.bin LAUNCHES.tsv EVENTS.bin SUMMARY.tsv SASS.txt REQUESTS.tsv PCS.tsv",
        program.display()
    )
    };
    let semantic_path = PathBuf::from(args.next().ok_or_else(&usage)?);
    let launch_path = PathBuf::from(args.next().ok_or_else(&usage)?);
    let event_path = PathBuf::from(args.next().ok_or_else(&usage)?);
    let summary_path = PathBuf::from(args.next().ok_or_else(&usage)?);
    let sass_path = PathBuf::from(args.next().ok_or_else(&usage)?);
    let request_output = PathBuf::from(args.next().ok_or_else(&usage)?);
    let pc_output = PathBuf::from(args.next().ok_or_else(&usage)?);
    if args.next().is_some() {
        return Err(usage().into());
    }

    let semantic = load_semantic(&semantic_path)?;
    if semantic.steps.is_empty()
        || semantic.loss_markers != 0
        || semantic.packed_steps != semantic.steps.len() as u64
    {
        return Err("semantic trace lacks lossless authoritative packed layouts".into());
    }
    validate_step_timeline(&semantic.steps)?;

    let target = "reshape_and_cache_flash_kernel";
    let functions = load_functions(&summary_path, target)?;
    let instructions = load_sass(&sass_path, target)?;
    let mut launches = load_launches(&launch_path)?;
    launches.sort_unstable_by_key(|launch| launch.launch_id);
    if launches.is_empty()
        || launches
            .windows(2)
            .any(|pair| pair[0].launch_id == pair[1].launch_id)
    {
        return Err("launch table is empty or has duplicate launch IDs".into());
    }

    let mut launch_steps = Vec::new();
    launch_steps.try_reserve_exact(launches.len())?;
    let mut launches_per_step = vec![0_u64; semantic.steps.len()];
    for launch in &launches {
        if !launch.function.contains(target) || launch.grid[1] != 1 || launch.grid[2] != 1 {
            return Err("launch table contains a non-target or non-linear grid".into());
        }
        let step_index = step_for_timestamp(launch.host_timestamp_ns, &semantic.steps)
            .ok_or("target launch does not map uniquely inside an engine step")?;
        let step = &semantic.steps[step_index];
        let packed = step
            .packed_begin
            .ok_or("step lacks packed-layout timestamp")?;
        if launch.host_timestamp_ns < packed.timestamp_ns
            || launch.grid[0] < step.begin.scheduled_tokens
        {
            return Err("launch precedes packed layout or cannot cover packed token rows".into());
        }
        launch_steps.push(step_index);
        launches_per_step[step_index] += 1;
    }
    if launches_per_step.contains(&0) {
        return Err("one or more measured steps have no target-kernel launch".into());
    }

    let (events, quality) = load_sample_events(&event_path)?;
    if events.is_empty() || quality.drops != 0 || quality.sequence_errors != 0 {
        return Err("sampled device trace is empty or lossy".into());
    }
    let total_function_emitted = functions.iter().try_fold(0_u64, |sum, function| {
        sum.checked_add(function.emitted)
            .ok_or("emitted count overflow")
    })?;
    let total_function_callbacks = functions.iter().try_fold(0_u64, |sum, function| {
        sum.checked_add(function.callbacks)
            .ok_or("callback count overflow")
    })?;
    if functions.iter().any(|function| function.dropped != 0)
        || total_function_emitted != quality.retained
        || quality.retained != events.len() as u64
        || quality.write_attempts != quality.retained
    {
        return Err("function counters and retained zero-drop event file disagree".into());
    }

    let mut requests: BTreeMap<(u64, u64, u8), RequestSamples> = BTreeMap::new();
    let mut pc_samples: BTreeMap<(u64, u16), u64> = BTreeMap::new();
    let mut sampled_launches = vec![false; launches.len()];
    let mut padding_events = 0_u64;
    let mut orphan_events = 0_u64;
    let mut pc_mismatches = 0_u64;
    let mut geometry_mismatches = 0_u64;
    let mut ownership_mismatches = 0_u64;

    for event in &events {
        let Ok(launch_index) =
            launches.binary_search_by_key(&event.launch_id, |launch| launch.launch_id)
        else {
            orphan_events += 1;
            continue;
        };
        let launch = &launches[launch_index];
        sampled_launches[launch_index] = true;
        if event.kernel_slot != launch.kernel_slot
            || event.block[1] != 0
            || event.block[2] != 0
            || event.block[0] >= launch.grid[0]
        {
            geometry_mismatches += 1;
            continue;
        }
        let Some(function) = functions
            .iter()
            .find(|function| function.slot == event.kernel_slot)
        else {
            pc_mismatches += 1;
            continue;
        };
        let function_end = function
            .pc
            .checked_add(function.size)
            .ok_or("PC range overflow")?;
        if event.pc < function.pc || event.pc >= function_end {
            pc_mismatches += 1;
            continue;
        }
        let pc_offset = event.pc - function.pc;
        if instructions
            .binary_search_by_key(&pc_offset, |instruction| instruction.offset)
            .is_err()
        {
            pc_mismatches += 1;
            continue;
        }
        if !(2..=8).contains(&event.kind)
            || (event.kind == 8 && event.address != 0)
            || (event.kind != 8 && (event.address == 0 || event.access_size == 0))
        {
            pc_mismatches += 1;
            continue;
        }

        let step = &semantic.steps[launch_steps[launch_index]];
        let owners = &semantic.owners[step.owner_start..step.owner_start + step.owner_count];
        let Some(owner) = owner_for_row(event.block[0], owners) else {
            if event.block[0] >= step.begin.scheduled_tokens {
                padding_events += 1;
            } else {
                ownership_mismatches += 1;
            }
            continue;
        };
        let key = (step.begin.step_id, owner.request_id, owner.phase);
        let aggregate = requests.entry(key).or_default();
        aggregate.events += 1;
        aggregate.kinds[event.kind as usize] += 1;
        aggregate.sampled_bytes = aggregate
            .sampled_bytes
            .checked_add(u64::from(event.access_size))
            .ok_or("sampled byte count overflow")?;
        if aggregate.first_device_timestamp == 0 {
            aggregate.first_device_timestamp = event.device_timestamp_raw;
        }
        aggregate.first_device_timestamp = aggregate
            .first_device_timestamp
            .min(event.device_timestamp_raw);
        aggregate.last_device_timestamp = aggregate
            .last_device_timestamp
            .max(event.device_timestamp_raw);
        *pc_samples.entry((pc_offset, event.kind)).or_default() += 1;
    }

    if orphan_events != 0
        || pc_mismatches != 0
        || geometry_mismatches != 0
        || ownership_mismatches != 0
    {
        return Err(format!(
            "instruction join failed: orphan={orphan_events} pc={pc_mismatches} geometry={geometry_mismatches} ownership={ownership_mismatches}"
        ).into());
    }
    let attributed_events = requests.values().map(|value| value.events).sum::<u64>();
    if attributed_events + padding_events != events.len() as u64 || attributed_events == 0 {
        return Err("sample accounting does not close".into());
    }

    write_requests(&request_output, &requests)?;
    write_pcs(&pc_output, &pc_samples, &instructions)?;
    let sampled_launch_count = sampled_launches.iter().filter(|value| **value).count();
    println!(
        "summary status=PASS steps={} launches={} sampled_launches={} callbacks={} retained_samples={} attributed_samples={} padding_samples={} sample_fraction={:.9} semantic_loss={} device_drops={} sequence_errors={} orphan_events={} pc_mismatches={} geometry_mismatches={} ownership_mismatches={} ownership=authoritative_packed_rows target={} scope=diagnostic_only",
        semantic.steps.len(),
        launches.len(),
        sampled_launch_count,
        total_function_callbacks,
        quality.retained,
        attributed_events,
        padding_events,
        quality.retained as f64 / total_function_callbacks as f64,
        semantic.loss_markers,
        quality.drops,
        quality.sequence_errors,
        orphan_events,
        pc_mismatches,
        geometry_mismatches,
        ownership_mismatches,
        target,
    );
    Ok(())
}

fn validate_step_timeline(steps: &[Step]) -> Result<(), Box<dyn Error>> {
    for step in steps {
        if step.begin.timestamp_ns >= step.end_timestamp_ns || step.owner_count == 0 {
            return Err("invalid step interval or empty packed ownership".into());
        }
    }
    if steps.windows(2).any(|pair| {
        pair[0].begin.timestamp_ns >= pair[1].begin.timestamp_ns
            || pair[0].end_timestamp_ns > pair[1].begin.timestamp_ns
    }) {
        return Err("EXP-0025 requires non-overlapping synchronous step intervals".into());
    }
    Ok(())
}

fn step_for_timestamp(timestamp: u64, steps: &[Step]) -> Option<usize> {
    let position = steps.partition_point(|step| step.begin.timestamp_ns <= timestamp);
    if position == 0 {
        return None;
    }
    let index = position - 1;
    (timestamp <= steps[index].end_timestamp_ns).then_some(index)
}

fn owner_for_row(row: u32, owners: &[OwnershipSlice]) -> Option<OwnershipSlice> {
    let position = owners.partition_point(|owner| owner.row_begin <= row);
    if position == 0 {
        return None;
    }
    let owner = owners[position - 1];
    (row < owner.row_end).then_some(owner)
}

fn load_sample_events(path: &Path) -> Result<(Vec<SampleEvent>, EventQuality), Box<dyn Error>> {
    let file = File::open(path)?;
    let bytes = file.metadata()?.len();
    if bytes == 0 || bytes > MAX_EVENT_BYTES {
        return Err("sampled event file violates size bound".into());
    }
    let mut input = BufReader::with_capacity(1024 * 1024, file);
    let mut events = Vec::new();
    events.try_reserve((bytes as usize / EVENT_BYTES).saturating_add(1))?;
    let mut quality = EventQuality::default();
    let mut consumed = 0_u64;
    while consumed < bytes {
        let mut header = [0_u8; EVENT_HEADER_BYTES];
        input.read_exact(&mut header)?;
        consumed += EVENT_HEADER_BYTES as u64;
        if &header[0..8] != b"GOSAN02\0"
            || le_u32(&header, 8) != EVENT_FORMAT_VERSION
            || le_u32(&header, 20) as usize != EVENT_BYTES
        {
            return Err("unsupported sampled event header".into());
        }
        let capacity = le_u64(&header, 24);
        let retained = le_u64(&header, 32);
        let write_attempts = le_u64(&header, 40);
        let drops = le_u64(&header, 48);
        if retained > capacity || retained > write_attempts || retained > u32::MAX as u64 {
            return Err("invalid sampled event counters".into());
        }
        quality.retained = quality
            .retained
            .checked_add(retained)
            .ok_or("retained overflow")?;
        quality.write_attempts = quality
            .write_attempts
            .checked_add(write_attempts)
            .ok_or("write-attempt overflow")?;
        quality.drops = quality.drops.checked_add(drops).ok_or("drop overflow")?;
        events.try_reserve(retained as usize)?;
        for expected_sequence in 0..retained {
            let mut raw = [0_u8; EVENT_BYTES];
            input.read_exact(&mut raw)?;
            consumed += EVENT_BYTES as u64;
            if u64::from(le_u32(&raw, 32)) != expected_sequence {
                quality.sequence_errors += 1;
            }
            events.push(SampleEvent {
                device_timestamp_raw: le_u64(&raw, 0),
                pc: le_u64(&raw, 8),
                address: le_u64(&raw, 16),
                launch_id: le_u64(&raw, 24),
                kernel_slot: le_u32(&raw, 36),
                block: [le_u32(&raw, 40), le_u32(&raw, 44), le_u32(&raw, 48)],
                kind: le_u16(&raw, 58),
                access_size: le_u16(&raw, 60),
            });
        }
    }
    if consumed != bytes {
        return Err("sampled event file is truncated".into());
    }
    Ok((events, quality))
}

fn le_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap())
}

fn le_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn le_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn load_functions(path: &Path, target: &str) -> Result<Vec<FunctionInfo>, Box<dyn Error>> {
    let expected = "context_slot\tkernel_slot\tlaunches\texpected_block_callbacks\tactual_callbacks\temitted\tdropped\tfunction_pc\tfunction_size\tmodule\tfunction";
    let mut header_seen = false;
    let mut output = Vec::new();
    for line in BufReader::new(bounded(path)?).lines() {
        let line = line?;
        if line.starts_with("context_slot\tkernel_slot\t") {
            if line != expected {
                return Err("unexpected sanitizer summary schema".into());
            }
            header_seen = true;
            continue;
        }
        if line.is_empty() || line.starts_with('#') || !line.contains(target) {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if !header_seen || fields.len() != 11 {
            return Err("malformed target function row".into());
        }
        let callbacks: u64 = fields[4].parse()?;
        let emitted: u64 = fields[5].parse()?;
        if emitted == 0 {
            continue;
        }
        output.try_reserve(1)?;
        output.push(FunctionInfo {
            slot: fields[1].parse()?,
            pc: parse_hex(fields[7])?,
            size: fields[8].parse()?,
            callbacks,
            emitted,
            dropped: fields[6].parse()?,
            name: fields[10].to_owned(),
        });
    }
    output.sort_unstable_by_key(|function| function.slot);
    output.dedup_by_key(|function| function.slot);
    if output.len() != 1 {
        return Err("expected exactly one active disassembled target specialization".into());
    }
    if output.iter().any(|function| {
        function.pc == 0
            || function.size == 0
            || function.callbacks < function.emitted
            || !function.name.contains(target)
    }) {
        return Err("invalid target function counters or PC range".into());
    }
    Ok(output)
}

fn load_sass(path: &Path, target: &str) -> Result<Vec<Instruction>, Box<dyn Error>> {
    let mut in_target = false;
    let mut output = Vec::new();
    for line in BufReader::new(bounded(path)?).lines() {
        let line = line?;
        if let Some((_, name)) = line.split_once("Function :") {
            in_target = name.contains(target);
            continue;
        }
        if !in_target {
            continue;
        }
        let Some(begin) = line.find("/*") else {
            continue;
        };
        let Some(relative_end) = line[begin + 2..].find("*/") else {
            continue;
        };
        let end = begin + 2 + relative_end;
        let value = line[begin + 2..end].trim();
        if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            continue;
        }
        if output.len() == 65_536 {
            return Err("target SASS exceeds instruction bound".into());
        }
        output.try_reserve(1)?;
        output.push(Instruction {
            offset: u64::from_str_radix(value, 16)?,
            text: line[end + 2..].trim().replace(['\t', '\n'], " "),
        });
    }
    output.sort_unstable_by_key(|instruction| instruction.offset);
    if output.is_empty()
        || output
            .windows(2)
            .any(|pair| pair[0].offset == pair[1].offset)
    {
        return Err("target SASS is empty or has duplicate offsets".into());
    }
    Ok(output)
}

fn write_requests(
    path: &Path,
    requests: &BTreeMap<(u64, u64, u8), RequestSamples>,
) -> Result<(), Box<dyn Error>> {
    let mut output = exclusive_writer(path)?;
    writeln!(output, "step_id\trequest_id_hash\tphase\tsamples\tsampled_bytes\tglobal_reads\tglobal_writes\tshared_reads\tshared_writes\tlocal_reads\tlocal_writes\tbarriers\tdevice_first_raw\tdevice_last_raw")?;
    for ((step_id, request_id, phase), value) in requests {
        writeln!(
            output,
            "{}\t0x{:016x}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            step_id,
            request_id,
            phase,
            value.events,
            value.sampled_bytes,
            value.kinds[2],
            value.kinds[3],
            value.kinds[4],
            value.kinds[5],
            value.kinds[6],
            value.kinds[7],
            value.kinds[8],
            value.first_device_timestamp,
            value.last_device_timestamp,
        )?;
    }
    output.flush()?;
    output.get_ref().sync_data()?;
    Ok(())
}

fn write_pcs(
    path: &Path,
    samples: &BTreeMap<(u64, u16), u64>,
    instructions: &[Instruction],
) -> Result<(), Box<dyn Error>> {
    let mut output = exclusive_writer(path)?;
    writeln!(
        output,
        "pc_offset\tevent_kind\tevent_name\tsamples\tstatic_class\tsass"
    )?;
    for ((offset, kind), count) in samples {
        let index = instructions
            .binary_search_by_key(offset, |instruction| instruction.offset)
            .map_err(|_| "sampled PC missing from static SASS")?;
        let instruction = &instructions[index];
        writeln!(
            output,
            "0x{:x}\t{}\t{}\t{}\t{}\t{}",
            offset,
            kind,
            kind_name(*kind),
            count,
            static_class(&instruction.text),
            instruction.text,
        )?;
    }
    output.flush()?;
    output.get_ref().sync_data()?;
    Ok(())
}

fn kind_name(kind: u16) -> &'static str {
    match kind {
        2 => "global_read",
        3 => "global_write",
        4 => "shared_read",
        5 => "shared_write",
        6 => "local_read",
        7 => "local_write",
        8 => "barrier",
        _ => "invalid",
    }
}

fn static_class(text: &str) -> &'static str {
    let upper = text.as_bytes();
    if contains_ascii(upper, b"BAR") {
        "barrier"
    } else if contains_ascii(upper, b"LDG") || contains_ascii(upper, b"LD.") {
        "load"
    } else if contains_ascii(upper, b"STG") || contains_ascii(upper, b"ST.") {
        "store"
    } else if contains_ascii(upper, b"MMA") {
        "tensor"
    } else if contains_ascii(upper, b"BRA") || contains_ascii(upper, b"JMP") {
        "control"
    } else {
        "other"
    }
}

fn contains_ascii(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|window| {
        window
            .iter()
            .zip(needle)
            .all(|(left, right)| left.to_ascii_uppercase() == *right)
    })
}

fn parse_hex(value: &str) -> Result<u64, Box<dyn Error>> {
    Ok(u64::from_str_radix(
        value.strip_prefix("0x").unwrap_or(value),
        16,
    )?)
}

fn bounded(path: &Path) -> Result<File, Box<dyn Error>> {
    let file = File::open(path)?;
    let bytes = file.metadata()?.len();
    if bytes == 0 || bytes > MAX_TEXT_BYTES {
        return Err(format!("input violates size bound: {}", path.display()).into());
    }
    Ok(file)
}

fn exclusive_writer(path: &Path) -> Result<BufWriter<File>, Box<dyn Error>> {
    let file = OpenOptions::new().create_new(true).write(true).open(path)?;
    Ok(BufWriter::with_capacity(1024 * 1024, file))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owner_lookup_respects_compacted_row_ranges() {
        let owners = [
            OwnershipSlice {
                request_id: 1,
                phase: 2,
                scheduled_tokens: 1,
                row_begin: 0,
                row_end: 1,
                authoritative: true,
            },
            OwnershipSlice {
                request_id: 9,
                phase: 1,
                scheduled_tokens: 3,
                row_begin: 1,
                row_end: 4,
                authoritative: true,
            },
        ];
        assert_eq!(owner_for_row(0, &owners).unwrap().request_id, 1);
        assert_eq!(owner_for_row(3, &owners).unwrap().request_id, 9);
        assert!(owner_for_row(4, &owners).is_none());
    }

    #[test]
    fn cold_projection_stays_smaller_than_wire_record() {
        assert!(core::mem::size_of::<SampleEvent>() < EVENT_BYTES);
    }

    #[test]
    fn instruction_classifier_is_bounded_and_case_insensitive() {
        assert_eq!(static_class("LDG.E.128 R4, [R2];"), "load");
        assert_eq!(static_class("bar.sync 0;"), "barrier");
        assert_eq!(static_class("HMMA.1688.F32 R0, R2, R4, R0;"), "tensor");
    }
}

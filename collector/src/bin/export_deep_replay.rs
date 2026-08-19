use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

use serde::Serialize;
use serde_json::Value;

const HEADER_BYTES: usize = 64;
const EVENT_BYTES: usize = 64;
const MAX_BYTES: u64 = 256 * 1024 * 1024;
const COMMENT_END: &str = concat!("*", "/");

#[derive(Clone)]
struct Function {
    slot: u32,
    pc: u64,
    size: u64,
    actual_callbacks: u64,
    emitted: u64,
    name: String,
}

#[derive(Clone)]
struct Instruction {
    offset: u64,
    text: String,
}

#[derive(Clone, Default)]
struct Samples {
    total: u64,
    offsets: BTreeMap<u64, u64>,
}

type EventSamples = (BTreeMap<u32, Samples>, u64, u64);

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DeepReplay {
    schema: &'static str,
    capture_status: &'static str,
    replay_fingerprint: String,
    semantic_signature: String,
    module_hash: String,
    total_events: u64,
    total_drops: u64,
    functions: Vec<FunctionView>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FunctionView {
    name: String,
    kernel_slot: u32,
    function_pc: String,
    function_size: u64,
    callback_count: u64,
    sampled_pc_count: usize,
    sass: Vec<SassView>,
}

#[derive(Serialize)]
struct SassView {
    offset: String,
    instruction: String,
    samples: u64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = args()?;
    let run = json(Path::new(required(&args, "--run")?))?;
    let replay_fingerprint = string(&run, "replayFingerprint")?.to_owned();
    let semantic_signature = string(&run, "semanticSignature")?.to_owned();
    let module_hash = required(&args, "--module-hash")?.to_owned();
    if module_hash.len() != 64 || !module_hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("module hash must be SHA-256 hex".into());
    }
    let target = required(&args, "--target")?;
    let functions = functions(Path::new(required(&args, "--summary")?), target)?;
    let (samples, total_events, total_drops) =
        events(Path::new(required(&args, "--events")?), &functions)?;
    if total_drops != 0 {
        return Err("refusing dropped sanitizer events".into());
    }
    let sampled_events = samples.values().try_fold(0_u64, |total, sample| {
        total
            .checked_add(sample.total)
            .ok_or("sample count overflow")
    })?;
    if total_events == 0 || sampled_events == 0 {
        return Err("refusing deep replay without target device callbacks".into());
    }
    if sampled_events != total_events {
        return Err("sanitizer event file contains callbacks outside selected functions".into());
    }
    let sections = sass(Path::new(required(&args, "--sass")?), target)?;

    let mut views = Vec::new();
    for function in functions {
        let sample = samples.get(&function.slot).cloned().unwrap_or_default();
        if sample.total != function.emitted || function.actual_callbacks != function.emitted {
            return Err(format!(
                "sanitizer function counters disagree for {}: callbacks={} emitted={} retained={}",
                function.name, function.actual_callbacks, function.emitted, sample.total
            )
            .into());
        }
        let instructions = match_section(&function, &sections)?;
        for offset in sample.offsets.keys() {
            if instructions
                .binary_search_by_key(offset, |instruction| instruction.offset)
                .is_err()
            {
                return Err(format!(
                    "sample PC offset {offset:#x} is absent from SASS for {}",
                    function.name
                )
                .into());
            }
        }
        views.push(FunctionView {
            name: function.name,
            kernel_slot: function.slot,
            function_pc: format!("0x{:016x}", function.pc),
            function_size: function.size,
            callback_count: sample.total,
            sampled_pc_count: sample.offsets.len(),
            sass: instructions
                .iter()
                .map(|instruction| SassView {
                    offset: format!("0x{:04x}", instruction.offset),
                    instruction: instruction.text.clone(),
                    samples: sample
                        .offsets
                        .get(&instruction.offset)
                        .copied()
                        .unwrap_or(0),
                })
                .collect(),
        });
    }

    let output = DeepReplay {
        schema: "GPU_OBSERVER_DEEP_REPLAY_01",
        capture_status: "matched_compute_sanitizer_replay",
        replay_fingerprint,
        semantic_signature,
        module_hash,
        total_events,
        total_drops,
        functions: views,
    };
    let path = Path::new(required(&args, "--output")?);
    let file = OpenOptions::new().create_new(true).write(true).open(path)?;
    let mut writer = BufWriter::with_capacity(1024 * 1024, file);
    serde_json::to_writer_pretty(&mut writer, &output)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_data()?;
    println!(
        "deep_replay functions={} events={} output={}",
        output.functions.len(),
        total_events,
        path.display()
    );
    Ok(())
}

fn args() -> Result<BTreeMap<String, String>, Box<dyn Error>> {
    let mut output = BTreeMap::new();
    let mut args = env::args().skip(1);
    while let Some(name) = args.next() {
        if !known_option(&name) {
            return Err(format!("unknown option: {name}\n{}", usage()).into());
        }
        let value = args.next().ok_or_else(usage)?;
        if output.insert(name, value).is_some() {
            return Err("duplicate option".into());
        }
    }
    for name in [
        "--run",
        "--summary",
        "--events",
        "--sass",
        "--module-hash",
        "--target",
        "--output",
    ] {
        if !output.contains_key(name) {
            return Err(usage().into());
        }
    }
    Ok(output)
}

fn known_option(name: &str) -> bool {
    matches!(
        name,
        "--run" | "--summary" | "--events" | "--sass" | "--module-hash" | "--target" | "--output"
    )
}

fn usage() -> String {
    "usage: export_deep_replay --run RUN.json --summary SAN.summary.tsv \
--events SAN.events.bin --sass CUOBJDUMP.sass --module-hash SHA256 \
--target SUBSTRING --output DEEP.json"
        .to_owned()
}

fn required<'a>(args: &'a BTreeMap<String, String>, key: &str) -> Result<&'a str, Box<dyn Error>> {
    args.get(key)
        .map(String::as_str)
        .ok_or_else(|| usage().into())
}

fn json(path: &Path) -> Result<Value, Box<dyn Error>> {
    let value: Value = serde_json::from_reader(BufReader::new(bounded(path)?))?;
    if !value.is_object() {
        return Err("run manifest must be an object".into());
    }
    Ok(value)
}

fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, Box<dyn Error>> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("run manifest lacks {key}").into())
}

fn functions(path: &Path, target: &str) -> Result<Vec<Function>, Box<dyn Error>> {
    let expected = "context_slot\tkernel_slot\tlaunches\texpected_block_callbacks\tactual_callbacks\temitted\tdropped\tfunction_pc\tfunction_size\tmodule\tfunction";
    let mut header = false;
    let mut output = Vec::new();
    for line in BufReader::new(bounded(path)?).lines() {
        let line = line?;
        if line.starts_with("context_slot\tkernel_slot\t") {
            if line != expected {
                return Err("sanitizer summary lacks function PC/size columns".into());
            }
            header = true;
            continue;
        }
        if line.is_empty() || line.starts_with('#') || !line.contains(target) {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if !header || fields.len() != 11 {
            return Err("malformed sanitizer function row".into());
        }
        let actual_callbacks: u64 = fields[4].parse()?;
        let emitted: u64 = fields[5].parse()?;
        let dropped: u64 = fields[6].parse()?;
        if dropped != 0 {
            return Err("refusing per-function sanitizer drops".into());
        }
        if actual_callbacks == 0 && emitted == 0 {
            continue;
        }
        if actual_callbacks != emitted {
            return Err("sanitizer callback and emitted counters disagree".into());
        }
        let pc = hex(fields[7])?;
        let size = fields[8].parse()?;
        if pc == 0 || size == 0 {
            return Err("target function has no PC range".into());
        }
        output.push(Function {
            slot: fields[1].parse()?,
            pc,
            size,
            actual_callbacks,
            emitted,
            name: fields[10].to_owned(),
        });
    }
    output.sort_by_key(|function| function.slot);
    output.dedup_by_key(|function| function.slot);
    if output.is_empty() {
        return Err("no target function in sanitizer summary".into());
    }
    Ok(output)
}

fn events(path: &Path, functions: &[Function]) -> Result<EventSamples, Box<dyn Error>> {
    let mut file = bounded(path)?;
    let bytes = file.metadata()?.len();
    let mut consumed = 0_u64;
    let mut total = 0_u64;
    let mut drops = 0_u64;
    let mut output: BTreeMap<u32, Samples> = BTreeMap::new();
    while consumed < bytes {
        let mut header = [0_u8; HEADER_BYTES];
        file.read_exact(&mut header)?;
        consumed += HEADER_BYTES as u64;
        if &header[0..8] != b"GOSAN02\0"
            || le32(&header, 8) != 2
            || le32(&header, 20) as usize != EVENT_BYTES
        {
            return Err("invalid GOSAN02 header".into());
        }
        let retained = le64(&header, 32);
        if retained > le64(&header, 24) || retained > le64(&header, 40) {
            return Err("invalid sanitizer event counts".into());
        }
        total = total.checked_add(retained).ok_or("event count overflow")?;
        drops = drops
            .checked_add(le64(&header, 48))
            .ok_or("drop count overflow")?;
        for sequence in 0..retained {
            let mut raw = [0_u8; EVENT_BYTES];
            file.read_exact(&mut raw)?;
            consumed += EVENT_BYTES as u64;
            if u64::from(le32(&raw, 32)) != sequence {
                return Err("event sequence gap".into());
            }
            let slot = le32(&raw, 36);
            let Some(function) = functions.iter().find(|function| function.slot == slot) else {
                continue;
            };
            let pc = le64(&raw, 8);
            let end = function
                .pc
                .checked_add(function.size)
                .ok_or("PC range overflow")?;
            if pc < function.pc || pc >= end {
                return Err(format!("PC {pc:#x} outside [{:#x},{end:#x})", function.pc).into());
            }
            let sample = output.entry(slot).or_default();
            sample.total += 1;
            *sample.offsets.entry(pc - function.pc).or_default() += 1;
        }
    }
    if consumed != bytes {
        return Err("truncated sanitizer event file".into());
    }
    Ok((output, total, drops))
}

fn sass(path: &Path, target: &str) -> Result<BTreeMap<String, Vec<Instruction>>, Box<dyn Error>> {
    let mut output: BTreeMap<String, Vec<Instruction>> = BTreeMap::new();
    let mut current = None;
    for line in BufReader::new(bounded(path)?).lines() {
        let line = line?;
        if let Some((_, value)) = line.split_once("Function :") {
            let name = value.trim().to_owned();
            current = name.contains(target).then_some(name.clone());
            if current.is_some() {
                output.entry(name).or_default();
            }
            continue;
        }
        let Some(name) = current.as_ref() else {
            continue;
        };
        let Some(begin) = line.find("/*") else {
            continue;
        };
        let Some(relative_end) = line[begin + 2..].find(COMMENT_END) else {
            continue;
        };
        let end = begin + 2 + relative_end;
        let value = line[begin + 2..end].trim();
        if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            continue;
        }
        let instructions = output.get_mut(name).unwrap();
        if instructions.len() == 65_536 {
            return Err("SASS function exceeds instruction bound".into());
        }
        instructions.push(Instruction {
            offset: u64::from_str_radix(value, 16)?,
            text: line[end + COMMENT_END.len()..].trim().to_owned(),
        });
    }
    for instructions in output.values_mut() {
        instructions.sort_by_key(|instruction| instruction.offset);
        if instructions.is_empty() {
            return Err("target SASS section contains no instructions".into());
        }
        if instructions
            .windows(2)
            .any(|pair| pair[0].offset == pair[1].offset)
        {
            return Err("target SASS section contains duplicate instruction offsets".into());
        }
    }
    if output.is_empty() {
        return Err("no target SASS section".into());
    }
    Ok(output)
}

fn match_section<'a>(
    function: &Function,
    sections: &'a BTreeMap<String, Vec<Instruction>>,
) -> Result<&'a [Instruction], Box<dyn Error>> {
    if let Some(value) = sections.get(&function.name) {
        return Ok(value);
    }
    let candidates: Vec<&Vec<Instruction>> = sections
        .iter()
        .filter(|(name, _)| name.contains(&function.name) || function.name.contains(name.as_str()))
        .map(|(_, value)| value)
        .collect();
    match candidates.as_slice() {
        [only] => Ok(only),
        [] => Err(format!("no SASS section for {}", function.name).into()),
        _ => Err(format!("ambiguous SASS section for {}", function.name).into()),
    }
}

fn bounded(path: &Path) -> Result<File, Box<dyn Error>> {
    let file = File::open(path)?;
    let bytes = file.metadata()?.len();
    if bytes == 0 || bytes > MAX_BYTES {
        return Err(format!("{} has invalid bounded size {bytes}", path.display()).into());
    }
    Ok(file)
}

fn hex(value: &str) -> Result<u64, Box<dyn Error>> {
    Ok(u64::from_str_radix(
        value.strip_prefix("0x").unwrap_or(value),
        16,
    )?)
}

fn le32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn le64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pc_parser_accepts_prefix() {
        assert_eq!(hex("0x120").unwrap(), 0x120);
    }

    #[test]
    fn option_allowlist_is_closed() {
        assert!(known_option("--events"));
        assert!(!known_option("--accept-anything"));
    }
}

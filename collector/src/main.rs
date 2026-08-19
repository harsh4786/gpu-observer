use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use gpu_observer_collector::compact::correlate_checked;
use gpu_observer_collector::{parse_jsonl, ObserverError, Result};

fn main() {
    if let Err(error) = run() {
        eprintln!("gpu-observer-collector: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut arguments = std::env::args().skip(1);
    let Some(command) = arguments.next() else {
        return Err(usage());
    };
    let options = Options::parse(arguments)?;

    let raw = fs::read_to_string(&options.input)?;
    if let Some(raw_copy) = &options.raw_copy {
        atomic_write(raw_copy, raw.as_bytes())?;
    }
    let events = parse_jsonl(&raw)?;

    match command.as_str() {
        "validate" => {
            println!(
                "valid events={} bytes={} source={}",
                events.len(),
                raw.len(),
                options.input.display()
            );
            Ok(())
        }
        "correlate" => {
            let report_path = options.report.ok_or_else(usage)?;
            let report = correlate_checked(&events)?;
            let encoded = serde_json::to_vec_pretty(&report).map_err(|source| {
                ObserverError::InvalidEvent {
                    event_id: "collector".to_owned(),
                    message: format!("unable to serialize report: {source}"),
                }
            })?;
            atomic_write(&report_path, &encoded)?;
            println!(
                "correlated events={} steps={} requests={} diagnostics={} index_bytes={} report={}",
                report.input_event_count,
                report.steps.len(),
                report.requests.len(),
                report.diagnostics.len(),
                report.core_index_bytes,
                report_path.display()
            );
            Ok(())
        }
        _ => Err(usage()),
    }
}

struct Options {
    input: PathBuf,
    report: Option<PathBuf>,
    raw_copy: Option<PathBuf>,
}

impl Options {
    fn parse(arguments: impl Iterator<Item = String>) -> Result<Self> {
        let mut input = None;
        let mut report = None;
        let mut raw_copy = None;
        let mut arguments = arguments.peekable();

        while let Some(argument) = arguments.next() {
            let target = match argument.as_str() {
                "--input" => &mut input,
                "--report" => &mut report,
                "--raw-copy" => &mut raw_copy,
                "--help" | "-h" => return Err(usage()),
                _ => return Err(usage()),
            };
            let value = arguments.next().ok_or_else(usage)?;
            *target = Some(PathBuf::from(value));
        }

        Ok(Self {
            input: input.ok_or_else(usage)?,
            report,
            raw_copy,
        })
    }
}

fn atomic_write(path: &Path, contents: &[u8]) -> Result<()> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| ObserverError::Usage("output path needs a file name".to_owned()))?;
    let temporary = path.with_file_name(format!(".{file_name}.{}.tmp", std::process::id()));
    let mut output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    output.write_all(contents)?;
    output.flush()?;
    drop(output);
    fs::rename(temporary, path)?;
    Ok(())
}

fn usage() -> ObserverError {
    ObserverError::Usage(
        "usage: gpu-observer-collector <validate|correlate> --input TRACE.jsonl \
         [--report REPORT.json] [--raw-copy RAW.jsonl]"
            .to_owned(),
    )
}

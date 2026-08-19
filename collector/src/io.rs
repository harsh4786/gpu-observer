use std::fs;
use std::path::Path;

use crate::error::{ObserverError, Result};
use crate::event::Event;

pub fn parse_jsonl(contents: &str) -> Result<Vec<Event>> {
    contents
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| {
            let event: Event =
                serde_json::from_str(line).map_err(|source| ObserverError::Json {
                    line: index + 1,
                    source,
                })?;
            event.validate()?;
            Ok(event)
        })
        .collect()
}

pub fn read_jsonl(path: impl AsRef<Path>) -> Result<(String, Vec<Event>)> {
    let contents = fs::read_to_string(path)?;
    let events = parse_jsonl(&contents)?;
    Ok((contents, events))
}

#[cfg(test)]
mod tests {
    use super::parse_jsonl;

    #[test]
    fn reports_the_physical_line_for_invalid_json() {
        let error = parse_jsonl("\n{}\n").unwrap_err().to_string();
        assert!(error.contains("line 2"), "unexpected error: {error}");
    }
}

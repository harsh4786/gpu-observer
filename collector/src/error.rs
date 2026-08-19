use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum ObserverError {
    Io(std::io::Error),
    Json {
        line: usize,
        source: serde_json::Error,
    },
    InvalidEvent {
        event_id: String,
        message: String,
    },
    Usage(String),
}

impl Display for ObserverError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "I/O error: {error}"),
            Self::Json { line, source } => {
                write!(formatter, "invalid JSON event on line {line}: {source}")
            }
            Self::InvalidEvent { event_id, message } => {
                write!(formatter, "invalid event {event_id}: {message}")
            }
            Self::Usage(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for ObserverError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json { source, .. } => Some(source),
            Self::InvalidEvent { .. } | Self::Usage(_) => None,
        }
    }
}

impl From<std::io::Error> for ObserverError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

pub type Result<T> = std::result::Result<T, ObserverError>;

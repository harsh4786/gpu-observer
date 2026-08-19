//! Bounded MessagePack decoder for cold-path query manifests.
//!
//! The serving hot path emits MessagePack to avoid JSON construction. This
//! decoder runs only after capture, when building a sealed visualization bundle.

use std::fmt::{Display, Formatter};

use serde_json::{Map, Number, Value};

pub const MAX_MESSAGEPACK_BYTES: usize = 1024 * 1024;
const MAX_DEPTH: usize = 32;
const MAX_CONTAINER_ITEMS: usize = 65_536;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeError {
    pub offset: usize,
    pub message: &'static str,
}

impl Display for DecodeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "invalid MessagePack at byte {}: {}",
            self.offset, self.message
        )
    }
}

impl std::error::Error for DecodeError {}

pub fn decode(input: &[u8]) -> Result<Value, DecodeError> {
    if input.is_empty() {
        return Err(DecodeError {
            offset: 0,
            message: "empty input",
        });
    }
    if input.len() > MAX_MESSAGEPACK_BYTES {
        return Err(DecodeError {
            offset: 0,
            message: "input exceeds the 1 MiB bound",
        });
    }

    let mut decoder = Decoder { input, cursor: 0 };
    let value = decoder.value(0)?;
    if decoder.cursor != input.len() {
        return Err(decoder.error("trailing bytes"));
    }
    Ok(value)
}

struct Decoder<'a> {
    input: &'a [u8],
    cursor: usize,
}

impl Decoder<'_> {
    fn error(&self, message: &'static str) -> DecodeError {
        DecodeError {
            offset: self.cursor,
            message,
        }
    }

    fn byte(&mut self) -> Result<u8, DecodeError> {
        let value = self
            .input
            .get(self.cursor)
            .copied()
            .ok_or_else(|| self.error("unexpected end of input"))?;
        self.cursor += 1;
        Ok(value)
    }

    fn bytes(&mut self, length: usize) -> Result<&[u8], DecodeError> {
        let end = self
            .cursor
            .checked_add(length)
            .ok_or_else(|| self.error("length overflow"))?;
        let value = self
            .input
            .get(self.cursor..end)
            .ok_or_else(|| self.error("unexpected end of input"))?;
        self.cursor = end;
        Ok(value)
    }

    fn unsigned(&mut self, bytes: usize) -> Result<u64, DecodeError> {
        let raw = self.bytes(bytes)?;
        Ok(raw
            .iter()
            .fold(0_u64, |value, byte| (value << 8) | u64::from(*byte)))
    }

    fn signed(&mut self, bytes: usize) -> Result<i64, DecodeError> {
        let value = self.unsigned(bytes)?;
        let shift = 64 - bytes * 8;
        Ok(((value << shift) as i64) >> shift)
    }

    fn length(&mut self, bytes: usize) -> Result<usize, DecodeError> {
        usize::try_from(self.unsigned(bytes)?).map_err(|_| self.error("length does not fit usize"))
    }

    fn value(&mut self, depth: usize) -> Result<Value, DecodeError> {
        if depth > MAX_DEPTH {
            return Err(self.error("nesting exceeds 32 levels"));
        }

        let marker = self.byte()?;
        match marker {
            0x00..=0x7f => Ok(Value::Number(Number::from(marker))),
            0x80..=0x8f => self.map(usize::from(marker & 0x0f), depth + 1),
            0x90..=0x9f => self.array(usize::from(marker & 0x0f), depth + 1),
            0xa0..=0xbf => self.string(usize::from(marker & 0x1f)),
            0xc0 => Ok(Value::Null),
            0xc2 => Ok(Value::Bool(false)),
            0xc3 => Ok(Value::Bool(true)),
            0xca => {
                let bits =
                    u32::try_from(self.unsigned(4)?).map_err(|_| self.error("invalid float32"))?;
                self.float(f64::from(f32::from_bits(bits)))
            }
            0xcb => {
                let bits = self.unsigned(8)?;
                self.float(f64::from_bits(bits))
            }
            0xcc => self.unsigned_value(1),
            0xcd => self.unsigned_value(2),
            0xce => self.unsigned_value(4),
            0xcf => self.unsigned_value(8),
            0xd0 => self.signed_value(1),
            0xd1 => self.signed_value(2),
            0xd2 => self.signed_value(4),
            0xd3 => self.signed_value(8),
            0xd9 => {
                let length = self.length(1)?;
                self.string(length)
            }
            0xda => {
                let length = self.length(2)?;
                self.string(length)
            }
            0xdb => {
                let length = self.length(4)?;
                self.string(length)
            }
            0xdc => {
                let length = self.length(2)?;
                self.array(length, depth + 1)
            }
            0xdd => {
                let length = self.length(4)?;
                self.array(length, depth + 1)
            }
            0xde => {
                let length = self.length(2)?;
                self.map(length, depth + 1)
            }
            0xdf => {
                let length = self.length(4)?;
                self.map(length, depth + 1)
            }
            0xe0..=0xff => Ok(Value::Number(Number::from(i64::from(marker as i8)))),
            _ => Err(self.error("unsupported MessagePack marker")),
        }
    }

    fn float(&self, value: f64) -> Result<Value, DecodeError> {
        Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| self.error("non-finite floating point value"))
    }

    fn unsigned_value(&mut self, bytes: usize) -> Result<Value, DecodeError> {
        Ok(Value::Number(Number::from(self.unsigned(bytes)?)))
    }

    fn signed_value(&mut self, bytes: usize) -> Result<Value, DecodeError> {
        Ok(Value::Number(Number::from(self.signed(bytes)?)))
    }

    fn string(&mut self, length: usize) -> Result<Value, DecodeError> {
        if length > MAX_MESSAGEPACK_BYTES {
            return Err(self.error("string exceeds the 1 MiB bound"));
        }
        let string_offset = self.cursor;
        let bytes = self.bytes(length)?;
        let string = std::str::from_utf8(bytes).map_err(|_| DecodeError {
            offset: string_offset,
            message: "string is not valid UTF-8",
        })?;
        Ok(Value::String(string.to_owned()))
    }

    fn array(&mut self, length: usize, depth: usize) -> Result<Value, DecodeError> {
        if length > MAX_CONTAINER_ITEMS {
            return Err(self.error("array exceeds the item bound"));
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(length)
            .map_err(|_| self.error("array allocation failed"))?;
        for _ in 0..length {
            values.push(self.value(depth)?);
        }
        Ok(Value::Array(values))
    }

    fn map(&mut self, length: usize, depth: usize) -> Result<Value, DecodeError> {
        if length > MAX_CONTAINER_ITEMS {
            return Err(self.error("map exceeds the item bound"));
        }
        let mut values = Map::new();
        for _ in 0..length {
            let key = match self.value(depth)? {
                Value::String(key) => key,
                _ => return Err(self.error("map key is not a string")),
            };
            if values.contains_key(&key) {
                return Err(self.error("duplicate map key"));
            }
            values.insert(key, self.value(depth)?);
        }
        Ok(Value::Object(values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_query_manifest_shapes() {
        let encoded = [
            0x83, 0xa6, b's', b'c', b'h', b'e', b'm', b'a', 0xb5, b'G', b'P', b'U', b'_', b'O',
            b'B', b'S', b'E', b'R', b'V', b'E', b'R', b'_', b'Q', b'U', b'E', b'R', b'Y', b'_',
            b'0', b'1', 0xa3, b'i', b'd', b's', 0x93, 0x01, 0xcd, 0x03, 0xe8, 0xff, 0xa2, b'o',
            b'k', 0xc3,
        ];
        let value = decode(&encoded).unwrap();
        assert_eq!(value["schema"], "GPU_OBSERVER_QUERY_01");
        assert_eq!(value["ids"][1], 1000);
        assert_eq!(value["ids"][2], -1);
        assert_eq!(value["ok"], true);
    }

    #[test]
    fn rejects_trailing_and_binary_data() {
        assert!(decode(&[0xc0, 0xc0]).is_err());
        assert!(decode(&[0xc4, 0x00]).is_err());
    }
}

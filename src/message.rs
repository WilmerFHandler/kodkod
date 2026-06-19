use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{ToolCall, ToolResult};

/// A single image attachment in a user message.
///
/// Images are stored as raw bytes plus a MIME type; providers that support
/// vision are expected to encode them as base64 data URLs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Image {
    mime: String,
    #[serde(with = "base64_bytes")]
    data: Vec<u8>,
}

impl Image {
    pub fn new(mime: impl Into<String>, data: impl Into<Vec<u8>>) -> Self {
        Self {
            mime: mime.into(),
            data: data.into(),
        }
    }

    pub fn mime(&self) -> &str {
        &self.mime
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Encode the image as a base64 data URL.
    pub fn to_data_url(&self) -> String {
        format!(
            "data:{};base64,{}",
            self.mime,
            base64_bytes::encode(&self.data)
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserMessage {
    content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    images: Vec<Image>,
}

impl UserMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            images: Vec::new(),
        }
    }

    pub fn with_images(mut self, images: Vec<Image>) -> Self {
        self.images = images;
        self
    }

    pub fn content(&self) -> &str {
        &self.content
    }

    pub fn images(&self) -> &[Image] {
        &self.images
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssistantMessage {
    content: String,
    tool_calls: Vec<ToolCall>,
}

impl AssistantMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            tool_calls: Vec::new(),
        }
    }

    pub fn content(&self) -> &str {
        &self.content
    }

    pub fn tool_calls(&self) -> &[ToolCall] {
        &self.tool_calls
    }

    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.tool_calls = tool_calls;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemMessage {
    content: String,
}

impl SystemMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
        }
    }

    pub fn content(&self) -> &str {
        &self.content
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "role")]
pub enum Message {
    #[serde(rename = "system")]
    System(SystemMessage),
    #[serde(rename = "user")]
    User(UserMessage),
    #[serde(rename = "assistant")]
    Assistant(AssistantMessage),
    #[serde(rename = "tool")]
    ToolResult(ToolResult),
}

mod base64_bytes {
    use super::*;

    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    pub fn serialize<S: Serializer>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&encode(bytes))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
        let text = String::deserialize(deserializer)?;
        decode(&text).map_err(serde::de::Error::custom)
    }

    pub fn encode(input: &[u8]) -> String {
        let mut out = String::with_capacity((input.len() * 4 + 2) / 3);
        for chunk in input.chunks(3) {
            let mut buf = [0u8; 3];
            for (i, byte) in chunk.iter().enumerate() {
                buf[i] = *byte;
            }
            let triple = ((buf[0] as u32) << 16) | ((buf[1] as u32) << 8) | (buf[2] as u32);
            out.push(ALPHABET[((triple >> 18) & 0x3F) as usize] as char);
            out.push(ALPHABET[((triple >> 12) & 0x3F) as usize] as char);
            if chunk.len() > 1 {
                out.push(ALPHABET[((triple >> 6) & 0x3F) as usize] as char);
            } else {
                out.push('=');
            }
            if chunk.len() > 2 {
                out.push(ALPHABET[(triple & 0x3F) as usize] as char);
            } else {
                out.push('=');
            }
        }
        out
    }

    pub fn decode(input: &str) -> Result<Vec<u8>, String> {
        let mut data = input.bytes().filter(|&b| b != b'=').peekable();
        let mut out = Vec::with_capacity((input.len() * 3) / 4);
        while data.peek().is_some() {
            let mut sextets = [0u8; 4];
            for i in 0..4 {
                let byte = data
                    .next()
                    .ok_or_else(|| "invalid base64 length".to_string())?;
                sextets[i] = decode_char(byte)?;
            }
            let triple = ((sextets[0] as u32) << 18)
                | ((sextets[1] as u32) << 12)
                | ((sextets[2] as u32) << 6)
                | (sextets[3] as u32);
            out.push(((triple >> 16) & 0xFF) as u8);
            out.push(((triple >> 8) & 0xFF) as u8);
            out.push((triple & 0xFF) as u8);
        }
        let padding = input.bytes().filter(|&b| b == b'=').count();
        out.truncate(out.len().saturating_sub(padding));
        Ok(out)
    }

    fn decode_char(byte: u8) -> Result<u8, String> {
        match byte {
            b'A'..=b'Z' => Ok(byte - b'A'),
            b'a'..=b'z' => Ok(byte - b'a' + 26),
            b'0'..=b'9' => Ok(byte - b'0' + 52),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => Err(format!("invalid base64 character: {}", byte as char)),
        }
    }
}

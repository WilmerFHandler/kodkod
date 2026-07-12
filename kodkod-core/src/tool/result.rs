use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::ops::Deref;

use crate::{Image, ToolExecutorError};

/// Structured output produced by a tool.
///
/// `value` is the ordinary JSON result presented to every model. Images are
/// kept separate so vision-capable providers can encode them as native
/// multimodal input instead of burying base64 data in text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolOutput {
    value: Value,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    images: Vec<Image>,
}

impl ToolOutput {
    pub fn new(value: Value) -> Self {
        Self {
            value,
            images: Vec::new(),
        }
    }

    pub fn with_images(mut self, images: Vec<Image>) -> Self {
        self.images = images;
        self
    }

    pub fn value(&self) -> &Value {
        &self.value
    }

    pub fn images(&self) -> &[Image] {
        &self.images
    }

    pub fn without_images(&self) -> Self {
        Self::new(self.value.clone())
    }
}

impl From<Value> for ToolOutput {
    fn from(value: Value) -> Self {
        Self::new(value)
    }
}

impl Deref for ToolOutput {
    type Target = Value;

    fn deref(&self) -> &Self::Target {
        self.value()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResult {
    tool_call_id: String,
    outcome: ToolResultOutcome,
}

impl ToolResult {
    pub fn new(tool_call_id: impl Into<String>, outcome: ToolResultOutcome) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            outcome,
        }
    }

    pub fn success(tool_call_id: impl Into<String>, output: impl Into<ToolOutput>) -> Self {
        Self::new(tool_call_id, ToolResultOutcome::Success(output.into()))
    }

    pub fn failure(tool_call_id: impl Into<String>, error: ToolExecutorError) -> Self {
        Self::new(tool_call_id, ToolResultOutcome::Error(error))
    }

    pub fn tool_call_id(&self) -> &str {
        &self.tool_call_id
    }

    pub fn outcome(&self) -> &ToolResultOutcome {
        &self.outcome
    }

    pub fn value(&self) -> Option<&Value> {
        match &self.outcome {
            ToolResultOutcome::Success(output) => Some(output.value()),
            ToolResultOutcome::Error(_) => None,
        }
    }

    pub fn error(&self) -> Option<&ToolExecutorError> {
        match &self.outcome {
            ToolResultOutcome::Success(_) => None,
            ToolResultOutcome::Error(error) => Some(error),
        }
    }

    pub fn without_images(&self) -> Self {
        match &self.outcome {
            ToolResultOutcome::Success(output) => {
                Self::success(self.tool_call_id.clone(), output.without_images())
            }
            ToolResultOutcome::Error(error) => {
                Self::failure(self.tool_call_id.clone(), error.clone())
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolResultOutcome {
    Success(ToolOutput),
    Error(ToolExecutorError),
}

impl Serialize for ToolResultOutcome {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Success(output) if output.images().is_empty() => {
                ToolResultOutcomeRef::Success(output.value()).serialize(serializer)
            }
            Self::Success(output) => {
                ToolResultOutcomeRef::SuccessWithImages(output).serialize(serializer)
            }
            Self::Error(error) => ToolResultOutcomeRef::Error(error).serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for ToolResultOutcome {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        match ToolResultOutcomeWire::deserialize(deserializer)? {
            ToolResultOutcomeWire::Success(value) => Ok(Self::Success(value.into())),
            ToolResultOutcomeWire::SuccessWithImages(output) => Ok(Self::Success(output)),
            ToolResultOutcomeWire::Error(error) => Ok(Self::Error(error)),
        }
    }
}

#[derive(Serialize)]
#[serde(tag = "type", content = "value")]
enum ToolResultOutcomeRef<'a> {
    #[serde(rename = "success")]
    Success(&'a Value),
    #[serde(rename = "success_with_images")]
    SuccessWithImages(&'a ToolOutput),
    #[serde(rename = "error")]
    Error(&'a ToolExecutorError),
}

#[derive(Deserialize)]
#[serde(tag = "type", content = "value")]
enum ToolResultOutcomeWire {
    #[serde(rename = "success")]
    Success(Value),
    #[serde(rename = "success_with_images")]
    SuccessWithImages(ToolOutput),
    #[serde(rename = "error")]
    Error(ToolExecutorError),
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn json_only_output_keeps_the_legacy_wire_shape() {
        let result = ToolResult::success("call_1", json!({"answer": 42}));
        let encoded = serde_json::to_value(&result).unwrap();

        assert_eq!(encoded["outcome"]["value"], json!({"answer": 42}));
        assert_eq!(
            serde_json::from_value::<ToolResult>(encoded).unwrap(),
            result
        );
    }

    #[test]
    fn image_output_round_trips_with_a_versioned_wire_shape() {
        let output = ToolOutput::new(json!({"path": "image.png"}))
            .with_images(vec![Image::new("image/png", b"png")]);
        let result = ToolResult::success("call_1", output);
        let encoded = serde_json::to_value(&result).unwrap();

        assert_eq!(encoded["outcome"]["type"], "success_with_images");
        assert_eq!(encoded["outcome"]["value"]["value"]["path"], "image.png");
        assert_eq!(
            serde_json::from_value::<ToolResult>(encoded).unwrap(),
            result
        );
    }

    #[test]
    fn legacy_output_cannot_be_confused_with_the_versioned_shape() {
        let value = json!({
            "__kodkod_tool_output": 1,
            "value": {"application_data": true},
            "images": [{"mime": "image/png", "data": "cG5n"}]
        });
        let encoded = json!({
            "tool_call_id": "call_1",
            "outcome": {"type": "success", "value": value}
        });
        let result = serde_json::from_value::<ToolResult>(encoded).unwrap();

        assert_eq!(result.value(), Some(&value));
        assert!(
            matches!(result.outcome(), ToolResultOutcome::Success(output) if output.images().is_empty())
        );
    }
}

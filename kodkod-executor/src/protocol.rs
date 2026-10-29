//! Wire contracts for a server-owned worker borrowing a client executor.
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Version 3 adds trusted per-operation execution context.
pub const EXECUTOR_PROTOCOL_HEADER: &str = "x-kodkod-executor-protocol";
pub const EXECUTOR_PROTOCOL_VERSION: u16 = 3;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DeviceCapability {
    Commands,
    AppAutomation,
    Desktop,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceHello {
    /// Installation identity; never an execution or consent capability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executor_id: Option<uuid::Uuid>,
    pub grant_generation: u64,
    pub name: String,
    pub capabilities: Vec<DeviceCapability>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceConnection {
    pub connection: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DevicePoll {
    pub grant_generation: u64,
    pub active_attempt: Option<String>,
    pub ready: bool,
    pub connection: String,
    pub capabilities: Vec<DeviceCapability>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DevicePollResult {
    pub operation: Option<DeviceOperation>,
    pub active_authorized: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceOperation {
    /// Trusted controller metadata, separate from model-supplied action arguments.
    pub context: Value,
    pub attempt: String,
    pub attachment: String,
    pub operation: DeviceAction,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum DeviceAction {
    RunCommand {
        command: String,
        cwd: String,
    },
    Screenshot {},
    ComputerInput {
        observation_id: String,
        input: crate::computer::ComputerInput,
    },
    ListApps {},
    ObserveApp {
        request: crate::computer::ObserveAppRequest,
    },
    ActOnApp {
        request: crate::computer::AppActionRequest,
    },
}
impl DeviceAction {
    pub fn capability(&self) -> DeviceCapability {
        match self {
            Self::RunCommand { .. } => DeviceCapability::Commands,
            Self::Screenshot {} | Self::ComputerInput { .. } => DeviceCapability::Desktop,
            _ => DeviceCapability::AppAutomation,
        }
    }
    pub fn validate(&self) -> anyhow::Result<()> {
        match self {
            Self::RunCommand { command, cwd } => {
                anyhow::ensure!(
                    !command.is_empty() && command.len() <= 64 * 1024,
                    "Command is empty or too large"
                );
                anyhow::ensure!(
                    is_absolute_target_path(cwd) && cwd.len() <= 4096 && !cwd.contains('\0'),
                    "Working directory must be an absolute path"
                );
                anyhow::ensure!(!command.contains('\0'), "Invalid command");
                Ok(())
            }
            Self::ObserveApp { request } => request.validate(),
            Self::ActOnApp { request } => request.validate(),
            Self::Screenshot {} | Self::ListApps {} => Ok(()),
            Self::ComputerInput {
                observation_id,
                input,
            } => {
                anyhow::ensure!(
                    !observation_id.is_empty() && observation_id.len() <= 128,
                    "Invalid desktop observation"
                );
                input.validate()
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceReceipt {
    pub connection: String,
    pub attempt: String,
    pub result: Result<Value, String>,
}

/// Validate wire syntax without applying the controller machine's path rules.
/// Existence, permissions and native semantics are checked at the executor.
pub fn is_absolute_target_path(path: &str) -> bool {
    let b = path.as_bytes();
    path.starts_with('/')
        || (b.len() >= 3
            && b[0].is_ascii_alphabetic()
            && b[1] == b':'
            && matches!(b[2], b'/' | b'\\'))
        || (path.starts_with("\\\\")
            && path[2..].split('\\').filter(|p| !p.is_empty()).count() >= 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn target_paths_do_not_depend_on_the_controller_platform() {
        for path in [
            "/home/worker",
            "C:\\Users\\worker",
            "C:/Users/worker",
            "\\\\host\\share",
        ] {
            assert!(is_absolute_target_path(path), "{path}");
        }
        for path in ["relative", "C:relative", "", "\\relative"] {
            assert!(!is_absolute_target_path(path), "{path}");
        }
    }
}

use anyhow::{Result, bail};
use base64::Engine;
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::HashSet, sync::Arc, time::Duration};
use uuid::Uuid;

/// Keep one screenshot from consuming an unbounded amount of model/tool memory.
pub const MAX_SCREENSHOT_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_SCREENSHOT_PIXELS: u64 = 16 * 1024 * 1024;
pub const LOCAL_APP_AUTOMATION_UNSUPPORTED: &str = "Local app automation is unavailable on this platform; use a macOS local session or a remote session.";

pub const MAX_LOCAL_APPS: usize = 64;
pub const MAX_LOCAL_APP_ELEMENTS: usize = 200;
pub const MAX_LOCAL_APP_NAME_BYTES: usize = 512;
pub const MAX_LOCAL_APP_BUNDLE_ID_BYTES: usize = 512;
pub const MAX_LOCAL_APP_WINDOW_TITLE_BYTES: usize = 512;
pub const MAX_LOCAL_APP_ROLE_BYTES: usize = 128;
pub const MAX_LOCAL_APP_LABEL_BYTES: usize = 1024;
pub const MAX_LOCAL_APP_ACTION_BYTES: usize = 64;
pub const MAX_LOCAL_APP_ACTIONS: usize = 2;
pub const MAX_LOCAL_APP_TEXT_BYTES: usize = 8192;
pub const APP_OBSERVATION_MAX_AGE: Duration = Duration::from_secs(120);

/// The permission states needed by a native desktop adapter. Remote guests use
/// `NotRequired`; macOS reports the two system permissions independently.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DesktopPermissionStatus {
    Granted,
    Required,
    NotRequired,
    Unsupported,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DesktopPermissions {
    pub screen_recording: DesktopPermissionStatus,
    pub accessibility: DesktopPermissionStatus,
}

impl DesktopPermissions {
    pub const fn not_required() -> Self {
        Self {
            screen_recording: DesktopPermissionStatus::NotRequired,
            accessibility: DesktopPermissionStatus::NotRequired,
        }
    }

    pub const fn unsupported() -> Self {
        Self {
            screen_recording: DesktopPermissionStatus::Unsupported,
            accessibility: DesktopPermissionStatus::Unsupported,
        }
    }

    pub fn onboarding_message(self) -> Option<&'static str> {
        match (self.screen_recording, self.accessibility) {
            (DesktopPermissionStatus::Required, DesktopPermissionStatus::Required) => Some(
                "This application needs Screen Recording and Accessibility access. Open System Settings > Privacy & Security, enable this application in both sections, then restart it.",
            ),
            (DesktopPermissionStatus::Required, _) => Some(
                "This application needs Screen Recording access. Open System Settings > Privacy & Security > Screen Recording, enable this application, then restart it.",
            ),
            (_, DesktopPermissionStatus::Required) => Some(
                "This application needs Accessibility access. Open System Settings > Privacy & Security > Accessibility, enable this application, then restart it.",
            ),
            _ => None,
        }
    }
}

/// A validated PNG frame returned by a desktop backend.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DesktopScreenshot {
    png: Vec<u8>,
    width: u32,
    height: u32,
}

/// A bounded list of applications visible to a local app-automation adapter.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AppList {
    pub apps: Vec<AppSummary>,
}

impl AppList {
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.apps.len() <= MAX_LOCAL_APPS,
            "The app list exceeds the limit of {MAX_LOCAL_APPS} entries"
        );
        for app in &self.apps {
            app.validate()?;
        }
        anyhow::ensure!(
            self.apps
                .iter()
                .map(|app| app.id.as_str())
                .collect::<HashSet<_>>()
                .len()
                == self.apps.len(),
            "The app list contains duplicate ids"
        );
        Ok(())
    }
}

/// Stable metadata for one locally visible application. The adapter owns the
/// opaque UUID handle and decides which applications are safe to expose.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AppSummary {
    pub id: String,
    pub bundle_id: String,
    pub name: String,
}

impl AppSummary {
    pub fn validate(&self) -> Result<()> {
        validate_app_id(&self.id, "App id")?;
        validate_text(
            &self.bundle_id,
            MAX_LOCAL_APP_BUNDLE_ID_BYTES,
            "App bundle id",
            false,
        )?;
        validate_text(&self.name, MAX_LOCAL_APP_NAME_BYTES, "App name", false)?;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ObserveAppRequest {
    pub app_id: String,
    pub window_title: Option<String>,
}

impl ObserveAppRequest {
    pub fn validate(&self) -> Result<()> {
        validate_app_id(&self.app_id, "App id")?;
        if let Some(window_title) = &self.window_title {
            validate_text(
                window_title,
                MAX_LOCAL_APP_WINDOW_TITLE_BYTES,
                "Window title",
                true,
            )?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ObservedAppTarget {
    pub bundle_id: String,
    pub name: String,
    pub window_title: String,
}

impl ObservedAppTarget {
    pub(crate) fn validate(&self) -> Result<()> {
        validate_text(
            &self.bundle_id,
            MAX_LOCAL_APP_BUNDLE_ID_BYTES,
            "Observed app bundle id",
            false,
        )?;
        validate_text(
            &self.name,
            MAX_LOCAL_APP_NAME_BYTES,
            "Observed app name",
            false,
        )?;
        validate_text(
            &self.window_title,
            MAX_LOCAL_APP_WINDOW_TITLE_BYTES,
            "Observed window title",
            true,
        )?;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ObservedAppElement {
    pub id: String,
    pub role: String,
    pub label: String,
    pub enabled: bool,
    pub editable: bool,
    pub actions: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
}

impl ObservedAppElement {
    pub(crate) fn validate(&self) -> Result<()> {
        validate_app_id(&self.id, "App element id")?;
        validate_text(&self.role, MAX_LOCAL_APP_ROLE_BYTES, "Element role", false)?;
        validate_text(
            &self.label,
            MAX_LOCAL_APP_LABEL_BYTES,
            "Element label",
            true,
        )?;
        if let Some(value) = &self.value {
            anyhow::ensure!(
                value.len() <= MAX_LOCAL_APP_TEXT_BYTES,
                "Element value is too long"
            );
        }
        anyhow::ensure!(
            self.actions.len() <= MAX_LOCAL_APP_ACTIONS,
            "An app element may advertise at most {MAX_LOCAL_APP_ACTIONS} actions"
        );
        for action in &self.actions {
            validate_text(action, MAX_LOCAL_APP_ACTION_BYTES, "Element action", false)?;
            anyhow::ensure!(
                matches!(action.as_str(), "press" | "set_text"),
                "Unsupported app element action"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AppObservation {
    pub id: String,
    pub target: ObservedAppTarget,
    pub elements: Vec<ObservedAppElement>,
    pub screenshot: Option<DesktopScreenshot>,
}

impl AppObservation {
    pub fn validate(&self) -> Result<()> {
        validate_app_id(&self.id, "App observation id")?;
        self.target.validate()?;
        anyhow::ensure!(
            self.elements.len() <= MAX_LOCAL_APP_ELEMENTS,
            "The app observation exceeds the limit of {MAX_LOCAL_APP_ELEMENTS} elements"
        );
        for element in &self.elements {
            element.validate()?;
        }
        anyhow::ensure!(
            self.elements
                .iter()
                .map(|element| element.id.as_str())
                .collect::<HashSet<_>>()
                .len()
                == self.elements.len(),
            "The app observation contains duplicate element ids"
        );
        Ok(())
    }

    /// Bounded JSON metadata for a tool result. Screenshot bytes remain a
    /// native image attachment and never enter this value.
    pub fn metadata(&self) -> Value {
        json!({
            "observation_id": &self.id,
            "target": &self.target,
            "elements": &self.elements,
            "screenshot": self.screenshot.as_ref().map(DesktopScreenshot::metadata),
        })
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AppActionRequest {
    pub observation_id: String,
    pub element_id: String,
    pub action: AppAction,
}

impl AppActionRequest {
    pub fn validate(&self) -> Result<()> {
        validate_app_id(&self.observation_id, "App observation id")?;
        validate_app_id(&self.element_id, "App element id")?;
        self.action.validate()
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum AppAction {
    Press {},
    SetText { text: String },
}

impl AppAction {
    fn validate(&self) -> Result<()> {
        if let Self::SetText { text } = self {
            anyhow::ensure!(
                text.len() <= MAX_LOCAL_APP_TEXT_BYTES,
                "App text exceeds the limit of {MAX_LOCAL_APP_TEXT_BYTES} bytes"
            );
        }
        Ok(())
    }
}

fn validate_app_id(value: &str, label: &str) -> Result<()> {
    anyhow::ensure!(
        value.len() == 36 && Uuid::parse_str(value).is_ok(),
        "{label} must use canonical UUID syntax"
    );
    anyhow::ensure!(
        !value.chars().any(char::is_control),
        "{label} cannot contain control characters"
    );
    Ok(())
}

fn validate_text(value: &str, max_bytes: usize, label: &str, allow_empty: bool) -> Result<()> {
    anyhow::ensure!(allow_empty || !value.is_empty(), "{label} cannot be empty");
    anyhow::ensure!(value.len() <= max_bytes, "{label} is too long");
    anyhow::ensure!(
        !value.chars().any(char::is_control),
        "{label} cannot contain control characters"
    );
    Ok(())
}

impl DesktopScreenshot {
    pub fn from_png(png: Vec<u8>) -> Result<Self> {
        Self::from_png_limited(png, MAX_SCREENSHOT_BYTES)
    }

    pub fn from_png_limited(png: Vec<u8>, max_bytes: usize) -> Result<Self> {
        anyhow::ensure!(
            !png.is_empty() && png.len() <= max_bytes,
            "Desktop screenshot exceeds the size limit"
        );
        let (width, height) = Self::dimensions_from_png(&png)
            .ok_or_else(|| anyhow::anyhow!("Desktop backend returned an invalid PNG"))?;
        Ok(Self { png, width, height })
    }

    pub fn dimensions_from_png(png: &[u8]) -> Option<(u32, u32)> {
        const PNG_SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";
        if !png.starts_with(PNG_SIGNATURE) || png.len() < 24 {
            return None;
        }
        let width = u32::from_be_bytes(png[16..20].try_into().ok()?);
        let height = u32::from_be_bytes(png[20..24].try_into().ok()?);
        (width > 0
            && height > 0
            && u64::from(width).saturating_mul(u64::from(height)) <= MAX_SCREENSHOT_PIXELS)
            .then_some((width, height))
    }

    pub fn png(&self) -> &[u8] {
        &self.png
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Metadata is kept separate from image bytes so providers receive a
    /// vision attachment rather than a large base64 string in tool text.
    pub fn metadata(&self) -> Value {
        json!({
            "screenshot": true,
            "width": self.width,
            "height": self.height,
            "coordinate_space": "screenshot_pixels",
        })
    }

    /// Retain the wire representation for adapters that still speak the guest
    /// JSON protocol, where the image is carried as a bounded base64 field.
    pub fn into_tool_value(self) -> Value {
        json!({
            "png": base64::engine::general_purpose::STANDARD.encode(self.png),
            "width": self.width,
            "height": self.height,
            "coordinate_space": "screenshot_pixels",
        })
    }
}

/// Execution-specific desktop access. The runtime owns the stable contract;
/// platform adapters own permissions, process launch, and input details.
pub trait DesktopBackend: Send + Sync {
    fn permissions(&self) -> DesktopPermissions;

    fn screenshot<'a>(&'a self) -> BoxFuture<'a, Result<DesktopScreenshot>>;

    fn input<'a>(&'a self, input: ComputerInput) -> BoxFuture<'a, Result<()>>;
}

/// Platform-owned local app automation. Unlike the remote computer backend,
/// this contract exposes semantic application elements instead of coordinates
/// or generic keyboard events.
pub trait AppAutomationSession: Send + Sync {
    fn list_apps<'a>(&'a self) -> BoxFuture<'a, Result<AppList>>;

    fn observe_app<'a>(
        &'a self,
        request: ObserveAppRequest,
    ) -> BoxFuture<'a, Result<AppObservation>>;

    fn act_on_app<'a>(&'a self, request: AppActionRequest) -> BoxFuture<'a, Result<()>>;
}

/// Composition-root adapter for native local app automation. The runtime owns
/// leases and tool contracts; the adapter owns OS permissions and accessibility
/// handles.
pub trait AppAutomationBackend: Send + Sync {
    fn permissions(&self) -> DesktopPermissions;
    fn app_automation_session(&self) -> Arc<dyn AppAutomationSession>;
}

/// An actual desktop input, used by both the agent and the manual UI.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum ComputerInput {
    Click {
        x: u32,
        y: u32,
    },
    /// Positive deltas move the viewed content right/down; units are pixels.
    Scroll {
        x: u32,
        y: u32,
        delta_x: i32,
        delta_y: i32,
    },
    Type {
        text: String,
    },
    Key {
        key: String,
    },
}
impl ComputerInput {
    pub const KEYS: [&'static str; 16] = [
        "Return",
        "Tab",
        "Escape",
        "BackSpace",
        "Up",
        "Down",
        "Left",
        "Right",
        "ctrl+l",
        "ctrl+a",
        "ctrl+c",
        "ctrl+v",
        "cmd+l",
        "cmd+a",
        "cmd+c",
        "cmd+v",
    ];
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Click { x, y } | Self::Scroll { x, y, .. } if *x >= 8192 || *y >= 8192 => {
                bail!("Invalid desktop coordinates")
            }
            Self::Scroll {
                delta_x, delta_y, ..
            } if !(-4096..=4096).contains(delta_x)
                || !(-4096..=4096).contains(delta_y)
                || (*delta_x == 0 && *delta_y == 0) =>
            {
                bail!("Scroll deltas must be nonzero and within 4096 pixels")
            }
            Self::Type { text } if text.len() > 8192 => bail!("Text too long"),
            Self::Key { key } if !Self::KEYS.contains(&key.as_str()) => bail!("Unsupported key"),
            _ => Ok(()),
        }
    }
}

pub fn validate_click_against_frame(input: &ComputerInput, dimensions: (u32, u32)) -> Result<()> {
    let (x, y) = match input {
        ComputerInput::Click { x, y } | ComputerInput::Scroll { x, y, .. } => (x, y),
        _ => return Ok(()),
    };
    let (width, height) = dimensions;
    if *x >= width || *y >= height {
        bail!("Desktop coordinates ({x}, {y}) are outside the current frame ({width}x{height})");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn png(width: u32, height: u32) -> Vec<u8> {
        let mut bytes = b"\x89PNG\r\n\x1a\n".to_vec();
        bytes.extend_from_slice(&[0; 8]);
        bytes.extend_from_slice(&width.to_be_bytes());
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes
    }

    #[test]
    fn scrolling_is_bounded_and_uses_the_observed_region() {
        let input = ComputerInput::Scroll {
            x: 20,
            y: 10,
            delta_x: -250,
            delta_y: 600,
        };
        input.validate().unwrap();
        assert_eq!(serde_json::to_value(&input).unwrap()["action"], "scroll");
        assert!(validate_click_against_frame(&input, (100, 100)).is_ok());
        assert!(validate_click_against_frame(&input, (10, 100)).is_err());
        assert!(
            ComputerInput::Scroll {
                x: 0,
                y: 0,
                delta_x: 0,
                delta_y: 0
            }
            .validate()
            .is_err()
        );
        assert!(
            ComputerInput::Scroll {
                x: 0,
                y: 0,
                delta_x: i32::MIN,
                delta_y: 1
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn screenshot_contract_bounds_bytes_and_pixels() {
        let screenshot = DesktopScreenshot::from_png(png(1280, 720)).unwrap();
        assert_eq!(screenshot.dimensions(), (1280, 720));
        assert_eq!(
            screenshot.metadata()["coordinate_space"],
            "screenshot_pixels"
        );
        assert_eq!(
            screenshot.clone().into_tool_value()["coordinate_space"],
            "screenshot_pixels"
        );
        let mut oversized_png = png(1280, 720);
        oversized_png.push(0);
        assert!(DesktopScreenshot::from_png_limited(oversized_png, 24).is_err());
        assert!(DesktopScreenshot::from_png(png(0, 720)).is_err());
        assert!(DesktopScreenshot::from_png(png(5000, 5000)).is_err());
        assert!(DesktopScreenshot::from_png(b"not a png".to_vec()).is_err());
    }

    #[test]
    fn computer_input_preserves_wire_shape_and_limits() {
        for value in [
            json!({"action":"click","x":2}),
            json!({"action":"type","text":"hello","key":"Return"}),
            json!({"action":"shell","command":"echo test"}),
        ] {
            assert!(serde_json::from_value::<ComputerInput>(value).is_err());
        }
        assert_eq!(
            serde_json::to_value(ComputerInput::Click { x: 2, y: 3 }).unwrap(),
            json!({"action":"click","x":2,"y":3})
        );
        assert!(ComputerInput::Click { x: 8192, y: 0 }.validate().is_err());
        assert!(
            ComputerInput::Key {
                key: "--window".into()
            }
            .validate()
            .is_err()
        );
        assert!(
            ComputerInput::Type {
                text: "x".repeat(8193)
            }
            .validate()
            .is_err()
        );
        for key in [
            "ctrl+a", "ctrl+c", "ctrl+v", "ctrl+l", "cmd+a", "cmd+c", "cmd+v", "cmd+l",
        ] {
            assert!(ComputerInput::Key { key: key.into() }.validate().is_ok());
        }
    }

    #[test]
    fn click_validation_uses_current_frame_dimensions() {
        assert!(
            validate_click_against_frame(&ComputerInput::Click { x: 9, y: 4 }, (10, 5)).is_ok()
        );
        assert!(
            validate_click_against_frame(&ComputerInput::Click { x: 10, y: 4 }, (10, 5)).is_err()
        );
        assert!(
            validate_click_against_frame(&ComputerInput::Click { x: 0, y: 0 }, (0, 0)).is_err()
        );
        assert!(
            validate_click_against_frame(
                &ComputerInput::Key {
                    key: "Return".into()
                },
                (1, 1)
            )
            .is_ok()
        );
    }

    #[test]
    fn app_contracts_validate_ids_sizes_and_wire_shape() {
        let press: AppAction = serde_json::from_value(json!({"type":"press"})).unwrap();
        assert_eq!(press, AppAction::Press {});
        let set_text: AppAction =
            serde_json::from_value(json!({"type":"set_text","text":"hello"})).unwrap();
        assert_eq!(
            set_text,
            AppAction::SetText {
                text: "hello".into()
            }
        );
        assert!(serde_json::from_value::<AppAction>(json!({"type":"press","text":"x"})).is_err());
        assert!(serde_json::from_value::<AppAction>(json!({"type":"key","key":"Return"})).is_err());

        let bad_id = AppActionRequest {
            observation_id: "not-an-id".into(),
            element_id: Uuid::new_v4().to_string(),
            action: AppAction::Press {},
        };
        assert!(bad_id.validate().is_err());
        let oversized = AppActionRequest {
            observation_id: Uuid::new_v4().to_string(),
            element_id: Uuid::new_v4().to_string(),
            action: AppAction::SetText {
                text: "x".repeat(MAX_LOCAL_APP_TEXT_BYTES + 1),
            },
        };
        assert!(oversized.validate().is_err());

        let app_id = Uuid::new_v4().to_string();
        let duplicate_apps = AppList {
            apps: vec![
                AppSummary {
                    id: app_id.clone(),
                    bundle_id: "com.example.App".into(),
                    name: "App".into(),
                },
                AppSummary {
                    id: app_id,
                    bundle_id: "com.example.Other".into(),
                    name: "Other".into(),
                },
            ],
        };
        assert!(duplicate_apps.validate().is_err());

        let element = ObservedAppElement {
            id: Uuid::new_v4().to_string(),
            role: "button".into(),
            label: "Run".into(),
            enabled: true,
            editable: false,
            actions: vec!["press".into()],
            value: None,
        };
        let observation = AppObservation {
            id: Uuid::new_v4().to_string(),
            target: ObservedAppTarget {
                bundle_id: "com.example.App".into(),
                name: "App".into(),
                window_title: "Main".into(),
            },
            elements: vec![element; MAX_LOCAL_APP_ELEMENTS + 1],
            screenshot: None,
        };
        assert!(observation.validate().is_err());
    }

    #[test]
    fn permission_onboarding_uses_generic_application_wording() {
        let permissions = DesktopPermissions {
            screen_recording: DesktopPermissionStatus::Required,
            accessibility: DesktopPermissionStatus::Granted,
        };
        let message = permissions.onboarding_message().unwrap();
        assert_eq!(
            message,
            "This application needs Screen Recording access. Open System Settings > Privacy & Security > Screen Recording, enable this application, then restart it."
        );
    }
}

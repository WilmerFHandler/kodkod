use crate::computer::*;
#[cfg(target_os = "linux")]
use anyhow::Context;
use anyhow::Result;
#[cfg(not(target_os = "macos"))]
use futures::future::BoxFuture;
#[cfg(not(windows))]
use serde_json::Value;
#[cfg(target_os = "linux")]
use serde_json::json;
use std::sync::Arc;

pub fn app_backend() -> Arc<dyn AppAutomationBackend> {
    #[cfg(target_os = "macos")]
    {
        crate::macos::backend()
    }
    #[cfg(not(target_os = "macos"))]
    {
        Arc::new(UnavailableApps)
    }
}
#[cfg(not(target_os = "macos"))]
struct UnavailableApps;
#[cfg(not(target_os = "macos"))]
impl AppAutomationBackend for UnavailableApps {
    fn permissions(&self) -> DesktopPermissions {
        DesktopPermissions::unsupported()
    }
    fn app_automation_session(&self) -> Arc<dyn AppAutomationSession> {
        Arc::new(Self)
    }
}
#[cfg(not(target_os = "macos"))]
impl AppAutomationSession for UnavailableApps {
    fn list_apps<'a>(&'a self) -> BoxFuture<'a, Result<AppList>> {
        Box::pin(async { anyhow::bail!("Native app automation is unavailable on this endpoint") })
    }
    fn observe_app<'a>(&'a self, _: ObserveAppRequest) -> BoxFuture<'a, Result<AppObservation>> {
        Box::pin(async { anyhow::bail!("Native app automation is unavailable on this endpoint") })
    }
    fn act_on_app<'a>(&'a self, _: AppActionRequest) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { anyhow::bail!("Native app automation is unavailable on this endpoint") })
    }
}

#[cfg(target_os = "macos")]
#[derive(Default)]
pub struct Desktop(Arc<crate::macos::desktop::Desktop>);
#[cfg(target_os = "macos")]
impl Desktop {
    pub fn invalidate(&self) {
        self.0.invalidate();
    }
    pub async fn screenshot(&self) -> Result<Value> {
        let native = self.0.clone();
        tokio::task::spawn_blocking(move || native.screenshot()).await?
    }
    pub async fn input(
        &self,
        id: &str,
        input: ComputerInput,
        authorized: impl Fn() -> bool + Send + 'static,
    ) -> Result<Value> {
        let native = self.0.clone();
        let id = id.to_owned();
        tokio::task::spawn_blocking(move || native.input(&id, input, authorized)).await?
    }
}

#[cfg(target_os = "linux")]
struct Observation {
    id: String,
    dimensions: (u32, u32),
    at: std::time::Instant,
}
#[cfg(target_os = "linux")]
pub struct Desktop {
    backend: crate::linux::LinuxDesktopBackend,
    last: std::sync::Mutex<Option<Observation>>,
}
#[cfg(target_os = "linux")]
impl Default for Desktop {
    fn default() -> Self {
        Self {
            backend: crate::linux::LinuxDesktopBackend::new(
                std::env::var("DISPLAY").unwrap_or_default(),
                std::env::var_os("XAUTHORITY").map(Into::into),
            ),
            last: Default::default(),
        }
    }
}
#[cfg(target_os = "linux")]
impl Desktop {
    pub fn invalidate(&self) {
        self.last.lock().unwrap().take();
    }
    pub async fn screenshot(&self) -> Result<Value> {
        anyhow::ensure!(
            desktop_available(),
            "X11 desktop is unavailable; headless command execution remains available"
        );
        let frame = self.backend.screenshot().await?;
        let id = uuid::Uuid::new_v4().to_string();
        *self.last.lock().unwrap() = Some(Observation {
            id: id.clone(),
            dimensions: frame.dimensions(),
            at: std::time::Instant::now(),
        });
        let mut value = frame.into_tool_value();
        value["observation_id"] = json!(id);
        value["coordinate_space"] = json!("screenshot_pixels");
        Ok(value)
    }
    pub async fn input(
        &self,
        id: &str,
        input: ComputerInput,
        authorized: impl Fn() -> bool + Send + 'static,
    ) -> Result<Value> {
        anyhow::ensure!(authorized(), "Desktop access revoked");
        input.validate()?;
        let observed = self
            .last
            .lock()
            .unwrap()
            .take()
            .context("Take a fresh executor screenshot before input")?;
        anyhow::ensure!(
            observed.id == id && observed.at.elapsed() <= std::time::Duration::from_secs(30),
            "Desktop observation is stale; take a fresh screenshot"
        );
        let dimensions = self.backend.screenshot().await?.dimensions();
        anyhow::ensure!(
            observed.dimensions == dimensions,
            "Display geometry changed; take a fresh screenshot"
        );
        validate_click_against_frame(&input, dimensions)?;
        anyhow::ensure!(authorized(), "Desktop access revoked before input");
        self.backend.input(input).await?;
        anyhow::ensure!(
            authorized(),
            "Input may have completed before access was revoked"
        );
        match self.screenshot().await {
            Ok(mut value) => {
                value["done"] = json!(true);
                Ok(value)
            }
            Err(_) => Ok(
                json!({"done":true,"observation_error":"Input completed but capture failed. Take a fresh screenshot; never replay the input."}),
            ),
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
#[derive(Default)]
pub struct Desktop;
#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
impl Desktop {
    pub fn invalidate(&self) {}
    pub async fn screenshot(&self) -> Result<Value> {
        anyhow::bail!("Desktop capture is unavailable on this endpoint")
    }
    pub async fn input(
        &self,
        _: &str,
        _: ComputerInput,
        _: impl Fn() -> bool + Send + 'static,
    ) -> Result<Value> {
        anyhow::bail!("Desktop input is unavailable on this endpoint")
    }
}

#[cfg(windows)]
pub use crate::windows::Desktop;

#[cfg(not(target_os = "macos"))]
pub fn desktop_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        if std::env::var_os("WAYLAND_DISPLAY").is_some() || std::env::var_os("DISPLAY").is_none() {
            return false;
        }
        ["scrot", "xdotool"].iter().all(|name| {
            std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
                .any(|p| p.join(name).is_file())
        })
    }
    #[cfg(windows)]
    {
        crate::windows::Desktop::is_available()
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        false
    }
}

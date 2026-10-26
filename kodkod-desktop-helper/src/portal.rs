//! An explicit, single-session xdg-desktop-portal backend for desktop capture
//! and input on personal Linux Wayland sessions.

use crate::capture::FrameCapture;
use crate::input::{InputActor, InputHealth};
use anyhow::{Context, Result, bail};
use ashpd::desktop::{
    PersistMode, Session,
    remote_desktop::{DeviceType, RemoteDesktop},
    screencast::{CursorMode, Screencast, SourceType},
};
use ashpd::enumflags2::BitFlags;
use ashpd::zvariant::{OwnedValue, SerializeDict, Type};
use futures::StreamExt;
use kodkod_executor::{
    computer::{ComputerInput, DesktopScreenshot, MAX_SCREENSHOT_PIXELS},
    desktop_helper::HelperInputs,
};
use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

type RemoteSession = Session<'static, RemoteDesktop<'static>>;

/// Lock-free readiness shared with status and input handling. A later PipeWire
/// frame cannot reopen a portal session after Closed or a geometry change.
#[derive(Clone)]
pub struct PortalHealth {
    capture_ready: Arc<AtomicBool>,
    session_open: Arc<AtomicBool>,
    geometry_changed: Arc<AtomicBool>,
    input: InputHealth,
}

impl PortalHealth {
    fn new(
        capture_ready: Arc<AtomicBool>,
        session_open: Arc<AtomicBool>,
        geometry_changed: Arc<AtomicBool>,
        input: InputHealth,
    ) -> Self {
        Self {
            capture_ready,
            session_open,
            geometry_changed,
            input,
        }
    }

    pub fn is_ready(&self) -> bool {
        self.capture_ready() && self.input.is_open()
    }

    pub fn capture_ready(&self) -> bool {
        self.capture_ready.load(Ordering::Acquire)
            && self.session_open.load(Ordering::Acquire)
            && !self.geometry_changed.load(Ordering::Acquire)
    }

    pub fn input_capabilities(&self) -> HelperInputs {
        if self.is_ready() {
            self.input.capabilities()
        } else {
            HelperInputs::default()
        }
    }

    pub fn geometry_changed(&self) -> bool {
        self.geometry_changed.load(Ordering::Acquire)
    }
}

/// Result of a fully dispatched input action. If the new frame didn't arrive
/// before timeout, the action is still complete and the helper must report an
/// observation error instead of implying the input failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InputOutcome {
    pub post_input_frame_ready: bool,
}

/// Read-only portal capabilities. Calling this does not create a portal
/// session and cannot trigger an authorization dialog.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PortalCapabilities {
    pub remote_desktop: bool,
    pub screen_cast: bool,
    pub keyboard: bool,
    pub pointer: bool,
    pub monitor: bool,
    pub window: bool,
}

/// Queries portal properties only. This is suitable for background status and
/// capability polling; consent begins only in [`PortalDesktop::start`].
pub async fn capabilities() -> PortalCapabilities {
    let Ok(remote) = RemoteDesktop::new().await else {
        return PortalCapabilities::default();
    };
    let remote_desktop_v2 = remote.get_property::<u32>("version").await.unwrap_or(0) >= 2;
    let devices = remote.available_device_types().await.ok();
    let Ok(screen_cast) = Screencast::new().await else {
        return PortalCapabilities {
            remote_desktop: remote_desktop_v2 && devices.is_some(),
            keyboard: devices.is_some_and(|flags| flags.contains(DeviceType::Keyboard)),
            pointer: devices.is_some_and(|flags| flags.contains(DeviceType::Pointer)),
            ..PortalCapabilities::default()
        };
    };
    let sources = screen_cast.available_source_types().await.ok();
    PortalCapabilities {
        remote_desktop: remote_desktop_v2 && devices.is_some(),
        screen_cast: sources.is_some(),
        keyboard: devices.is_some_and(|flags| flags.contains(DeviceType::Keyboard)),
        pointer: devices.is_some_and(|flags| flags.contains(DeviceType::Pointer)),
        monitor: sources.is_some_and(|flags| flags.contains(SourceType::Monitor)),
        window: sources.is_some_and(|flags| flags.contains(SourceType::Window)),
    }
}

/// Owns one portal session and its PipeWire capture worker. Construct this
/// only from the user's explicit start action; dropping or closing it ends the
/// capture session and never requests another one.
pub struct PortalDesktop {
    session: RemoteSession,
    capture: FrameCapture,
    input: InputActor,
    initial_frame_dimensions: (u32, u32),
    health: PortalHealth,
}

impl PortalDesktop {
    /// Alias for the read-only property probe used by status polling.
    pub async fn supported() -> PortalCapabilities {
        capabilities().await
    }

    /// Starts one RemoteDesktop session and asks the portal for keyboard,
    /// pointer, and one monitor/window stream. The portal's Start response is
    /// the sole consent point; persistence and restore tokens stay disabled.
    pub async fn start() -> Result<Self> {
        let remote = RemoteDesktop::new()
            .await
            .context("RemoteDesktop portal is unavailable")?;
        anyhow::ensure!(
            remote.get_property::<u32>("version").await.unwrap_or(0) >= 2,
            "The RemoteDesktop portal does not support ConnectToEIS"
        );
        let mut guard = PendingSession::new(remote.create_session().await?);
        let session = guard.session();
        let closed_stream = match session.receive_closed().await {
            Ok(stream) => stream,
            Err(error) => {
                guard.close().await;
                return Err(error).context("Could not watch the portal session");
            }
        };
        let session_open = Arc::new(AtomicBool::new(true));
        let session_open_watch = session_open.clone();
        tokio::spawn(async move {
            let mut stream = closed_stream;
            let _ = stream.next().await;
            session_open_watch.store(false, Ordering::Release);
        });

        let setup = async {
            let devices = remote
                .available_device_types()
                .await
                .context("Could not read RemoteDesktop capabilities")?;
            let requested_devices = devices & (DeviceType::Keyboard | DeviceType::Pointer);
            anyhow::ensure!(
                !requested_devices.is_empty(),
                "The portal does not provide a keyboard or pointer device"
            );

            let screen_cast = Screencast::new()
                .await
                .context("ScreenCast portal is unavailable")?;
            let sources = screen_cast
                .available_source_types()
                .await
                .context("Could not read ScreenCast capabilities")?;
            let source_types = match (
                sources.contains(SourceType::Monitor),
                sources.contains(SourceType::Window),
            ) {
                (true, true) => SourceType::Monitor | SourceType::Window,
                (true, false) => SourceType::Monitor.into(),
                (false, true) => SourceType::Window.into(),
                (false, false) => bail!("The portal has no monitor or window capture source"),
            };

            remote
                .select_devices(session, requested_devices, None, PersistMode::DoNot)
                .await?
                .response()?;
            screen_cast
                .select_sources(
                    session,
                    CursorMode::Embedded,
                    source_types,
                    false,
                    None,
                    PersistMode::DoNot,
                )
                .await?
                .response()?;

            let selected = start_remote_desktop(
                &remote,
                session,
                screen_cast
                    .get_property::<u32>("version")
                    .await
                    .context("Could not read ScreenCast version")?,
            )
            .await?;
            let mut streams = selected.streams;
            anyhow::ensure!(
                streams.len() == 1,
                "The portal did not return exactly one selected screen stream"
            );
            let stream = streams.remove(0);
            let pipewire_fd = screen_cast
                .open_pipe_wire_remote(session)
                .await
                .context("Could not open the portal's PipeWire remote")?;
            let eis_fd = remote
                .connect_to_eis(session)
                .await
                .context("Could not connect to the portal EIS session")?;
            let input =
                InputActor::start(eis_fd, stream.mapping_id.clone().unwrap_or_default()).await?;
            let node_id = stream.node_id;
            let serial = stream.pipewire_serial;
            let capture = tokio::task::spawn_blocking(move || {
                FrameCapture::start(pipewire_fd, node_id, serial)
            })
            .await
            .context("PipeWire capture startup task failed")??;

            Ok::<_, anyhow::Error>((capture, input))
        }
        .await;

        let (capture, input) = match setup {
            Ok(setup) => setup,
            Err(error) => {
                guard.close().await;
                return Err(error);
            }
        };

        anyhow::ensure!(
            !capture.geometry_changed_flag().load(Ordering::Acquire),
            "Display geometry changed during portal startup"
        );
        let initial_frame_dimensions = capture.dimensions()?;
        anyhow::ensure!(
            session_open.load(Ordering::Acquire),
            "Portal closed the desktop session during startup"
        );
        let input_health = input.health();
        anyhow::ensure!(
            input_health.is_open(),
            "The portal EIS session closed during startup"
        );
        let health = PortalHealth::new(
            capture.ready_flag(),
            session_open,
            capture.geometry_changed_flag(),
            input_health,
        );

        Ok(Self {
            session: guard.disarm(),
            capture,
            input,
            initial_frame_dimensions,
            health,
        })
    }

    /// True only while the portal session remains open and PipeWire is
    /// producing bounded, supported frames.
    pub fn is_ready(&self) -> bool {
        self.health.is_ready()
    }

    pub fn health(&self) -> PortalHealth {
        self.health.clone()
    }

    /// Returns the current raw frame encoded as a validated PNG. Encoding runs
    /// on a blocking worker and is performed only for screenshot requests.
    pub async fn screenshot(&self) -> Result<DesktopScreenshot> {
        anyhow::ensure!(
            !self.health.geometry_changed(),
            "Display geometry changed; close this session and explicitly restart sharing"
        );
        anyhow::ensure!(
            self.is_ready(),
            "Desktop capture session is no longer active"
        );
        let capture = self.capture.clone();
        let screenshot = tokio::task::spawn_blocking(move || capture.screenshot())
            .await
            .context("Screenshot encoding task failed")??;
        anyhow::ensure!(
            !self.health.geometry_changed(),
            "Display geometry changed; close this session and explicitly restart sharing"
        );
        anyhow::ensure!(
            self.is_ready(),
            "Desktop access closed while the screenshot was being encoded"
        );
        Ok(screenshot)
    }

    pub async fn view_frame(&self) -> Result<kodkod_executor::view::ViewFrame> {
        anyhow::ensure!(self.health.is_ready(), "Desktop sharing is unavailable");
        let capture = self.capture.clone();
        let frame = tokio::task::spawn_blocking(move || capture.view_frame()).await??;
        anyhow::ensure!(
            self.health.is_ready(),
            "Desktop sharing was revoked during capture"
        );
        Ok(frame)
    }

    /// Sends one action against the screenshot dimensions the caller already
    /// validated by observation ID. Pointer pixels are mapped only through the
    /// unique active EIS region whose mapping_id matches this portal stream.
    pub async fn input(
        &self,
        input: ComputerInput,
        source_frame_dimensions: (u32, u32),
        authorized: Arc<AtomicBool>,
    ) -> Result<InputOutcome> {
        anyhow::ensure!(
            !self.health.geometry_changed(),
            "Display geometry changed; close this session and explicitly restart sharing"
        );
        anyhow::ensure!(
            self.is_ready(),
            "Desktop access has been revoked or capture stopped"
        );
        anyhow::ensure!(
            authorized.load(Ordering::Acquire),
            "IPC client is no longer connected"
        );
        input.validate()?;
        let current_dimensions = self.capture.dimensions()?;
        anyhow::ensure!(
            current_dimensions == self.initial_frame_dimensions,
            "Display geometry changed; close this session and explicitly restart sharing"
        );
        anyhow::ensure!(
            current_dimensions == source_frame_dimensions,
            "Display geometry changed; take a fresh screenshot before input"
        );
        self.input
            .send(
                input,
                source_frame_dimensions,
                authorized.clone(),
                self.health.clone(),
            )
            .await?;
        anyhow::ensure!(
            authorized.load(Ordering::Acquire),
            "Input may have completed before the IPC client disconnected; do not replay it"
        );
        if !self.is_ready() {
            return Ok(InputOutcome {
                post_input_frame_ready: false,
            });
        }
        // Take the baseline only after the portal accepted every input event,
        // so a frame received while notifications were in flight cannot be
        // mistaken for the post-action observation.
        let sequence = match self.capture.frame_sequence() {
            Ok(sequence) => sequence,
            Err(_) if !self.is_ready() => {
                return Ok(InputOutcome {
                    post_input_frame_ready: false,
                });
            }
            Err(error) => return Err(error).context("Input was dispatched; do not replay it"),
        };
        let capture = self.capture.clone();
        let wait_authorized = authorized.clone();
        let wait_result = tokio::task::spawn_blocking(move || {
            capture.wait_for_newer_frame(sequence, Duration::from_secs(3), wait_authorized)
        })
        .await;
        let wait_result = match wait_result {
            Ok(result) => result,
            Err(_) => {
                anyhow::ensure!(
                    authorized.load(Ordering::Acquire),
                    "Input may have completed before the IPC client disconnected; do not replay it"
                );
                return Ok(InputOutcome {
                    post_input_frame_ready: false,
                });
            }
        };
        anyhow::ensure!(
            authorized.load(Ordering::Acquire),
            "Input may have completed before the IPC client disconnected; do not replay it"
        );
        let post_input_frame_ready = match wait_result {
            Ok(ready) => ready,
            Err(_) if !self.is_ready() => false,
            Err(error) => return Err(error).context("Input was dispatched; do not replay it"),
        };
        if !self.is_ready() {
            return Ok(InputOutcome {
                post_input_frame_ready: false,
            });
        }
        anyhow::ensure!(
            authorized.load(Ordering::Acquire),
            "Input may have completed before the IPC client disconnected; do not replay it"
        );
        Ok(InputOutcome {
            post_input_frame_ready,
        })
    }

    /// Stops the EIS actor and PipeWire worker before closing the portal session.
    pub async fn close(&mut self) -> Result<()> {
        self.input.close().await;
        self.capture.stop();
        self.health.session_open.store(false, Ordering::Release);
        self.session
            .close()
            .await
            .context("Could not close the RemoteDesktop portal session")
    }
}

impl Drop for PortalDesktop {
    fn drop(&mut self) {
        self.capture.stop();
        self.health.session_open.store(false, Ordering::Release);
    }
}

struct PendingSession(Option<RemoteSession>);

impl PendingSession {
    fn new(session: RemoteSession) -> Self {
        Self(Some(session))
    }

    fn session(&self) -> &RemoteSession {
        self.0
            .as_ref()
            .expect("pending portal session was disarmed")
    }

    async fn close(&mut self) {
        if let Some(session) = self.0.take() {
            let _ = session.close().await;
        }
    }

    fn disarm(&mut self) -> RemoteSession {
        self.0.take().expect("pending portal session was disarmed")
    }
}

impl Drop for PendingSession {
    fn drop(&mut self) {
        let Some(session) = self.0.take() else {
            return;
        };
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                let _ = session.close().await;
            });
        }
    }
}

#[derive(Debug)]
struct PortalStreamInfo {
    node_id: u32,
    mapping_id: Option<String>,
    pipewire_serial: Option<u64>,
}

#[derive(SerializeDict, Type, Debug)]
#[zvariant(signature = "dict")]
struct StartRequestOptions {
    handle_token: String,
}

struct RemoteStartResponse {
    streams: Vec<PortalStreamInfo>,
}

async fn start_remote_desktop(
    remote: &RemoteDesktop<'_>,
    session: &RemoteSession,
    screen_cast_version: u32,
) -> Result<RemoteStartResponse> {
    let options = StartRequestOptions {
        handle_token: format!("kodkod_{}", uuid::Uuid::new_v4().simple()),
    };
    let unique_name = remote
        .connection()
        .unique_name()
        .context("The D-Bus connection has no unique name")?
        .to_string();
    let sender = unique_name.trim_start_matches(':').replace('.', "_");
    let request_path = zbus::zvariant::ObjectPath::try_from(format!(
        "/org/freedesktop/portal/desktop/request/{sender}/{}",
        options.handle_token
    ))
    .context("Could not construct the portal Start request path")?;
    let request = zbus::Proxy::new(
        remote.connection(),
        "org.freedesktop.portal.Desktop",
        request_path,
        "org.freedesktop.portal.Request",
    )
    .await
    .context("Could not subscribe to the portal Start response")?;
    let mut responses = request
        .receive_signal("Response")
        .await
        .context("Could not subscribe to the portal Start response")?;

    // Subscribe before calling Start so a fast response cannot race listener
    // registration. This is the sole user-consent request in the session.
    remote
        .call_method("Start", &(session, "", &options))
        .await
        .context("Could not request the RemoteDesktop Start response")?;
    let response = responses
        .next()
        .await
        .context("The portal closed the Start request without a response")?;
    let (response_code, mut results): (u32, HashMap<String, OwnedValue>) = response
        .body()
        .deserialize()
        .context("Could not decode the portal Start response")?;
    anyhow::ensure!(
        response_code == 0,
        "The portal did not grant desktop sharing (response code {response_code})"
    );

    let devices = property::<u32>(&results, "devices")?
        .context("The portal Start response omitted selected devices")?;
    let _selected_devices = BitFlags::<DeviceType>::from_bits(devices)
        .map_err(|error| anyhow::anyhow!("The portal returned invalid device flags: {error}"))?;
    let stream_values = results
        .remove("streams")
        .context("The portal Start response omitted streams")?;
    let raw_streams: Vec<(u32, HashMap<String, OwnedValue>)> = Vec::try_from(stream_values)
        .map_err(|error| anyhow::anyhow!("The portal returned invalid stream data: {error}"))?;
    let streams = raw_streams
        .into_iter()
        .map(|(node_id, properties)| portal_stream(node_id, properties, screen_cast_version))
        .collect::<Result<Vec<_>>>()?;
    Ok(RemoteStartResponse { streams })
}

fn portal_stream(
    node_id: u32,
    properties: HashMap<String, OwnedValue>,
    screen_cast_version: u32,
) -> Result<PortalStreamInfo> {
    let (width, height) = property::<(i32, i32)>(&properties, "size")?
        .context("Portal stream omitted its logical size")?;
    anyhow::ensure!(
        width > 0 && height > 0,
        "Portal stream dimensions are invalid"
    );
    anyhow::ensure!(
        u64::from(width as u32) * u64::from(height as u32) <= MAX_SCREENSHOT_PIXELS,
        "Portal logical stream exceeds the desktop size limit"
    );
    let pipewire_serial = property::<u64>(&properties, "pipewire-serial")?;
    if screen_cast_version >= 6 {
        anyhow::ensure!(
            pipewire_serial.is_some(),
            "ScreenCast v6 omitted pipewire-serial; refusing to target a reusable node ID"
        );
    }
    Ok(PortalStreamInfo {
        node_id,
        mapping_id: property::<String>(&properties, "mapping_id")?,
        pipewire_serial,
    })
}

fn property<T>(properties: &HashMap<String, OwnedValue>, name: &str) -> Result<Option<T>>
where
    T: TryFrom<OwnedValue>,
    <T as TryFrom<OwnedValue>>::Error: std::fmt::Display,
{
    let Some(value) = properties.get(name) else {
        return Ok(None);
    };
    let value = value
        .try_clone()
        .map_err(|error| anyhow::anyhow!("Could not copy portal property {name}: {error}"))?;
    T::try_from(value)
        .map(Some)
        .map_err(|error| anyhow::anyhow!("Portal property {name} has an invalid type: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_frames_cannot_reopen_a_closed_or_resized_portal_session() {
        let capture_ready = Arc::new(AtomicBool::new(true));
        let session_open = Arc::new(AtomicBool::new(true));
        let geometry_changed = Arc::new(AtomicBool::new(false));
        let input_health = InputHealth::new();
        input_health.set_open(true);
        let health = PortalHealth::new(
            capture_ready.clone(),
            session_open.clone(),
            geometry_changed.clone(),
            input_health,
        );
        assert!(health.is_ready());
        session_open.store(false, Ordering::Release);
        capture_ready.store(false, Ordering::Release);
        capture_ready.store(true, Ordering::Release);
        assert!(!health.is_ready());

        session_open.store(true, Ordering::Release);
        geometry_changed.store(true, Ordering::Release);
        assert!(!health.is_ready());
    }
}

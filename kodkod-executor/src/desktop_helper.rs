//! Bounded IPC client for the optional Linux desktop helper.
//!
//! The helper owns the Wayland portal and PipeWire session. This module only
//! connects to an already-running per-user Unix socket; it never starts a
//! helper, requests desktop permission, or retries an input operation.

use crate::computer::{ComputerInput, DesktopScreenshot, MAX_SCREENSHOT_BYTES};
use anyhow::{Context, Result, bail};
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    fs,
    io::{Read, Write},
    os::{
        fd::AsRawFd,
        unix::fs::{FileTypeExt, MetadataExt},
        unix::net::UnixStream as StdUnixStream,
    },
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::UnixStream,
    sync::Mutex,
};
use uuid::Uuid;

pub const HELPER_PROTOCOL_VERSION: u16 = 1;
pub const MAX_IPC_MESSAGE_BYTES: usize = 12 * 1024 * 1024;
const STATUS_IPC_MESSAGE_BYTES: usize = 64 * 1024;
const IPC_TIMEOUT: Duration = Duration::from_secs(30);
const STATUS_TIMEOUT: Duration = Duration::from_millis(200);
const MAX_GENERATION_BYTES: usize = 128;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HelperRequest {
    pub version: u16,
    pub id: String,
    pub operation: HelperOperation,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum HelperOperation {
    Status,
    Screenshot,
    Input {
        observation_id: String,
        input: ComputerInput,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HelperReply {
    pub version: u16,
    pub id: String,
    pub result: std::result::Result<Value, String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StatusValue {
    pub available: bool,
    pub granted: bool,
    pub generation: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inputs: Option<HelperInputs>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HelperInputs {
    pub pointer: bool,
    pub keyboard: bool,
    pub text: bool,
}

#[derive(Clone, Debug)]
struct ObservationBinding {
    observation_id: String,
    generation: String,
}

#[derive(Default)]
struct ClientState {
    stream: Option<UnixStream>,
    binding: Option<ObservationBinding>,
    seen_epoch: u64,
}

/// If an IPC future is cancelled while it owns the persistent stream, the
/// partially-used stream must be discarded before any later request.
struct CancelledExchange {
    epoch: Arc<AtomicU64>,
    armed: bool,
}

impl CancelledExchange {
    fn new(epoch: Arc<AtomicU64>) -> Self {
        Self { epoch, armed: true }
    }

    fn finish(&mut self) {
        self.armed = false;
    }
}

impl Drop for CancelledExchange {
    fn drop(&mut self) {
        if self.armed {
            self.epoch.fetch_add(1, Ordering::AcqRel);
        }
    }
}

/// A persistent helper connection scoped to one executor controller.
///
/// Dropping the stream releases the helper's control lease. `invalidate` also
/// advances an epoch so an in-flight reply cannot restore a stale binding.
#[derive(Clone)]
pub struct DesktopHelperClient {
    socket: PathBuf,
    state: Arc<Mutex<ClientState>>,
    invalidation_epoch: Arc<AtomicU64>,
}

impl DesktopHelperClient {
    pub fn new(socket: PathBuf) -> Self {
        Self {
            socket,
            state: Arc::new(Mutex::new(ClientState::default())),
            invalidation_epoch: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Ask the helper for one current frame. The helper's response is returned
    /// in the common screenshot tool-value shape plus its opaque observation
    /// ID and desktop generation.
    pub async fn screenshot(&self) -> Result<Value> {
        let mut state = self.state.lock().await;
        self.sync_invalidation(&mut state);
        self.ensure_stream(&mut state).await?;
        state.binding = None;
        let mut stream = state.stream.take().expect("stream initialized");
        let mut cancellation = CancelledExchange::new(self.invalidation_epoch.clone());

        let request = HelperRequest {
            version: HELPER_PROTOCOL_VERSION,
            id: Uuid::now_v7().to_string(),
            operation: HelperOperation::Screenshot,
        };
        let response = exchange(&mut stream, &request).await;
        cancellation.finish();
        if self.invalidation_epoch.load(Ordering::Acquire) != state.seen_epoch {
            state.binding = None;
            bail!("Desktop access was invalidated while the screenshot was being captured");
        }
        let reply = match response {
            Ok(reply) => reply,
            Err(error) => {
                state.stream = None;
                state.binding = None;
                return Err(error).context("Could not get a screenshot from the desktop helper");
            }
        };
        let value = match validate_reply(reply, &request.id) {
            Ok(Ok(value)) => value,
            Ok(Err(message)) => bail!("Desktop helper rejected screenshot: {message}"),
            Err(error) => return Err(error).context("Desktop helper screenshot reply was invalid"),
        };
        let binding = validate_screenshot_value(&value)?;
        state.stream = Some(stream);
        state.binding = Some(binding);
        Ok(value)
    }

    /// Dispatch input only against the latest screenshot this client returned.
    /// The authorization callback runs immediately before writing the request.
    /// Once dispatch starts, any transport/protocol failure is treated as an
    /// uncertain outcome and is never automatically replayed.
    pub async fn input<F>(
        &self,
        observation_id: &str,
        input: ComputerInput,
        authorized: F,
    ) -> Result<Value>
    where
        F: FnOnce(&str) -> bool,
    {
        input.validate()?;
        validate_uuid(observation_id, "Desktop observation id")?;

        let mut state = self.state.lock().await;
        self.sync_invalidation(&mut state);
        self.ensure_stream(&mut state).await?;

        let binding = state
            .binding
            .as_ref()
            .context("Take a fresh desktop screenshot before sending input")?;
        anyhow::ensure!(
            binding.observation_id == observation_id,
            "Desktop input must use the most recent screenshot observation"
        );
        anyhow::ensure!(
            !binding.generation.is_empty(),
            "Desktop screenshot is no longer bound to an authorized generation"
        );
        anyhow::ensure!(
            authorized(&binding.generation),
            "Desktop input is no longer authorized"
        );

        // The observation is single-use, including when a write or reply is
        // interrupted, because the helper may have dispatched the input.
        state.binding = None;
        let mut stream = state.stream.take().expect("stream initialized");
        let mut cancellation = CancelledExchange::new(self.invalidation_epoch.clone());
        let request = HelperRequest {
            version: HELPER_PROTOCOL_VERSION,
            id: Uuid::now_v7().to_string(),
            operation: HelperOperation::Input {
                observation_id: observation_id.to_owned(),
                input,
            },
        };
        let response = exchange(&mut stream, &request).await;
        cancellation.finish();
        if self.invalidation_epoch.load(Ordering::Acquire) != state.seen_epoch {
            state.binding = None;
            bail!("Desktop input outcome is uncertain because access was revoked during dispatch");
        }
        let reply = match response {
            Ok(reply) => reply,
            Err(error) => {
                state.stream = None;
                state.binding = None;
                return Err(error).context(
                    "Desktop input outcome is uncertain; take a fresh screenshot before continuing",
                );
            }
        };
        let value = match validate_reply(reply, &request.id) {
            Ok(Ok(value)) => value,
            Ok(Err(message)) => bail!("Desktop helper rejected input: {message}"),
            Err(error) => {
                return Err(error).context(
                    "Desktop input outcome is uncertain; take a fresh screenshot before continuing",
                );
            }
        };
        let binding = match parse_observation_binding(&value)? {
            Some(binding) => Some(binding),
            None if value.get("done").and_then(Value::as_bool) == Some(true)
                && value.get("observation_error").is_some() =>
            {
                // The helper confirms input completion but could not capture
                // its follow-up frame. Return that success and require a new
                // explicit screenshot before any subsequent input.
                None
            }
            None => {
                bail!(
                    "Desktop helper did not return a fresh post-input observation; take a new screenshot"
                )
            }
        };
        state.stream = Some(stream);
        state.binding = binding;
        Ok(value)
    }

    /// Revoke the local observation and release the helper control stream.
    /// The epoch also prevents a concurrent request from restoring its result.
    pub fn invalidate(&self) {
        self.invalidation_epoch.fetch_add(1, Ordering::AcqRel);
        if let Ok(mut state) = self.state.try_lock() {
            state.stream = None;
            state.binding = None;
            state.seen_epoch = self.invalidation_epoch.load(Ordering::Acquire);
        }
    }

    fn sync_invalidation(&self, state: &mut ClientState) {
        let epoch = self.invalidation_epoch.load(Ordering::Acquire);
        if state.seen_epoch != epoch {
            state.stream = None;
            state.binding = None;
            state.seen_epoch = epoch;
        }
    }

    async fn ensure_stream(&self, state: &mut ClientState) -> Result<()> {
        if state.stream.is_some() {
            return Ok(());
        }
        validate_socket_path(&self.socket)?;
        let stream = UnixStream::connect(&self.socket).await.with_context(|| {
            format!(
                "Could not connect to desktop helper at {}",
                self.socket.display()
            )
        })?;
        verify_peer_uid(&stream)?;
        state.stream = Some(stream);
        Ok(())
    }
}

/// Query helper status over a short-lived status-only connection. This method
/// never launches a helper or prompts for desktop permission.
pub fn status(socket: &Path) -> Result<Value> {
    validate_socket_path(socket)?;
    let stream = StdUnixStream::connect(socket).with_context(|| {
        format!(
            "Could not connect to desktop helper at {}",
            socket.display()
        )
    })?;
    stream.set_read_timeout(Some(STATUS_TIMEOUT))?;
    stream.set_write_timeout(Some(STATUS_TIMEOUT))?;
    verify_peer_uid(&stream)?;
    let request = HelperRequest {
        version: HELPER_PROTOCOL_VERSION,
        id: Uuid::now_v7().to_string(),
        operation: HelperOperation::Status,
    };
    write_frame_sync(&stream, &request, STATUS_IPC_MESSAGE_BYTES)?;
    let reply = read_reply_sync(&stream, STATUS_IPC_MESSAGE_BYTES)?;
    let value = validate_reply(reply, &request.id)?
        .map_err(|message| anyhow::anyhow!("Desktop helper status failed: {message}"))?;
    serde_json::from_value::<StatusValue>(value.clone())
        .context("Desktop helper returned invalid status")?;
    Ok(value)
}

async fn exchange(stream: &mut UnixStream, request: &HelperRequest) -> Result<HelperReply> {
    let body = serde_json::to_vec(request).context("Could not encode desktop helper request")?;
    anyhow::ensure!(
        !body.is_empty() && body.len() <= MAX_IPC_MESSAGE_BYTES,
        "Desktop helper request exceeds the IPC size limit"
    );

    let exchange = async {
        stream.write_all(&(body.len() as u32).to_be_bytes()).await?;
        stream.write_all(&body).await?;
        let mut length = [0u8; 4];
        stream.read_exact(&mut length).await?;
        let length = u32::from_be_bytes(length) as usize;
        anyhow::ensure!(
            length > 0 && length <= MAX_IPC_MESSAGE_BYTES,
            "Desktop helper reply exceeds the IPC size limit"
        );
        let mut bytes = vec![0u8; length];
        stream.read_exact(&mut bytes).await?;
        serde_json::from_slice(&bytes).context("Desktop helper returned invalid JSON")
    };
    tokio::time::timeout(IPC_TIMEOUT, exchange)
        .await
        .context("Desktop helper IPC timed out")?
}

fn validate_reply(
    reply: HelperReply,
    expected_id: &str,
) -> Result<std::result::Result<Value, String>> {
    anyhow::ensure!(
        reply.version == HELPER_PROTOCOL_VERSION,
        "Desktop helper protocol version mismatch"
    );
    anyhow::ensure!(
        reply.id == expected_id,
        "Desktop helper reply ID did not match the request"
    );
    Ok(reply.result)
}

fn validate_screenshot_value(value: &Value) -> Result<ObservationBinding> {
    let object = value
        .as_object()
        .context("Desktop helper screenshot was not an object")?;
    let observation_id = object
        .get("observation_id")
        .and_then(Value::as_str)
        .context("Desktop helper screenshot omitted its observation ID")?;
    validate_uuid(observation_id, "Desktop observation id")?;
    let generation = object
        .get("generation")
        .and_then(Value::as_str)
        .context("Desktop helper screenshot omitted its generation")?;
    validate_generation(generation)?;
    anyhow::ensure!(
        object.get("coordinate_space").and_then(Value::as_str) == Some("screenshot_pixels"),
        "Desktop helper screenshot used an unsupported coordinate space"
    );
    let width = object
        .get("width")
        .and_then(Value::as_u64)
        .context("Desktop helper screenshot omitted its width")?;
    let height = object
        .get("height")
        .and_then(Value::as_u64)
        .context("Desktop helper screenshot omitted its height")?;
    anyhow::ensure!(
        width > 0
            && height > 0
            && width.saturating_mul(height) <= crate::computer::MAX_SCREENSHOT_PIXELS,
        "Desktop helper screenshot dimensions exceed the limit"
    );
    let encoded = object
        .get("png")
        .and_then(Value::as_str)
        .context("Desktop helper screenshot omitted its PNG")?;
    anyhow::ensure!(
        encoded.len() <= MAX_SCREENSHOT_BYTES.div_ceil(3) * 4,
        "Desktop helper screenshot exceeds the size limit"
    );
    let png = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .context("Desktop helper screenshot contained invalid base64")?;
    anyhow::ensure!(
        png.len() <= MAX_SCREENSHOT_BYTES,
        "Desktop helper screenshot exceeds the size limit"
    );
    anyhow::ensure!(
        DesktopScreenshot::dimensions_from_png(&png) == Some((width as u32, height as u32)),
        "Desktop helper screenshot dimensions did not match its PNG"
    );
    Ok(ObservationBinding {
        observation_id: observation_id.to_owned(),
        generation: generation.to_owned(),
    })
}

fn parse_observation_binding(value: &Value) -> Result<Option<ObservationBinding>> {
    let Some(object) = value.as_object() else {
        return Ok(None);
    };
    let (Some(id), Some(generation)) = (
        object.get("observation_id").and_then(Value::as_str),
        object.get("generation").and_then(Value::as_str),
    ) else {
        return Ok(None);
    };
    validate_uuid(id, "Desktop observation id")?;
    validate_generation(generation)?;
    Ok(Some(ObservationBinding {
        observation_id: id.to_owned(),
        generation: generation.to_owned(),
    }))
}

fn validate_generation(generation: &str) -> Result<()> {
    anyhow::ensure!(
        !generation.is_empty()
            && generation.len() <= MAX_GENERATION_BYTES
            && !generation.chars().any(char::is_control),
        "Desktop helper returned an invalid generation"
    );
    Ok(())
}

fn validate_uuid(value: &str, label: &str) -> Result<()> {
    let parsed = Uuid::parse_str(value).with_context(|| format!("{label} was invalid"))?;
    anyhow::ensure!(
        parsed.to_string() == value,
        "{label} must be a canonical lowercase UUID"
    );
    Ok(())
}

fn validate_socket_path(socket: &Path) -> Result<()> {
    anyhow::ensure!(
        socket.is_absolute(),
        "Desktop helper socket path must be absolute"
    );
    let parent = socket
        .parent()
        .context("Desktop helper socket has no parent directory")?;
    let parent_metadata = fs::symlink_metadata(parent).with_context(|| {
        format!(
            "Could not inspect desktop helper directory {}",
            parent.display()
        )
    })?;
    anyhow::ensure!(
        parent_metadata.file_type().is_dir()
            && parent_metadata.uid() == unsafe { libc::geteuid() }
            && parent_metadata.mode() & 0o777 == 0o700,
        "Desktop helper directory must be owned by the current user with mode 0700"
    );
    let socket_metadata = fs::symlink_metadata(socket).with_context(|| {
        format!(
            "Could not inspect desktop helper socket {}",
            socket.display()
        )
    })?;
    anyhow::ensure!(
        socket_metadata.file_type().is_socket()
            && socket_metadata.uid() == unsafe { libc::geteuid() }
            && socket_metadata.mode() & 0o777 == 0o600,
        "Desktop helper socket must be owned by the current user with mode 0600"
    );
    Ok(())
}

fn verify_peer_uid(stream: &impl AsRawFd) -> Result<()> {
    // Linux SO_PEERCRED binds each IPC stream to the local user that owns the
    // helper process, independently of the filesystem check before connect.
    let mut credentials = std::mem::MaybeUninit::<libc::ucred>::zeroed();
    let mut length = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    let result = unsafe {
        libc::getsockopt(
            stream.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            credentials.as_mut_ptr().cast(),
            &mut length,
        )
    };
    anyhow::ensure!(
        result == 0,
        "Could not verify desktop helper peer credentials"
    );
    anyhow::ensure!(
        length as usize == std::mem::size_of::<libc::ucred>()
            && unsafe { credentials.assume_init() }.uid == unsafe { libc::geteuid() },
        "Desktop helper is running as a different user"
    );
    Ok(())
}

fn write_frame_sync(stream: &StdUnixStream, request: &HelperRequest, max: usize) -> Result<()> {
    let body = serde_json::to_vec(request)?;
    anyhow::ensure!(
        !body.is_empty() && body.len() <= max,
        "Desktop helper request is too large"
    );
    let mut stream = stream;
    stream.write_all(&(body.len() as u32).to_be_bytes())?;
    stream.write_all(&body)?;
    Ok(())
}

fn read_reply_sync(mut stream: &StdUnixStream, max: usize) -> Result<HelperReply> {
    let mut header = [0; 4];
    stream.read_exact(&mut header)?;
    let length = u32::from_be_bytes(header) as usize;
    anyhow::ensure!(
        length > 0 && length <= max,
        "Desktop helper reply is too large"
    );
    let mut body = vec![0; length];
    stream.read_exact(&mut body)?;
    serde_json::from_slice(&body).context("Desktop helper returned invalid JSON")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::{os::unix::fs::PermissionsExt, sync::atomic::AtomicUsize};
    use tempfile::TempDir;
    use tokio::{net::UnixListener, sync::oneshot, task::JoinHandle};

    fn secure_socket_dir() -> (TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o700)).unwrap();
        let socket = dir.path().join("helper.sock");
        (dir, socket)
    }

    async fn bind_listener(socket: &Path) -> UnixListener {
        let listener = UnixListener::bind(socket).unwrap();
        fs::set_permissions(socket, fs::Permissions::from_mode(0o600)).unwrap();
        listener
    }

    async fn read_frame(stream: &mut UnixStream) -> HelperRequest {
        let mut header = [0; 4];
        stream.read_exact(&mut header).await.unwrap();
        let length = u32::from_be_bytes(header) as usize;
        assert!(length > 0 && length <= MAX_IPC_MESSAGE_BYTES);
        let mut body = vec![0; length];
        stream.read_exact(&mut body).await.unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    async fn write_reply(stream: &mut UnixStream, reply: &HelperReply) {
        let body = serde_json::to_vec(reply).unwrap();
        stream
            .write_all(&(body.len() as u32).to_be_bytes())
            .await
            .unwrap();
        stream.write_all(&body).await.unwrap();
    }

    fn screenshot_value(id: &str, generation: &str) -> Value {
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png.extend_from_slice(&[0; 8]);
        png.extend_from_slice(&1u32.to_be_bytes());
        png.extend_from_slice(&1u32.to_be_bytes());
        json!({
            "png": base64::engine::general_purpose::STANDARD.encode(png),
            "width": 1,
            "height": 1,
            "coordinate_space": "screenshot_pixels",
            "observation_id": id,
            "generation": generation,
        })
    }

    fn reply_for(
        request: &HelperRequest,
        result: std::result::Result<Value, String>,
    ) -> HelperReply {
        HelperReply {
            version: HELPER_PROTOCOL_VERSION,
            id: request.id.clone(),
            result,
        }
    }

    #[tokio::test]
    async fn screenshot_observation_is_single_use_and_input_rebinds() {
        let (_dir, socket) = secure_socket_dir();
        let listener = bind_listener(&socket).await;
        let first = Uuid::now_v7().to_string();
        let second = Uuid::now_v7().to_string();
        let third = Uuid::now_v7().to_string();
        let first_server_id = first.clone();
        let second_server_id = second.clone();
        let third_server_id = third.clone();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let screenshot = read_frame(&mut stream).await;
            assert!(matches!(screenshot.operation, HelperOperation::Screenshot));
            write_reply(
                &mut stream,
                &reply_for(&screenshot, Ok(screenshot_value(&first_server_id, "g1"))),
            )
            .await;

            let input = read_frame(&mut stream).await;
            match &input.operation {
                HelperOperation::Input { observation_id, .. } => {
                    assert_eq!(observation_id, &first_server_id);
                }
                other => panic!("unexpected request: {other:?}"),
            }
            write_reply(
                &mut stream,
                &reply_for(
                    &input,
                    Ok(json!({"observation_id": second_server_id, "generation": "g1"})),
                ),
            )
            .await;

            let input = read_frame(&mut stream).await;
            match &input.operation {
                HelperOperation::Input { observation_id, .. } => {
                    assert_eq!(observation_id, &second_server_id);
                }
                other => panic!("unexpected request: {other:?}"),
            }
            write_reply(
                &mut stream,
                &reply_for(
                    &input,
                    Ok(json!({
                        "done": true,
                        "observation_id": third_server_id,
                        "generation": "g1"
                    })),
                ),
            )
            .await;
        });

        let client = DesktopHelperClient::new(socket);
        let screenshot = client.screenshot().await.unwrap();
        let observed = screenshot["observation_id"].as_str().unwrap().to_owned();
        let first_input = client
            .input(&observed, ComputerInput::Click { x: 0, y: 0 }, |_| true)
            .await
            .unwrap();
        assert_eq!(first_input["observation_id"], second);
        assert!(
            client
                .input(&observed, ComputerInput::Click { x: 0, y: 0 }, |_| true)
                .await
                .is_err()
        );
        client
            .input(&second, ComputerInput::Click { x: 0, y: 0 }, |_| true)
            .await
            .unwrap();
        server.await.unwrap();
    }

    #[tokio::test]
    async fn invalidation_and_authorization_prevent_dispatch() {
        let (_dir, socket) = secure_socket_dir();
        let listener = bind_listener(&socket).await;
        let accepts = Arc::new(AtomicUsize::new(0));
        let seen_accepts = accepts.clone();
        let server: JoinHandle<()> = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            seen_accepts.fetch_add(1, Ordering::SeqCst);
            let request = read_frame(&mut stream).await;
            write_reply(
                &mut stream,
                &reply_for(
                    &request,
                    Ok(screenshot_value(&Uuid::now_v7().to_string(), "g")),
                ),
            )
            .await;
            // Dropping the listener after the first connection makes any
            // accidental second request fail instead of silently dispatching.
        });
        let client = DesktopHelperClient::new(socket);
        let screenshot = client.screenshot().await.unwrap();
        let id = screenshot["observation_id"].as_str().unwrap().to_owned();
        assert!(
            client
                .input(&id, ComputerInput::Click { x: 0, y: 0 }, |_| false)
                .await
                .is_err()
        );
        client.invalidate();
        assert!(
            client
                .input(&id, ComputerInput::Click { x: 0, y: 0 }, |_| true)
                .await
                .is_err()
        );
        server.await.unwrap();
        assert_eq!(accepts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn disconnect_after_input_is_uncertain_and_not_replayed() {
        let (_dir, socket) = secure_socket_dir();
        let listener = bind_listener(&socket).await;
        let accepted = Arc::new(AtomicUsize::new(0));
        let count = accepted.clone();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            count.fetch_add(1, Ordering::SeqCst);
            let screenshot = read_frame(&mut stream).await;
            write_reply(
                &mut stream,
                &reply_for(
                    &screenshot,
                    Ok(screenshot_value(&Uuid::now_v7().to_string(), "g")),
                ),
            )
            .await;
            let input = read_frame(&mut stream).await;
            assert!(matches!(input.operation, HelperOperation::Input { .. }));
            drop(stream);
        });
        let client = DesktopHelperClient::new(socket);
        let screenshot = client.screenshot().await.unwrap();
        let id = screenshot["observation_id"].as_str().unwrap().to_owned();
        let error = client
            .input(&id, ComputerInput::Click { x: 0, y: 0 }, |_| true)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("outcome is uncertain"));
        assert!(
            client
                .input(&id, ComputerInput::Click { x: 0, y: 0 }, |_| true)
                .await
                .is_err()
        );
        server.await.unwrap();
        assert_eq!(accepted.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn completed_input_without_a_followup_frame_is_still_successful() {
        let (_dir, socket) = secure_socket_dir();
        let listener = bind_listener(&socket).await;
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let screenshot = read_frame(&mut stream).await;
            write_reply(
                &mut stream,
                &reply_for(
                    &screenshot,
                    Ok(screenshot_value(&Uuid::now_v7().to_string(), "g")),
                ),
            )
            .await;
            let input = read_frame(&mut stream).await;
            write_reply(
                &mut stream,
                &reply_for(
                    &input,
                    Ok(json!({
                        "done": true,
                        "observation_error": "follow-up capture failed"
                    })),
                ),
            )
            .await;
        });

        let client = DesktopHelperClient::new(socket);
        let screenshot = client.screenshot().await.unwrap();
        let id = screenshot["observation_id"].as_str().unwrap().to_owned();
        let result = client
            .input(&id, ComputerInput::Click { x: 0, y: 0 }, |generation| {
                generation == "g"
            })
            .await
            .unwrap();
        assert_eq!(result["done"], true);
        assert_eq!(result["observation_error"], "follow-up capture failed");
        assert!(
            client
                .input(&id, ComputerInput::Click { x: 0, y: 0 }, |_| true)
                .await
                .is_err()
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn cancelling_an_exchange_drops_its_stream_before_reconnecting() {
        let (_dir, socket) = secure_socket_dir();
        let listener = bind_listener(&socket).await;
        let next_observation = Uuid::now_v7().to_string();
        let server_observation = next_observation.clone();
        let (seen_tx, seen_rx) = oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut first, _) = listener.accept().await.unwrap();
            let request = read_frame(&mut first).await;
            assert!(matches!(request.operation, HelperOperation::Screenshot));
            seen_tx.send(()).unwrap();

            let mut byte = [0; 1];
            assert_eq!(first.read(&mut byte).await.unwrap(), 0);
            drop(first);

            let (mut second, _) = listener.accept().await.unwrap();
            let request = read_frame(&mut second).await;
            assert!(matches!(request.operation, HelperOperation::Screenshot));
            write_reply(
                &mut second,
                &reply_for(&request, Ok(screenshot_value(&server_observation, "g2"))),
            )
            .await;
        });

        let client = DesktopHelperClient::new(socket);
        let task_client = client.clone();
        let task = tokio::spawn(async move { task_client.screenshot().await });
        seen_rx.await.unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());

        let screenshot = client.screenshot().await.unwrap();
        assert_eq!(screenshot["observation_id"], next_observation);
        server.await.unwrap();
    }

    #[test]
    fn status_uses_only_a_secure_existing_socket() {
        let (_dir, socket) = secure_socket_dir();
        // No connection is attempted and no helper is spawned when the path
        // doesn't already identify a correctly-protected Unix socket.
        assert!(status(&socket).is_err());
        assert!(!socket.exists());
    }

    #[test]
    fn status_uses_a_short_lived_framed_connection() {
        use std::os::unix::net::UnixListener as StdUnixListener;

        let (_dir, socket) = secure_socket_dir();
        let listener = StdUnixListener::bind(&socket).unwrap();
        fs::set_permissions(&socket, fs::Permissions::from_mode(0o600)).unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut header = [0; 4];
            stream.read_exact(&mut header).unwrap();
            let length = u32::from_be_bytes(header) as usize;
            assert!(length > 0 && length <= STATUS_IPC_MESSAGE_BYTES);
            let mut body = vec![0; length];
            stream.read_exact(&mut body).unwrap();
            let request: HelperRequest = serde_json::from_slice(&body).unwrap();
            assert!(matches!(request.operation, HelperOperation::Status));
            let reply = reply_for(
                &request,
                Ok(json!({
                    "available": true,
                    "granted": true,
                    "generation": "local-test"
                })),
            );
            let body = serde_json::to_vec(&reply).unwrap();
            stream
                .write_all(&(body.len() as u32).to_be_bytes())
                .unwrap();
            stream.write_all(&body).unwrap();
        });

        let value = status(&socket).unwrap();
        server.join().unwrap();
        assert_eq!(value["available"], true);
        assert_eq!(value["granted"], true);
        assert_eq!(value["generation"], "local-test");
    }
}

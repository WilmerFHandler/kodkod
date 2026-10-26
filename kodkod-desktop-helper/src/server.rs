use crate::portal::{PortalDesktop, PortalHealth};
use anyhow::{Context, Result, bail};
use kodkod_executor::{
    computer::DesktopScreenshot,
    desktop_helper::{
        HELPER_PROTOCOL_VERSION, HelperOperation, HelperReply, HelperRequest,
        MAX_IPC_MESSAGE_BYTES, StatusValue,
    },
};
use serde_json::{Value, json};
use std::{
    collections::HashSet,
    fs::OpenOptions,
    os::unix::fs::{FileTypeExt, MetadataExt, OpenOptionsExt, PermissionsExt},
    path::PathBuf,
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWriteExt},
    net::{UnixListener, UnixStream},
    sync::{Semaphore, mpsc},
    task::{JoinHandle, JoinSet},
};
use uuid::Uuid;

struct State {
    desktop: tokio::sync::Mutex<Option<PortalDesktop>>,
    health: Mutex<Option<PortalHealth>>,
    available: bool,
    generation: String,
    controller: Arc<Semaphore>,
    clients: Mutex<Vec<Weak<AtomicBool>>>,
}
impl State {
    fn ready(&self) -> bool {
        self.health
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(PortalHealth::is_ready)
    }
    fn status(&self) -> Value {
        serde_json::to_value(StatusValue {
            available: self.available,
            granted: self.ready(),
            generation: self.generation.clone(),
            inputs: self
                .health
                .lock()
                .unwrap()
                .as_ref()
                .map(PortalHealth::input_capabilities),
        })
        .expect("status serialization")
    }
    fn revoke_clients(&self) {
        for client in self
            .clients
            .lock()
            .unwrap()
            .iter()
            .filter_map(Weak::upgrade)
        {
            client.store(false, Ordering::Release);
        }
    }
}
struct Observation {
    id: String,
    dimensions: (u32, u32),
    at: Instant,
}
struct ReaderGuard {
    task: JoinHandle<()>,
    alive: Arc<AtomicBool>,
}
impl Drop for ReaderGuard {
    fn drop(&mut self) {
        self.alive.store(false, Ordering::Release);
        self.task.abort();
    }
}
struct SocketCleanup(PathBuf);
impl Drop for SocketCleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

pub async fn run(args: impl Iterator<Item = String>) -> Result<()> {
    let mut args = args.peekable();
    let mut request_access = false;
    let mut socket = None;
    let mut status_only = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "serve" => {}
            "status" => status_only = true,
            "--socket" => {
                anyhow::ensure!(socket.is_none(), "Duplicate socket path");
                socket = Some(PathBuf::from(
                    args.next().context("--socket requires a path")?,
                ));
            }
            "--request-access" => request_access = true,
            "--help" | "-h" => {
                println!(
                    "kodkod-desktop-helper [serve|status] [--socket PATH] [--request-access]\nRun inside your Wayland user session. --request-access shows the desktop's sharing permission dialog. Background status never requests permission."
                );
                return Ok(());
            }
            _ => bail!("Unknown desktop helper option"),
        }
    }
    let socket = socket
        .or_else(|| {
            std::env::var_os("XDG_RUNTIME_DIR")
                .map(|dir| PathBuf::from(dir).join("kodkod-desktop-helper/desktop.sock"))
        })
        .context("Pass --socket or run in a session with XDG_RUNTIME_DIR")?;
    anyhow::ensure!(socket.is_absolute(), "Desktop socket path must be absolute");
    if status_only {
        println!("{}", kodkod_executor::desktop_helper::status(&socket)?);
        return Ok(());
    }
    let parent = socket.parent().context("Socket has no parent")?;
    if !parent.exists() {
        std::fs::create_dir_all(parent)?;
        std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700))?;
    }
    let metadata = std::fs::symlink_metadata(parent)?;
    anyhow::ensure!(
        metadata.is_dir()
            && metadata.uid() == unsafe { libc::geteuid() }
            && metadata.mode() & 0o077 == 0,
        "Desktop socket directory must be private and owned by this user"
    );
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(parent.join("helper.lock"))?;
    lock.try_lock()
        .context("A desktop helper is already running for this socket directory")?;
    match std::fs::symlink_metadata(&socket) {
        Ok(metadata) => {
            anyhow::ensure!(
                metadata.file_type().is_socket() && metadata.uid() == unsafe { libc::geteuid() },
                "Existing socket path is not an owned Unix socket"
            );
            anyhow::ensure!(
                std::os::unix::net::UnixStream::connect(&socket).is_err(),
                "Desktop helper is already running"
            );
            std::fs::remove_file(&socket)?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let listener = UnixListener::bind(&socket)?;
    std::fs::set_permissions(&socket, std::fs::Permissions::from_mode(0o600))?;
    let _socket_cleanup = SocketCleanup(socket);
    let supported = tokio::time::timeout(Duration::from_secs(5), PortalDesktop::supported())
        .await
        .unwrap_or_default();
    let state = Arc::new(State {
        desktop: tokio::sync::Mutex::new(None),
        health: Mutex::new(None),
        available: supported.remote_desktop
            && supported.screen_cast
            && supported.keyboard
            && supported.pointer,
        generation: Uuid::new_v4().to_string(),
        controller: Arc::new(Semaphore::new(1)),
        clients: Mutex::new(Vec::new()),
    });
    let access_task = if request_access {
        anyhow::ensure!(
            state.available,
            "This desktop does not provide a RemoteDesktop and ScreenCast portal with keyboard/pointer input"
        );
        let state = state.clone();
        Some(tokio::task::spawn_local(async move {
            match PortalDesktop::start().await {
                Ok(desktop) => {
                    let health = desktop.health();
                    *state.desktop.lock().await = Some(desktop);
                    *state.health.lock().unwrap() = Some(health);
                    eprintln!("Wayland sharing approved and ready");
                }
                Err(error) => eprintln!(
                    "Wayland sharing was declined or could not start; no desktop access was granted: {error:#}"
                ),
            }
        }))
    } else {
        None
    };
    eprintln!("Desktop helper listening; sharing requires explicit portal approval");
    let mut connections = JoinSet::new();
    let connection_slots = Arc::new(Semaphore::new(16));
    let shutdown = shutdown_signal();
    tokio::pin!(shutdown);
    loop {
        tokio::select! {
            result=&mut shutdown=> { result?; break; },
            accepted=listener.accept()=> {
                let (stream,_)=accepted?;
                if stream.peer_cred()?.uid()!=unsafe{libc::geteuid()} { continue; }
                let Ok(slot)=connection_slots.clone().try_acquire_owned() else { continue; };
                let state=state.clone();
                connections.spawn(async move { let _slot=slot; let _=connection(stream,state).await; });
            },
            _=connections.join_next(), if !connections.is_empty()=>{},
        }
    }
    state.revoke_clients();
    if let Some(task) = access_task
        && !task.is_finished()
    {
        task.abort();
        let _ = task.await;
    }
    // Let input finish its key releases before closing the compositor session.
    let _ = tokio::time::timeout(Duration::from_secs(3), async {
        while connections.join_next().await.is_some() {}
    })
    .await;
    if let Some(mut desktop) = state.desktop.lock().await.take() {
        let _ = desktop.close().await;
    }
    connections.abort_all();
    Ok(())
}

async fn connection(stream: UnixStream, state: Arc<State>) -> Result<()> {
    let (mut reader, mut writer) = stream.into_split();
    let alive = Arc::new(AtomicBool::new(true));
    {
        let mut clients = state.clients.lock().unwrap();
        clients.retain(|c| c.strong_count() > 0);
        clients.push(Arc::downgrade(&alive));
    }
    let reader_alive = alive.clone();
    let (sender, mut receiver) = mpsc::channel(1);
    let task = tokio::spawn(async move {
        loop {
            let read =
                tokio::time::timeout(Duration::from_secs(60), read_request(&mut reader)).await;
            let Ok(Ok(request)) = read else {
                break;
            };
            if sender.try_send(request).is_err() {
                break;
            }
        }
        reader_alive.store(false, Ordering::Release);
    });
    let _reader = ReaderGuard {
        task,
        alive: alive.clone(),
    };
    let mut controller = None;
    let mut observation = None;
    let mut requests = HashSet::new();
    while let Some(request) = receiver.recv().await {
        if !alive.load(Ordering::Acquire) {
            break;
        }
        let id = request.id.clone();
        let result = process_request(
            request,
            &state,
            &alive,
            &mut controller,
            &mut observation,
            &mut requests,
        )
        .await;
        let reply = HelperReply {
            version: HELPER_PROTOCOL_VERSION,
            id,
            result: result.map_err(|e: anyhow::Error| e.to_string()),
        };
        let bytes = serde_json::to_vec(&reply)?;
        anyhow::ensure!(
            !bytes.is_empty() && bytes.len() <= MAX_IPC_MESSAGE_BYTES,
            "Desktop reply exceeds size limit"
        );
        let write = async {
            writer
                .write_all(&(bytes.len() as u32).to_be_bytes())
                .await?;
            writer.write_all(&bytes).await
        };
        tokio::time::timeout(Duration::from_secs(5), write)
            .await
            .context("Desktop reply timed out")??;
    }
    Ok(())
}
async fn process_request(
    request: HelperRequest,
    state: &State,
    alive: &Arc<AtomicBool>,
    controller: &mut Option<tokio::sync::OwnedSemaphorePermit>,
    observation: &mut Option<Observation>,
    requests: &mut HashSet<Uuid>,
) -> Result<Value> {
    anyhow::ensure!(
        request.version == HELPER_PROTOCOL_VERSION,
        "Desktop helper protocol version mismatch"
    );
    let parsed = Uuid::parse_str(&request.id).context("Invalid helper request ID")?;
    anyhow::ensure!(
        requests.len() < 4096 && requests.insert(parsed),
        "Duplicate request ID or connection request limit reached"
    );
    if matches!(request.operation, HelperOperation::Status) {
        return Ok(state.status());
    }
    anyhow::ensure!(
        state.ready(),
        "Desktop sharing is not approved or the session is unavailable"
    );
    if controller.is_none() {
        *controller = Some(
            state
                .controller
                .clone()
                .try_acquire_owned()
                .context("Another executor currently controls this desktop")?,
        );
    }
    let desktop = state.desktop.lock().await;
    let desktop = desktop.as_ref().context("Desktop sharing is unavailable")?;
    match request.operation {
        HelperOperation::Status => unreachable!(),
        HelperOperation::Screenshot => {
            *observation = None;
            let image = desktop.screenshot().await?;
            anyhow::ensure!(
                alive.load(Ordering::Acquire),
                "Controller disconnected during capture"
            );
            let mut value = frame_value(image, &state.generation, observation);
            value["supported_inputs"] =
                serde_json::to_value(desktop.health().input_capabilities())?;
            Ok(value)
        }
        HelperOperation::Input {
            observation_id,
            input,
        } => {
            input.validate()?;
            let source = observation
                .take()
                .context("Take a fresh screenshot before input")?;
            anyhow::ensure!(
                source.id == observation_id && source.at.elapsed() <= Duration::from_secs(30),
                "Desktop observation is stale; take a fresh screenshot"
            );
            let outcome = desktop
                .input(input, source.dimensions, alive.clone())
                .await?;
            if !outcome.post_input_frame_ready {
                return Ok(
                    json!({"done":true,"observation_error":"Input completed, but no new frame arrived. Take a fresh screenshot; never replay the input."}),
                );
            }
            match desktop.screenshot().await {
                Ok(image) => {
                    let mut value = frame_value(image, &state.generation, observation);
                    value["supported_inputs"] =
                        serde_json::to_value(desktop.health().input_capabilities())?;
                    value["done"] = json!(true);
                    Ok(value)
                }
                Err(_) => Ok(
                    json!({"done":true,"observation_error":"Input completed, but capture failed. Take a fresh screenshot; never replay the input."}),
                ),
            }
        }
    }
}

fn frame_value(
    image: DesktopScreenshot,
    generation: &str,
    observation: &mut Option<Observation>,
) -> Value {
    let id = Uuid::new_v4().to_string();
    let dimensions = image.dimensions();
    *observation = Some(Observation {
        id: id.clone(),
        dimensions,
        at: Instant::now(),
    });
    let mut value = image.into_tool_value();
    value["observation_id"] = json!(id);
    value["generation"] = json!(generation);
    value["coordinate_space"] = json!("screenshot_pixels");
    value
}
async fn read_request(reader: &mut (impl AsyncRead + Unpin)) -> Result<HelperRequest> {
    let length = reader.read_u32().await? as usize;
    // Requests contain bounded text/actions, never image bytes.
    anyhow::ensure!(
        length > 0 && length <= 64 * 1024,
        "Desktop request exceeds size limit"
    );
    let mut bytes = vec![0; length];
    reader.read_exact(&mut bytes).await?;
    serde_json::from_slice(&bytes).context("Invalid desktop helper request")
}
async fn shutdown_signal() -> Result<()> {
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    tokio::select! { result=tokio::signal::ctrl_c()=>{result?;}, _=terminate.recv()=>{} }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncWriteExt;

    fn unapproved_state() -> State {
        State {
            desktop: tokio::sync::Mutex::new(None),
            health: Mutex::new(None),
            available: true,
            generation: Uuid::new_v4().to_string(),
            controller: Arc::new(Semaphore::new(1)),
            clients: Mutex::new(Vec::new()),
        }
    }

    #[tokio::test]
    async fn status_needs_no_controller_and_never_grants_access() {
        let state = unapproved_state();
        let _existing_controller = state.controller.clone().acquire_owned().await.unwrap();
        let request = HelperRequest {
            version: HELPER_PROTOCOL_VERSION,
            id: Uuid::new_v4().to_string(),
            operation: HelperOperation::Status,
        };
        let mut controller = None;
        let mut observation = None;
        let mut requests = HashSet::new();
        let alive = Arc::new(AtomicBool::new(true));
        let value = process_request(
            request.clone(),
            &state,
            &alive,
            &mut controller,
            &mut observation,
            &mut requests,
        )
        .await
        .unwrap();
        assert_eq!(value["available"], true);
        assert_eq!(value["granted"], false);
        assert!(controller.is_none());
        assert!(
            process_request(
                request,
                &state,
                &alive,
                &mut controller,
                &mut observation,
                &mut requests
            )
            .await
            .is_err()
        );
    }

    #[tokio::test]
    async fn unapproved_capture_does_not_acquire_a_controller_lease() {
        let state = unapproved_state();
        let request = HelperRequest {
            version: HELPER_PROTOCOL_VERSION,
            id: Uuid::new_v4().to_string(),
            operation: HelperOperation::Screenshot,
        };
        let mut controller = None;
        let result = process_request(
            request,
            &state,
            &Arc::new(AtomicBool::new(true)),
            &mut controller,
            &mut None,
            &mut HashSet::new(),
        )
        .await;
        assert!(result.unwrap_err().to_string().contains("not approved"));
        assert!(controller.is_none());
        assert_eq!(state.controller.available_permits(), 1);
    }

    #[tokio::test]
    async fn rejects_oversized_request_before_reading_body() {
        let (mut writer, mut reader) = tokio::io::duplex(8);
        writer
            .write_all(&(64_u32 * 1024 + 1).to_be_bytes())
            .await
            .unwrap();
        let error = read_request(&mut reader).await.unwrap_err();
        assert!(error.to_string().contains("size limit"));
    }

    #[tokio::test]
    async fn rejects_truncated_request_without_waiting_for_another_message() {
        let (mut writer, mut reader) = tokio::io::duplex(32);
        writer.write_all(&20_u32.to_be_bytes()).await.unwrap();
        writer.write_all(b"{}").await.unwrap();
        drop(writer);
        assert!(read_request(&mut reader).await.is_err());
    }

    #[tokio::test]
    async fn request_framing_preserves_sequential_messages() {
        let first = HelperRequest {
            version: HELPER_PROTOCOL_VERSION,
            id: Uuid::new_v4().to_string(),
            operation: HelperOperation::Status,
        };
        let second = HelperRequest {
            version: HELPER_PROTOCOL_VERSION,
            id: Uuid::new_v4().to_string(),
            operation: HelperOperation::Screenshot,
        };
        let (mut writer, mut reader) = tokio::io::duplex(1024);
        for request in [&first, &second] {
            let bytes = serde_json::to_vec(request).unwrap();
            writer
                .write_all(&(bytes.len() as u32).to_be_bytes())
                .await
                .unwrap();
            writer.write_all(&bytes).await.unwrap();
        }
        assert_eq!(read_request(&mut reader).await.unwrap().id, first.id);
        assert_eq!(read_request(&mut reader).await.unwrap().id, second.id);
    }
}

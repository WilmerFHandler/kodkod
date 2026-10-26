//! An outbound executor, never an agent loop. Permissions remain local.
use crate::{computer::AppAutomationBackend, protocol::*};
use anyhow::{Context, Result};
use base64::Engine;
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering},
    },
    time::Duration,
};

/// Caller-owned connection and storage configuration; no model credentials are required.
#[derive(Clone)]
pub struct ExecutorConfig {
    pub server: String,
    pub token: String,
    pub state_dir: PathBuf,
    pub consent_file: PathBuf,
    pub name: String,
    pub capability_mask: u8,
    pub allow_insecure_http: bool,
    pub desktop_helper_socket: Option<PathBuf>,
    pub env_remove: Vec<std::ffi::OsString>,
}
impl ExecutorConfig {
    pub fn new(server: String, token: String, state_dir: PathBuf) -> Self {
        Self {
            server,
            token,
            consent_file: state_dir.join("access.json"),
            state_dir,
            name: device_name(),
            capability_mask: 7,
            allow_insecure_http: false,
            desktop_helper_socket: None,
            env_remove: Vec::new(),
        }
    }
    pub fn validate(&self) -> Result<()> {
        let url = url::Url::parse(&self.server)?;
        anyhow::ensure!(
            url.username().is_empty()
                && url.password().is_none()
                && url.query().is_none()
                && url.fragment().is_none(),
            "Server URL must not contain credentials, query or fragment"
        );
        anyhow::ensure!(
            url.scheme() == "https"
                || (url.scheme() == "http"
                    && (self.allow_insecure_http
                        || matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "[::1]")))),
            "Use HTTPS, or explicitly allow HTTP for a trusted private connection"
        );
        anyhow::ensure!(
            !self.token.trim().is_empty()
                && self.token.len() <= 8192
                && !self.token.chars().any(char::is_control),
            "Invalid executor credential"
        );
        anyhow::ensure!(
            self.state_dir.is_absolute() && self.consent_file.is_absolute(),
            "Executor state paths must be absolute"
        );
        anyhow::ensure!(
            self.capability_mask & !7 == 0,
            "Invalid capability selection"
        );
        if let Some(socket) = &self.desktop_helper_socket {
            anyhow::ensure!(
                socket.is_absolute(),
                "Desktop helper socket path must be absolute"
            );
            #[cfg(not(target_os = "linux"))]
            anyhow::bail!("The Wayland desktop helper is supported only on Linux");
        }
        anyhow::ensure!(
            !self.name.trim().is_empty()
                && self.name.len() <= 200
                && !self.name.chars().any(char::is_control),
            "Invalid executor name"
        );
        Ok(())
    }
}

pub struct Executor {
    access: Arc<crate::access::DeviceAccess>,
    alive: Arc<AtomicBool>,
    transport_error: Arc<std::sync::Mutex<Option<String>>>,
    worker: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
}
impl Executor {
    pub fn start(config: ExecutorConfig) -> Self {
        let base = config.server.clone();
        let token = config.token.clone();
        let this = Self {
            access: Arc::new(crate::access::DeviceAccess::load(
                config.consent_file.clone(),
                &base,
                &token,
            )),
            alive: Arc::new(AtomicBool::new(true)),
            transport_error: Arc::new(std::sync::Mutex::new(None)),
            worker: Default::default(),
        };
        let access = this.access.clone();
        let alive = this.alive.clone();
        let transport_error = this.transport_error.clone();
        let worker = std::thread::spawn(move || {
            let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            else {
                access.revoke();
                alive.store(false, Ordering::Release);
                *transport_error.lock().unwrap() =
                    Some("Could not initialize executor runtime".into());
                return;
            };
            runtime.block_on(async move {
                if let Err(error) = run(
                    config,
                    access.clone(),
                    alive.clone(),
                    transport_error.clone(),
                )
                .await
                {
                    *transport_error.lock().unwrap() = Some(
                        "Executor connection stopped; restart the executor to reconnect.".into(),
                    );
                    access.revoke();
                    alive.store(false, Ordering::Release);
                    eprintln!("Device executor stopped: {error}");
                }
            });
        });
        *this.worker.lock().unwrap() = Some(worker);
        this
    }
    pub async fn join(&self) {
        let worker = self.worker.lock().unwrap().take();
        if let Some(worker) = worker {
            let _ = tokio::task::spawn_blocking(move || worker.join()).await;
        }
    }
    pub fn is_running(&self) -> bool {
        self.alive.load(Ordering::Acquire)
    }
    pub fn enabled(&self) -> bool {
        self.access.enabled()
    }
    pub fn error(&self) -> Option<String> {
        self.access.error()
    }
    pub fn connection_error(&self) -> Option<String> {
        self.transport_error.lock().unwrap().clone()
    }
    pub fn set_enabled(&self, enabled: bool) -> Result<()> {
        self.access.set_enabled(enabled)
    }
    pub fn revoke(&self) {
        self.access.revoke();
    }
    pub fn shutdown(&self) {
        if self.alive.load(Ordering::Acquire) {
            self.revoke();
            self.alive.store(false, Ordering::Release);
        }
    }
}
impl Drop for Executor {
    fn drop(&mut self) {
        self.shutdown();
    }
}
pub fn available_capabilities() -> Vec<DeviceCapability> {
    available_capabilities_with_helper(None)
}
pub fn available_capabilities_with_helper(
    desktop_helper_socket: Option<&Path>,
) -> Vec<DeviceCapability> {
    capabilities(
        &AtomicU8::new(7),
        crate::platform::app_backend().as_ref(),
        7,
        desktop_helper_socket,
    )
}
fn capabilities(
    allowed: &AtomicU8,
    backend: &dyn AppAutomationBackend,
    mask: u8,
    desktop_helper_socket: Option<&Path>,
) -> Vec<DeviceCapability> {
    #[cfg(target_os = "macos")]
    let _ = desktop_helper_socket;
    let bits = allowed.load(Ordering::Acquire) & mask;
    if bits & 6 == 0 {
        return if bits & 1 != 0 {
            vec![DeviceCapability::Commands]
        } else {
            Vec::new()
        };
    }
    #[allow(unused_mut)]
    let mut capabilities = crate::access::capabilities(bits, backend.permissions());
    #[cfg(not(target_os = "macos"))]
    if bits & 4 != 0 && crate::platform::desktop_available_with_helper(desktop_helper_socket) {
        capabilities.push(DeviceCapability::Desktop);
    }
    capabilities
}
#[derive(Debug)]
struct HttpFailure(reqwest::StatusCode);
impl std::fmt::Display for HttpFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Device service returned {}", self.0)
    }
}
impl std::error::Error for HttpFailure {}
fn receipt_scope(base: &str, token: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    hash.update(b"handler-device-receipts-v1\0");
    hash.update(base.as_bytes());
    hash.update([0]);
    hash.update(token.as_bytes());
    format!("{:x}", hash.finalize())
}
async fn post<T: serde::Serialize, R: serde::de::DeserializeOwned>(
    client: &reqwest::Client,
    base: &str,
    token: &str,
    path: &str,
    body: &T,
) -> Result<R> {
    let response = client
        .post(format!("{base}/v1/device-executor/{path}"))
        .header(EXECUTOR_PROTOCOL_HEADER, EXECUTOR_PROTOCOL_VERSION)
        .bearer_auth(token.trim())
        .json(body)
        .send()
        .await?;
    if !response.status().is_success() {
        return Err(HttpFailure(response.status()).into());
    }
    let mut response = response;
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        anyhow::ensure!(
            bytes.len() + chunk.len() <= 12 * 1024 * 1024,
            "Device response too large"
        );
        bytes.extend_from_slice(&chunk);
    }
    Ok(serde_json::from_slice(&bytes)?)
}
async fn announce(
    client: &reqwest::Client,
    base: &str,
    token: &str,
    hello: &DeviceHello,
) -> Result<DeviceConnection> {
    match post(client, base, token, "connect", hello).await {
        Err(error)
            if hello.executor_id.is_some()
                && error.downcast_ref::<HttpFailure>().is_some_and(|failure| {
                    failure.0 == reqwest::StatusCode::UNPROCESSABLE_ENTITY
                }) =>
        {
            // Older services reject the additive identity field before admission.
            // Negotiation changes neither the existing consent nor its capabilities.
            let mut legacy = hello.clone();
            legacy.executor_id = None;
            post(client, base, token, "connect", &legacy).await
        }
        response => response,
    }
}

async fn acquire_instance_lock(file: &std::fs::File, alive: &AtomicBool) -> Result<()> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(12);
    loop {
        anyhow::ensure!(
            alive.load(Ordering::Acquire),
            "Executor stopped before connection"
        );
        match file.try_lock() {
            Ok(()) => return Ok(()),
            Err(std::fs::TryLockError::WouldBlock) if tokio::time::Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(100)).await
            }
            Err(error) => {
                return Err(anyhow::anyhow!(error).context(
                    "Another executor is already using this server/account state directory",
                ));
            }
        }
    }
}

async fn run(
    config: ExecutorConfig,
    access: Arc<crate::access::DeviceAccess>,
    alive: Arc<AtomicBool>,
    transport_error: Arc<std::sync::Mutex<Option<String>>>,
) -> Result<()> {
    config.validate()?;
    let base = config.server.clone();
    let token = config.token.clone();
    let allowed = access.allowed.clone();
    let generation = access.generation.clone();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .redirect(reqwest::redirect::Policy::none())
        .build()?;
    let backend = crate::platform::app_backend();
    let mut session = backend.app_automation_session();
    let mut desktop = Arc::new(crate::platform::new_desktop(
        config.desktop_helper_socket.clone(),
    ));
    let mut bound_attachment = None;
    let mut connection = None;
    let mut view_worker: Option<crate::view::ViewWorker> = None;
    let mut active: Option<tokio::task::JoinHandle<DeviceReceipt>> = None;
    let mut active_identity: Option<(String, String)> = None;
    let journal = config
        .state_dir
        .join("device-attempts")
        .join(receipt_scope(&base, &token));
    std::fs::create_dir_all(&journal)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&journal, std::fs::Permissions::from_mode(0o700))?;
    }
    let mut lock_options = std::fs::OpenOptions::new();
    lock_options
        .read(true)
        .write(true)
        .create(true)
        .truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        lock_options.mode(0o600);
    }
    let instance_lock = lock_options.open(journal.join("executor.lock"))?;
    acquire_instance_lock(&instance_lock, &alive).await?;
    let executor_id = crate::access::executor_id(&journal.join("executor-id"))?;
    let mut receipts = load_receipts(&journal)?;
    let mut pending_receipt: Option<PendingReceiptSave> = None;
    #[cfg(target_os = "macos")]
    let mut activity = None;
    while alive.load(Ordering::Acquire) {
        #[cfg(target_os = "macos")]
        {
            if allowed.load(Ordering::Acquire) != 0 {
                activity.get_or_insert_with(crate::macos::ExecutorActivity::begin);
            } else {
                activity = None;
            }
        }
        if active.as_ref().is_some_and(|a| a.is_finished()) {
            let task = active.take().unwrap();
            match task.await {
                Ok(receipt) => {
                    active_identity = None;
                    retain_receipt_or_queue(
                        &journal,
                        receipt,
                        &mut receipts,
                        &mut pending_receipt,
                        &transport_error,
                    );
                }
                Err(_) => {
                    if let Some((connection, attempt)) = active_identity.take() {
                        retain_receipt_or_queue(
                            &journal,
                            DeviceReceipt {
                                connection,
                                attempt,
                                result: Err(
                                    "Executor action task stopped unexpectedly; outcome may be uncertain. Inspect before repeating.".into(),
                                ),
                            },
                            &mut receipts,
                            &mut pending_receipt,
                            &transport_error,
                        );
                    }
                }
            }
        }
        retry_pending_receipt_save(
            &journal,
            &mut receipts,
            &mut pending_receipt,
            &transport_error,
        );
        if let Some(receipt) = receipts.pop_front() {
            let id = uuid::Uuid::parse_str(&receipt.attempt)?;
            let path = journal.join(format!("{id}.receipt.json"));
            match post::<_, Value>(&client, &base, &token, "receipt", &receipt).await {
                Ok(_) => std::fs::remove_file(path)?,
                Err(error)
                    if error
                        .downcast_ref::<HttpFailure>()
                        .is_some_and(|e| matches!(e.0.as_u16(), 400 | 404 | 409)) =>
                {
                    std::fs::rename(path, journal.join(format!("{id}.rejected.json")))?;
                    eprintln!("A device receipt was retained for reconciliation after rejection");
                }
                Err(_) => receipts.push_back(receipt),
            }
        }
        if connection.is_none() {
            let hello = DeviceHello {
                executor_id: Some(executor_id),
                name: config.name.clone(),
                capabilities: capabilities(
                    &allowed,
                    backend.as_ref(),
                    config.capability_mask,
                    config.desktop_helper_socket.as_deref(),
                ),
                grant_generation: generation.load(Ordering::Acquire),
            };
            match announce(&client, &base, &token, &hello).await {
                Ok(c) => {
                    view_worker = None;
                    connection = Some(c.connection);
                    bound_attachment = None;
                    session = backend.app_automation_session();
                    desktop = Arc::new(crate::platform::new_desktop(
                        config.desktop_helper_socket.clone(),
                    ));
                }
                Err(error) => {
                    record_transport_failure(&journal, "connect", &error, &transport_error);
                    if pending_receipt.is_some() {
                        report_receipt_storage_blocked(&transport_error);
                    }
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    continue;
                }
            }
        }
        let id = connection.as_ref().unwrap().clone();
        // Capture consent before the request. A delayed response cannot inherit
        // a newer grant after disable/re-enable.
        let epoch = generation.load(Ordering::Acquire);
        let current_capabilities = capabilities(
            &allowed,
            backend.as_ref(),
            config.capability_mask,
            config.desktop_helper_socket.as_deref(),
        );
        if !current_capabilities.contains(&DeviceCapability::Desktop) {
            view_worker = None;
        }
        if current_capabilities.contains(&DeviceCapability::Desktop)
            && view_worker
                .as_ref()
                .is_none_or(|worker| worker.is_finished())
        {
            view_worker = Some(crate::view::ViewWorker::start(
                client.clone(),
                base.clone(),
                token.clone(),
                id.clone(),
                epoch,
                allowed.clone(),
                generation.clone(),
                alive.clone(),
                config.capability_mask,
                config.desktop_helper_socket.clone(),
            ));
        }
        let poll = DevicePoll {
            connection: id.clone(),
            capabilities: current_capabilities,
            ready: active.is_none() && pending_receipt.is_none(),

            grant_generation: epoch,
            active_attempt: active_identity.as_ref().map(|(_, attempt)| attempt.clone()),
        };
        let response = post::<_, DevicePollResult>(&client, &base, &token, "poll", &poll).await;
        match &response {
            Ok(_) if pending_receipt.is_none() => *transport_error.lock().unwrap() = None,
            Ok(_) => {}
            Err(error) => record_transport_failure(&journal, "poll", error, &transport_error),
        }
        if pending_receipt.is_some() {
            report_receipt_storage_blocked(&transport_error);
        }
        let authorized = response.as_ref().is_ok_and(|p| p.active_authorized)
            && generation.load(Ordering::Acquire) == epoch;
        if !authorized && let Some(task) = active.take() {
            generation.fetch_add(1, Ordering::AcqRel);
            let receipt = settle_action_task(
                task,
                active_identity.take(),
                "Device authorization was lost before the action result was available; outcome may be uncertain. Inspect before repeating.",
            )
            .await;
            if let Some(receipt) = receipt {
                active_identity = None;
                retain_receipt_or_queue(
                    &journal,
                    receipt,
                    &mut receipts,
                    &mut pending_receipt,
                    &transport_error,
                );
            }
            bound_attachment = None;
            session = backend.app_automation_session();
            desktop = Arc::new(crate::platform::new_desktop(
                config.desktop_helper_socket.clone(),
            ));
        }
        match response {
            Ok(DevicePollResult {
                operation: Some(op),
                ..
            }) if pending_receipt.is_none() => {
                if generation.load(Ordering::Acquire) != epoch {
                    let receipt = DeviceReceipt {
                        connection: id,
                        attempt: op.attempt,
                        result: Err(
                            "Local consent changed before execution; action was not executed"
                                .into(),
                        ),
                    };
                    retain_receipt_or_queue(
                        &journal,
                        receipt,
                        &mut receipts,
                        &mut pending_receipt,
                        &transport_error,
                    );
                    continue;
                }
                if bound_attachment.as_ref() != Some(&op.attachment) {
                    session = backend.app_automation_session();
                    desktop = Arc::new(crate::platform::new_desktop(
                        config.desktop_helper_socket.clone(),
                    ));
                    bound_attachment = Some(op.attachment.clone());
                }
                match &op.operation {
                    DeviceAction::RunCommand { .. } => {
                        desktop.invalidate();
                        session = backend.app_automation_session();
                    }
                    DeviceAction::ActOnApp { .. } => desktop.invalidate(),
                    DeviceAction::ComputerInput { .. } => {
                        session = backend.app_automation_session()
                    }
                    _ => {}
                }
                active_identity = Some((id.clone(), op.attempt.clone()));
                let action_session = session.clone();
                if matches!(op.operation, DeviceAction::ActOnApp { .. }) {
                    session = backend.app_automation_session();
                }
                let session = action_session;
                let desktop = desktop.clone();
                let allowed = allowed.clone();
                let generation = generation.clone();
                let alive = alive.clone();
                let backend = backend.clone();
                let journal = journal.clone();
                let access = ExecutionAccess {
                    session,
                    allowed: allowed.clone(),
                    backend,
                    desktop,
                    generation: generation.clone(),
                    epoch,
                    capability_mask: config.capability_mask,
                    desktop_helper_socket: config.desktop_helper_socket.clone(),
                    env_remove: config.env_remove.clone(),
                };
                active = Some(tokio::spawn(async move {
                    let revoked = async {
                        loop {
                            if generation.load(Ordering::Acquire) != epoch
                                || !alive.load(Ordering::Acquire)
                            {
                                break;
                            }
                            tokio::time::sleep(Duration::from_millis(50)).await;
                        }
                    };
                    let result = tokio::select! {biased;
                        result=execute_once(op.clone(),journal,access)=>result,
                        _=revoked=>Err(anyhow::anyhow!("Local access revoked; action may have completed")),
                    };
                    DeviceReceipt {
                        connection: id,
                        attempt: op.attempt,
                        result: result.map_err(|e| e.to_string()),
                    }
                }));
            }
            Ok(_) => {}
            Err(error) => {
                // A lost HTTP response is not a new device or a revoked lease.
                // In-flight work is still aborted above; never replay an action.
                if error
                    .downcast_ref::<HttpFailure>()
                    .is_some_and(|failure| matches!(failure.0.as_u16(), 401 | 403 | 409))
                {
                    view_worker = None;
                    connection = None;
                }
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    if let Some(task) = active {
        let receipt = settle_action_task(
            task,
            active_identity.take(),
            "Executor stopped before the action result was available; outcome may be uncertain. Inspect before repeating.",
        )
        .await;
        if let Some(receipt) = receipt {
            retain_receipt_or_queue(
                &journal,
                receipt,
                &mut receipts,
                &mut pending_receipt,
                &transport_error,
            );
        }
    }
    retry_pending_receipt_save_until_shutdown(
        &journal,
        &mut receipts,
        &mut pending_receipt,
        &transport_error,
    )
    .await;
    Ok(())
}

async fn settle_action_task(
    mut task: tokio::task::JoinHandle<DeviceReceipt>,
    identity: Option<(String, String)>,
    uncertain_result: &'static str,
) -> Option<DeviceReceipt> {
    match tokio::time::timeout(Duration::from_millis(100), &mut task).await {
        Ok(Ok(receipt)) => Some(receipt),
        Ok(Err(_)) => identity.map(|(connection, attempt)| DeviceReceipt {
            connection,
            attempt,
            result: Err(uncertain_result.into()),
        }),
        Err(_) => {
            task.abort();
            let _ = task.await;
            identity.map(|(connection, attempt)| DeviceReceipt {
                connection,
                attempt,
                result: Err(uncertain_result.into()),
            })
        }
    }
}

/// Store metadata only: never the bearer, response body, command, or input value.
fn record_transport_failure(
    journal: &std::path::Path,
    phase: &str,
    error: &anyhow::Error,
    status: &std::sync::Mutex<Option<String>>,
) {
    let code = if let Some(failure) = error.downcast_ref::<HttpFailure>() {
        format!("http_{}", failure.0.as_u16())
    } else if let Some(failure) = error.downcast_ref::<reqwest::Error>() {
        if failure.is_timeout() {
            "timeout".into()
        } else {
            "network".into()
        }
    } else if error.is::<serde_json::Error>() {
        "invalid_response".into()
    } else {
        "transport".into()
    };
    *status.lock().unwrap() =
        Some("Connection to your server was interrupted. Reconnecting…".into());
    let record = json!({
        "phase": phase, "code": code,
        "at_unix": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs(),
    });
    if let Ok(bytes) = serde_json::to_vec(&record) {
        let _ = crate::access::save_private(&journal.join("last-transport-error.json"), &bytes);
    }
}

#[cfg(unix)]
fn device_name() -> String {
    let mut buffer = [0u8; 128];
    if unsafe { libc::gethostname(buffer.as_mut_ptr().cast(), buffer.len()) } == 0 {
        let end = buffer.iter().position(|b| *b == 0).unwrap_or(buffer.len());
        if let Ok(name) = std::str::from_utf8(&buffer[..end]) {
            return name.to_owned();
        }
    }
    "This computer".into()
}
#[cfg(not(unix))]
fn device_name() -> String {
    std::env::var("COMPUTERNAME").unwrap_or_else(|_| "This computer".into())
}
fn save_receipt(journal: &std::path::Path, receipt: &DeviceReceipt) -> Result<()> {
    use std::io::Write;
    let id = uuid::Uuid::parse_str(&receipt.attempt)?;
    let path = journal.join(format!("{id}.receipt.json"));
    let temporary = journal.join(format!("{id}.receipt.tmp"));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(temporary.clone())?;
    file.write_all(&serde_json::to_vec(receipt)?)?;
    file.sync_all()?;
    std::fs::rename(temporary, path)?;
    crate::access::sync_directory(journal)?;
    Ok(())
}
fn load_receipts(journal: &std::path::Path) -> Result<std::collections::VecDeque<DeviceReceipt>> {
    let mut receipts = std::collections::VecDeque::new();
    for entry in std::fs::read_dir(journal)? {
        let entry = entry?;
        if entry
            .file_name()
            .to_string_lossy()
            .ends_with(".receipt.json")
        {
            let bytes = std::fs::read(entry.path())?;
            anyhow::ensure!(
                bytes.len() <= 12 * 1024 * 1024,
                "Stored device receipt is too large"
            );
            receipts.push_back(serde_json::from_slice(&bytes)?);
        }
    }
    Ok(receipts)
}

const RECEIPT_SAVE_RETRY_INITIAL: Duration = Duration::from_millis(250);
const RECEIPT_SAVE_RETRY_MAX: Duration = Duration::from_secs(5);
const RECEIPT_SAVE_SHUTDOWN_WINDOW: Duration = Duration::from_secs(2);
const RECEIPT_STORAGE_BLOCKED_STATUS: &str = "Executor storage is unavailable. A completed action result is retained in memory and will be saved before new work is accepted.";

struct PendingReceiptSave {
    receipt: DeviceReceipt,
    retry_at: tokio::time::Instant,
    backoff: Duration,
}

fn report_receipt_storage_blocked(status: &std::sync::Mutex<Option<String>>) {
    *status.lock().unwrap() = Some(RECEIPT_STORAGE_BLOCKED_STATUS.into());
}

fn retain_receipt_or_queue(
    journal: &std::path::Path,
    receipt: DeviceReceipt,
    receipts: &mut std::collections::VecDeque<DeviceReceipt>,
    pending: &mut Option<PendingReceiptSave>,
    status: &std::sync::Mutex<Option<String>>,
) {
    debug_assert!(
        pending.is_none(),
        "only one completed receipt may await storage"
    );
    match save_receipt(journal, &receipt) {
        Ok(()) => receipts.push_back(receipt),
        Err(_) => {
            eprintln!(
                "A completed executor result could not be saved locally; it is retained in memory while storage recovers."
            );
            report_receipt_storage_blocked(status);
            *pending = Some(PendingReceiptSave {
                receipt,
                retry_at: tokio::time::Instant::now() + RECEIPT_SAVE_RETRY_INITIAL,
                backoff: RECEIPT_SAVE_RETRY_INITIAL.saturating_mul(2),
            });
        }
    }
}

fn retry_pending_receipt_save(
    journal: &std::path::Path,
    receipts: &mut std::collections::VecDeque<DeviceReceipt>,
    pending: &mut Option<PendingReceiptSave>,
    status: &std::sync::Mutex<Option<String>>,
) {
    let Some(waiting) = pending.as_mut() else {
        return;
    };
    if tokio::time::Instant::now() < waiting.retry_at {
        return;
    }
    match save_receipt(journal, &waiting.receipt) {
        Ok(()) => {
            let waiting = pending.take().expect("pending receipt was just saved");
            receipts.push_back(waiting.receipt);
        }
        Err(_) => {
            let waiting = pending
                .as_mut()
                .expect("pending receipt remains after failure");
            waiting.retry_at = tokio::time::Instant::now() + waiting.backoff;
            waiting.backoff = waiting
                .backoff
                .saturating_mul(2)
                .min(RECEIPT_SAVE_RETRY_MAX);
            report_receipt_storage_blocked(status);
            eprintln!(
                "Executor receipt storage is still unavailable; retrying with bounded backoff."
            );
        }
    }
}

async fn retry_pending_receipt_save_until_shutdown(
    journal: &std::path::Path,
    receipts: &mut std::collections::VecDeque<DeviceReceipt>,
    pending: &mut Option<PendingReceiptSave>,
    status: &std::sync::Mutex<Option<String>>,
) {
    if let Some(waiting) = pending.as_mut() {
        waiting.retry_at = tokio::time::Instant::now();
    }
    let deadline = tokio::time::Instant::now() + RECEIPT_SAVE_SHUTDOWN_WINDOW;
    while pending.is_some() {
        let now = tokio::time::Instant::now();
        if now >= deadline {
            break;
        }
        let remaining = deadline.duration_since(now);
        let retry_after = pending
            .as_ref()
            .expect("pending receipt checked above")
            .retry_at
            .saturating_duration_since(now)
            .min(remaining);
        if !retry_after.is_zero() {
            tokio::time::sleep(retry_after).await;
        }
        if tokio::time::Instant::now() >= deadline {
            break;
        }
        retry_pending_receipt_save(journal, receipts, pending, status);
    }
    if pending.is_some() {
        *status.lock().unwrap() = Some(
            "Executor stopped before a completed action result could be saved locally; its outcome may be uncertain. Inspect before repeating.".into(),
        );
        eprintln!(
            "Executor stopped before a completed result could be saved; its outcome may be uncertain."
        );
    }
}

fn write_attempt_summary(record: &mut std::fs::File, succeeded: bool) -> std::io::Result<()> {
    use std::io::{Seek, SeekFrom, Write};
    let summary = if succeeded {
        b"{\"state\":\"completed\"}".as_slice()
    } else {
        b"{\"state\":\"failed\"}".as_slice()
    };
    record.set_len(0)?;
    record.seek(SeekFrom::Start(0))?;
    record.write_all(summary)?;
    record.sync_all()
}

fn preserve_effect_result<T>(
    result: Result<T>,
    update_summary: impl FnOnce(bool) -> std::io::Result<()>,
) -> Result<T> {
    if update_summary(result.is_ok()).is_err() {
        eprintln!(
            "The action finished, but its local attempt summary could not be updated; the original result is retained for delivery."
        );
    }
    result
}

#[derive(Clone)]
struct ExecutionAccess {
    session: Arc<dyn crate::computer::AppAutomationSession>,
    allowed: Arc<AtomicU8>,
    backend: Arc<dyn AppAutomationBackend>,
    desktop: Arc<crate::platform::Desktop>,
    generation: Arc<AtomicU64>,
    epoch: u64,
    capability_mask: u8,
    desktop_helper_socket: Option<PathBuf>,
    env_remove: Vec<std::ffi::OsString>,
}

async fn execute_once(
    op: DeviceOperation,
    journal: PathBuf,
    access: ExecutionAccess,
) -> Result<Value> {
    let ExecutionAccess {
        session,
        allowed,
        backend,
        desktop,
        generation,
        epoch,
        capability_mask,
        desktop_helper_socket,
        env_remove,
    } = access;
    op.operation.validate()?;
    anyhow::ensure!(
        generation.load(Ordering::Acquire) == epoch,
        "Local consent changed before execution"
    );
    anyhow::ensure!(
        capabilities(
            &allowed,
            backend.as_ref(),
            capability_mask,
            desktop_helper_socket.as_deref(),
        )
        .contains(&op.operation.capability()),
        "Device capability was revoked locally"
    );
    // Server-generated identifiers cannot select arbitrary local paths.
    let id = uuid::Uuid::parse_str(&op.attempt)?;
    let path = journal.join(format!("{id}.json"));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    use std::io::Write;
    let mut record = options
        .open(&path)
        .context("Attempt already exists; outcome may be uncertain, refusing replay")?;
    record.write_all(b"{\"state\":\"started\"}")?;
    record.sync_all()?;
    crate::access::sync_directory(&journal)?;
    anyhow::ensure!(
        generation.load(Ordering::Acquire) == epoch
            && capabilities(
                &allowed,
                backend.as_ref(),
                capability_mask,
                desktop_helper_socket.as_deref(),
            )
            .contains(&op.operation.capability()),
        "Local consent changed before effect dispatch"
    );
    let result = match op.operation {
        DeviceAction::RunCommand { command, cwd } => {
            let mut request = crate::command::CommandRequest::shell(command, PathBuf::from(cwd));
            request.env_remove = env_remove;
            crate::command::run(request)
                .await
                .and_then(|output| Ok(serde_json::to_value(output)?))
        }
        DeviceAction::Screenshot {} => {
            anyhow::ensure!(
                generation.load(Ordering::Acquire) == epoch
                    && allowed.load(Ordering::Acquire) & 4 != 0,
                "Desktop access revoked"
            );
            desktop.screenshot().await
        }
        DeviceAction::ComputerInput {
            observation_id,
            input,
        } => {
            desktop
                .input(&observation_id, input, move || {
                    generation.load(Ordering::Acquire) == epoch
                        && allowed.load(Ordering::Acquire) & 4 != 0
                })
                .await
        }
        DeviceAction::ListApps {} => {
            let apps = session.list_apps().await?;
            apps.validate()?;
            Ok(serde_json::to_value(apps)?)
        }
        DeviceAction::ObserveApp { request } => {
            let observation = session.observe_app(request).await?;
            observation.validate()?;
            let mut metadata = observation.metadata();
            if let Some(image) = observation.screenshot {
                metadata["png"] =
                    Value::String(base64::engine::general_purpose::STANDARD.encode(image.png()));
            }
            Ok(metadata)
        }
        DeviceAction::ActOnApp { request } => session
            .act_on_app(request)
            .await
            .map(|_| json!({"done":true})),
    };
    // Keep a bounded non-secret summary; the pre-effect marker remains the
    // replay fence if updating it fails after the effect has completed.
    preserve_effect_result(result, |succeeded| {
        write_attempt_summary(&mut record, succeeded)
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn device_transport_failure_does_not_change_local_consent_controls() {
        let dir = tempfile::tempdir().unwrap();
        let access = Arc::new(crate::access::DeviceAccess::load(
            dir.path().join("private/mac-tools.json"),
            "https://fixture",
            "dummy-token",
        ));
        access.set_enabled(true).unwrap();
        let executor = Executor {
            access,
            alive: Arc::new(AtomicBool::new(true)),
            transport_error: Arc::new(std::sync::Mutex::new(Some("Reconnecting".into()))),
            worker: Default::default(),
        };
        assert!(executor.enabled());
        assert!(executor.error().is_none());
        assert!(executor.connection_error().is_some());
        executor.set_enabled(false).unwrap();
        assert!(!executor.enabled());
    }

    use super::*;
    use futures::future::BoxFuture;
    struct FakeBackend;
    impl AppAutomationBackend for FakeBackend {
        fn permissions(&self) -> crate::computer::DesktopPermissions {
            crate::computer::DesktopPermissions {
                screen_recording: crate::computer::DesktopPermissionStatus::Unsupported,
                accessibility: crate::computer::DesktopPermissionStatus::Unsupported,
            }
        }
        fn app_automation_session(&self) -> Arc<dyn crate::computer::AppAutomationSession> {
            Arc::new(FakeBackend)
        }
    }
    impl crate::computer::AppAutomationSession for FakeBackend {
        fn list_apps<'a>(&'a self) -> BoxFuture<'a, Result<crate::computer::AppList>> {
            Box::pin(async { anyhow::bail!("Unused fixture") })
        }
        fn observe_app<'a>(
            &'a self,
            _: crate::computer::ObserveAppRequest,
        ) -> BoxFuture<'a, Result<crate::computer::AppObservation>> {
            Box::pin(async { anyhow::bail!("Unused fixture") })
        }
        fn act_on_app<'a>(
            &'a self,
            _: crate::computer::AppActionRequest,
        ) -> BoxFuture<'a, Result<()>> {
            Box::pin(async { anyhow::bail!("Unused fixture") })
        }
    }
    #[cfg(target_os = "linux")]
    #[test]
    fn linux_wayland_capability_requires_local_permission_and_a_healthy_helper() {
        use std::{
            io::{Read, Write},
            os::unix::fs::PermissionsExt,
        };

        let directory = tempfile::tempdir().unwrap();
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let backend = FakeBackend;
        let allowed = AtomicU8::new(5);

        for (name, granted, expected) in [("denied", false, false), ("granted", true, true)] {
            let socket = directory.path().join(format!("{name}.sock"));
            let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
            std::fs::set_permissions(&socket, std::fs::Permissions::from_mode(0o600)).unwrap();
            let server = std::thread::spawn(move || {
                let (mut stream, _) = listener.accept().unwrap();
                let mut header = [0; 4];
                stream.read_exact(&mut header).unwrap();
                let length = u32::from_be_bytes(header) as usize;
                let mut body = vec![0; length];
                stream.read_exact(&mut body).unwrap();
                let request: crate::desktop_helper::HelperRequest =
                    serde_json::from_slice(&body).unwrap();
                assert!(matches!(
                    request.operation,
                    crate::desktop_helper::HelperOperation::Status
                ));
                let reply = crate::desktop_helper::HelperReply {
                    version: crate::desktop_helper::HELPER_PROTOCOL_VERSION,
                    id: request.id,
                    result: Ok(json!({
                        "available": true,
                        "granted": granted,
                        "generation": "fixture-generation"
                    })),
                };
                let body = serde_json::to_vec(&reply).unwrap();
                stream
                    .write_all(&(body.len() as u32).to_be_bytes())
                    .unwrap();
                stream.write_all(&body).unwrap();
            });

            let capabilities = capabilities(&allowed, &backend, 5, Some(&socket));
            assert_eq!(capabilities.contains(&DeviceCapability::Desktop), expected);
            assert!(capabilities.contains(&DeviceCapability::Commands));
            server.join().unwrap();
        }

        let commands_only = AtomicU8::new(1);
        assert!(
            !capabilities(
                &commands_only,
                &backend,
                5,
                Some(Path::new("/missing.sock"))
            )
            .contains(&DeviceCapability::Desktop)
        );
    }

    #[tokio::test]
    async fn native_command_executor_writes_only_its_owned_fixture_and_refuses_replay() {
        let dir = tempfile::tempdir().unwrap();
        let journal = dir.path().join("journal");
        std::fs::create_dir(&journal).unwrap();
        let operation = DeviceOperation {
            attempt: uuid::Uuid::new_v4().to_string(),
            attachment: "test".into(),
            operation: DeviceAction::RunCommand {
                command: if cfg!(windows) {
                    "echo handler-connected> result.txt & type result.txt"
                } else {
                    "printf handler-connected > result.txt; cat result.txt"
                }
                .into(),
                cwd: dir.path().to_str().unwrap().into(),
            },
        };
        let backend = Arc::new(FakeBackend);
        let allowed = Arc::new(AtomicU8::new(1));
        let desktop = Arc::new(crate::platform::Desktop::default());
        let generation = Arc::new(AtomicU64::new(0));
        let access = ExecutionAccess {
            session: backend.app_automation_session(),
            allowed,
            backend,
            desktop,
            generation,
            epoch: 0,
            capability_mask: 7,
            desktop_helper_socket: None,
            env_remove: Vec::new(),
        };
        let reply = execute_once(operation.clone(), journal.clone(), access.clone())
            .await
            .unwrap();
        assert_eq!(
            reply["stdout"].as_str().unwrap().trim(),
            "handler-connected"
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join("result.txt"))
                .unwrap()
                .trim(),
            "handler-connected"
        );
        assert!(
            execute_once(operation, journal, access)
                .await
                .unwrap_err()
                .to_string()
                .contains("refusing replay")
        );
    }
    #[tokio::test]
    async fn replacement_waits_for_the_previous_executor_to_release_its_lock() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("executor.lock");
        let old = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        old.lock().unwrap();
        let replacement = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let release = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            drop(old);
        });
        acquire_instance_lock(&replacement, &AtomicBool::new(true))
            .await
            .unwrap();
        release.join().unwrap();
        drop(replacement);
        let stopped = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        assert!(
            acquire_instance_lock(&stopped, &AtomicBool::new(false))
                .await
                .is_err()
        );
    }

    #[test]
    fn receipts_survive_restart_without_reexecuting_commands() {
        let dir = tempfile::tempdir().unwrap();
        let receipt = DeviceReceipt {
            connection: "old-connection".into(),
            attempt: uuid::Uuid::new_v4().to_string(),
            result: Ok(json!({"stdout":"fixture"})),
        };
        save_receipt(dir.path(), &receipt).unwrap();
        let restored = load_receipts(dir.path()).unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].attempt, receipt.attempt);
        assert_eq!(restored[0].result, receipt.result);
    }
    #[tokio::test]
    async fn completed_receipt_stays_pending_until_storage_recovers() {
        let dir = tempfile::tempdir().unwrap();
        let journal = dir.path().join("journal");
        std::fs::write(&journal, b"temporary path obstruction").unwrap();
        let receipt = DeviceReceipt {
            connection: "fixture-connection".into(),
            attempt: uuid::Uuid::new_v4().to_string(),
            result: Ok(json!({"stdout":"original output"})),
        };
        let status = std::sync::Mutex::new(None);
        let mut queued = std::collections::VecDeque::new();
        let mut pending = None;

        retain_receipt_or_queue(
            &journal,
            receipt.clone(),
            &mut queued,
            &mut pending,
            &status,
        );
        assert!(queued.is_empty());
        assert_eq!(pending.as_ref().unwrap().receipt.result, receipt.result);
        assert!(
            status
                .lock()
                .unwrap()
                .as_deref()
                .unwrap()
                .contains("storage")
        );

        std::fs::remove_file(&journal).unwrap();
        std::fs::create_dir(&journal).unwrap();
        tokio::time::sleep(RECEIPT_SAVE_RETRY_INITIAL).await;
        retry_pending_receipt_save(&journal, &mut queued, &mut pending, &status);

        assert!(pending.is_none());
        assert_eq!(queued.len(), 1);
        assert_eq!(queued[0].attempt, receipt.attempt);
        assert_eq!(queued[0].result, receipt.result);
        assert_eq!(load_receipts(&journal).unwrap()[0].result, receipt.result);
    }
    #[test]
    fn marker_write_failure_does_not_replace_the_effect_result() {
        let output = json!({"stdout":"effect completed"});
        let result = preserve_effect_result(Ok(output.clone()), |_| {
            Err(std::io::Error::other("fixture write failure"))
        });
        assert_eq!(result.unwrap(), output);
    }
    #[test]
    fn receipt_namespaces_separate_servers_and_accounts() {
        assert_ne!(
            receipt_scope("https://one", "account"),
            receipt_scope("https://two", "account")
        );
        assert_ne!(
            receipt_scope("https://one", "account"),
            receipt_scope("https://one", "other-account")
        );
    }
}

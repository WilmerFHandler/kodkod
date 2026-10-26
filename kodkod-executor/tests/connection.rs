use kodkod_executor::{
    protocol::{
        DeviceAction, DeviceCapability, DeviceConnection, DeviceHello, DeviceOperation, DevicePoll,
        DevicePollResult, DeviceReceipt,
    },
    runner::{Executor, ExecutorConfig},
};
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    io::{self, Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};
use tempfile::tempdir;
use tokio::{sync::mpsc, time::timeout};
use uuid::Uuid;

const FIXTURE_CONNECTION: &str = "fixture-connection";
const FIXTURE_TOKEN: &str = "fixture-token";
const MAX_REQUEST_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug, PartialEq)]
struct ReceiptSeen {
    connection: String,
    attempt: String,
    result: Result<Value, String>,
}

enum ServerEvent {
    Receipt {
        ordinal: usize,
        receipt: ReceiptSeen,
    },
    Failure(String),
}

#[derive(Clone, Default)]
struct ServerState {
    connect_count: usize,
    poll_count: usize,
    operation_count: usize,
    receipt_count: usize,
    auth_failures: usize,
    protocol_failures: usize,
    mismatched_connections: usize,
    saw_command_capability: bool,
    operation_delivered: bool,
    receipts: Vec<ReceiptSeen>,
}

struct FixtureServer {
    base_url: String,
    stop: Arc<AtomicBool>,
    state: Arc<Mutex<ServerState>>,
    worker: Option<thread::JoinHandle<()>>,
}

impl FixtureServer {
    fn start(
        command: String,
        cwd: PathBuf,
    ) -> io::Result<(Self, mpsc::UnboundedReceiver<ServerEvent>)> {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        listener.set_nonblocking(true)?;
        let address = listener.local_addr()?;
        let base_url = format!("http://{address}");
        let stop = Arc::new(AtomicBool::new(false));
        let state = Arc::new(Mutex::new(ServerState::default()));
        let (events_tx, events_rx) = mpsc::unbounded_channel();
        let worker_stop = stop.clone();
        let worker_state = state.clone();
        let worker = thread::spawn(move || {
            serve(
                listener,
                worker_stop,
                worker_state,
                events_tx,
                command,
                cwd,
                Uuid::new_v4().to_string(),
            );
        });
        Ok((
            Self {
                base_url,
                stop,
                state,
                worker: Some(worker),
            },
            events_rx,
        ))
    }

    fn stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            worker
                .join()
                .expect("fixture server thread should stop cleanly");
        }
    }

    fn snapshot(&self) -> ServerState {
        self.state.lock().unwrap().clone()
    }
}

impl Drop for FixtureServer {
    fn drop(&mut self) {
        self.stop();
    }
}

struct HttpRequest {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

fn serve(
    listener: TcpListener,
    stop: Arc<AtomicBool>,
    state: Arc<Mutex<ServerState>>,
    events: mpsc::UnboundedSender<ServerEvent>,
    command: String,
    cwd: PathBuf,
    attempt: String,
) {
    while !stop.load(Ordering::Acquire) {
        match listener.accept() {
            Ok((mut stream, _)) => {
                // macOS may inherit the listener's nonblocking flag on accept.
                let _ = stream.set_nonblocking(false);
                let _ = stream.set_read_timeout(Some(Duration::from_secs(3)));
                if let Err(error) =
                    handle_request(&mut stream, &state, &events, &command, &cwd, &attempt)
                {
                    let _ = events.send(ServerEvent::Failure(error.to_string()));
                    let _ = write_json(&mut stream, "500 Internal Server Error", &json!({}));
                }
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(5));
            }
            Err(error) => {
                let _ = events.send(ServerEvent::Failure(error.to_string()));
                break;
            }
        }
    }
}

fn handle_request(
    stream: &mut TcpStream,
    state: &Arc<Mutex<ServerState>>,
    events: &mpsc::UnboundedSender<ServerEvent>,
    command: &str,
    cwd: &Path,
    attempt: &str,
) -> anyhow::Result<()> {
    let request = read_request(stream)?;
    let expected_authorization = format!("Bearer {FIXTURE_TOKEN}");
    let authorization_valid = request
        .headers
        .get("authorization")
        .is_some_and(|value| value == &expected_authorization);
    let protocol_valid = request
        .headers
        .get("x-kodkod-executor-protocol")
        .is_some_and(|value| value == "1");
    {
        let mut state = state.lock().unwrap();
        if !authorization_valid {
            state.auth_failures += 1;
        }
        if !protocol_valid {
            state.protocol_failures += 1;
        }
    }
    anyhow::ensure!(
        authorization_valid,
        "runner did not use the configured bearer token"
    );
    anyhow::ensure!(
        protocol_valid,
        "runner omitted the executor protocol header"
    );
    anyhow::ensure!(request.method == "POST", "expected POST requests");

    match request.path.as_str() {
        "/v1/device-executor/connect" => {
            let hello: DeviceHello = serde_json::from_slice(&request.body)?;
            anyhow::ensure!(hello.name == "outbound fixture", "unexpected executor name");
            anyhow::ensure!(hello.executor_id.is_some(), "executor identity was missing");
            state.lock().unwrap().connect_count += 1;
            write_json(
                stream,
                "200 OK",
                &DeviceConnection {
                    connection: FIXTURE_CONNECTION.into(),
                },
            )?;
        }
        "/v1/device-executor/poll" => {
            let poll: DevicePoll = serde_json::from_slice(&request.body)?;
            let operation = {
                let mut state = state.lock().unwrap();
                state.poll_count += 1;
                if poll.connection != FIXTURE_CONNECTION {
                    state.mismatched_connections += 1;
                }
                let can_run = poll.capabilities.contains(&DeviceCapability::Commands);
                state.saw_command_capability |= can_run;
                if can_run && poll.ready && !state.operation_delivered {
                    state.operation_delivered = true;
                    state.operation_count += 1;
                    Some(DeviceOperation {
                        attempt: attempt.into(),
                        attachment: "fixture-attachment".into(),
                        operation: DeviceAction::RunCommand {
                            command: command.into(),
                            cwd: cwd.to_string_lossy().into_owned(),
                        },
                    })
                } else {
                    None
                }
            };
            write_json(
                stream,
                "200 OK",
                &DevicePollResult {
                    operation,
                    active_authorized: true,
                },
            )?;
        }
        "/v1/device-executor/receipt" => {
            let receipt: DeviceReceipt = serde_json::from_slice(&request.body)?;
            let seen = ReceiptSeen {
                connection: receipt.connection,
                attempt: receipt.attempt,
                result: receipt.result,
            };
            let ordinal = {
                let mut state = state.lock().unwrap();
                state.receipt_count += 1;
                state.receipts.push(seen.clone());
                state.receipt_count
            };
            match ordinal {
                1 => {
                    // Simulate an accepted receipt whose HTTP acknowledgement is lost.
                    events.send(ServerEvent::Receipt {
                        ordinal,
                        receipt: seen,
                    })?;
                }
                2 => {
                    write_json(stream, "200 OK", &json!({"accepted":true}))?;
                    events.send(ServerEvent::Receipt {
                        ordinal,
                        receipt: seen,
                    })?;
                }
                _ => {
                    events.send(ServerEvent::Failure(
                        "runner submitted an unexpected extra receipt".into(),
                    ))?;
                    write_json(stream, "500 Internal Server Error", &json!({}))?;
                }
            }
        }
        path => anyhow::bail!("unexpected runner route: {path}"),
    }
    Ok(())
}

fn read_request(stream: &mut TcpStream) -> io::Result<HttpRequest> {
    let mut bytes = Vec::new();
    let mut chunk = [0u8; 8192];
    let mut header_end = None;
    let mut content_length = None;
    loop {
        let count = stream.read(&mut chunk)?;
        if count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "request ended before its body was complete",
            ));
        }
        bytes.extend_from_slice(&chunk[..count]);
        if bytes.len() > MAX_REQUEST_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "fixture request exceeded its size limit",
            ));
        }
        if header_end.is_none()
            && let Some(separator) = bytes.windows(4).position(|window| window == b"\r\n\r\n")
        {
            let headers = std::str::from_utf8(&bytes[..separator])
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
            let length = headers
                .lines()
                .skip(1)
                .filter_map(|line| line.split_once(':'))
                .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
                .map(|(_, value)| value.trim().parse::<usize>())
                .transpose()
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?
                .unwrap_or(0);
            if length > MAX_REQUEST_BYTES {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "fixture request body exceeded its size limit",
                ));
            }
            header_end = Some(separator + 4);
            content_length = Some(length);
        }
        if let (Some(header_end), Some(content_length)) = (header_end, content_length)
            && bytes.len() >= header_end + content_length
        {
            break;
        }
    }

    let header_end = header_end.expect("complete HTTP headers were parsed");
    let header_text = std::str::from_utf8(&bytes[..header_end - 4])
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let mut lines = header_text.split("\r\n");
    let mut request_line = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "request line was missing"))?
        .split_whitespace();
    let method = request_line
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "request method was missing"))?
        .to_owned();
    let path = request_line
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "request path was missing"))?
        .to_owned();
    let mut headers = HashMap::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_owned());
        }
    }
    let body_end = header_end + content_length.unwrap_or(0);
    Ok(HttpRequest {
        method,
        path,
        headers,
        body: bytes[header_end..body_end].to_vec(),
    })
}

fn write_json<T: Serialize>(stream: &mut TcpStream, status: &str, value: &T) -> io::Result<()> {
    let body = serde_json::to_vec(value)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(&body)?;
    stream.flush()
}

async fn receive_receipt(
    events: &mut mpsc::UnboundedReceiver<ServerEvent>,
) -> anyhow::Result<(usize, ReceiptSeen)> {
    match events.recv().await {
        Some(ServerEvent::Receipt { ordinal, receipt }) => Ok((ordinal, receipt)),
        Some(ServerEvent::Failure(message)) => anyhow::bail!("fixture server failed: {message}"),
        None => anyhow::bail!("fixture server stopped before delivering the receipt"),
    }
}

#[cfg(unix)]
fn fixture_command() -> &'static str {
    r#"printf 'run\n' >> execution-count.txt; printf 'outbound-executor-ok'"#
}

#[cfg(windows)]
fn fixture_command() -> &'static str {
    "echo run>>execution-count.txt & echo outbound-executor-ok"
}

#[tokio::test]
async fn outbound_runner_retries_lost_receipt_without_replaying_the_command() -> anyhow::Result<()>
{
    let fixture = tempdir()?;
    let output_file = fixture.path().join("execution-count.txt");
    let (mut server, mut events) =
        FixtureServer::start(fixture_command().into(), fixture.path().to_path_buf())?;
    let mut config = ExecutorConfig::new(
        server.base_url.clone(),
        FIXTURE_TOKEN.into(),
        fixture.path().join("executor-state"),
    );
    config.name = "outbound fixture".into();
    config.capability_mask = 1;
    config.allow_insecure_http = true;

    let (first, second) = timeout(Duration::from_secs(8), async {
        let executor = Executor::start(config);
        executor.set_enabled(true)?;

        let (first_ordinal, first) = receive_receipt(&mut events).await?;
        anyhow::ensure!(first_ordinal == 1, "first receipt ordinal was unexpected");
        anyhow::ensure!(
            first.connection == FIXTURE_CONNECTION,
            "first receipt used an unexpected logical connection"
        );
        let (second_ordinal, second) = receive_receipt(&mut events).await?;
        anyhow::ensure!(
            second_ordinal == 2,
            "receipt acknowledgement was not retried"
        );
        anyhow::ensure!(
            second.connection == first.connection && second.attempt == first.attempt,
            "receipt retry changed its connection or attempt identity"
        );
        anyhow::ensure!(
            second.result == first.result,
            "receipt retry changed the original command result"
        );
        let output = second
            .result
            .as_ref()
            .map_err(|error| anyhow::anyhow!("command failed: {error}"))?;
        anyhow::ensure!(
            output["exit_code"] == 0,
            "command exit code was not returned"
        );
        anyhow::ensure!(
            output["stdout"]
                .as_str()
                .is_some_and(|stdout| stdout.trim() == "outbound-executor-ok"),
            "command output was not returned"
        );

        executor.shutdown();
        executor.join().await;
        anyhow::ensure!(!executor.is_running(), "executor worker did not stop");
        Ok::<_, anyhow::Error>((first, second))
    })
    .await
    .expect("outbound runner fixture should finish within eight seconds")?;

    server.stop();
    let state = server.snapshot();
    anyhow::ensure!(state.connect_count == 1, "runner unexpectedly reconnected");
    anyhow::ensure!(state.operation_count == 1, "command action was replayed");
    anyhow::ensure!(
        state.receipt_count == 2,
        "receipt was not retried exactly once"
    );
    anyhow::ensure!(
        state.auth_failures == 0,
        "runner sent an invalid bearer token"
    );
    anyhow::ensure!(
        state.protocol_failures == 0,
        "runner omitted its protocol header"
    );
    anyhow::ensure!(
        state.mismatched_connections == 0,
        "poll changed logical connection"
    );
    anyhow::ensure!(
        state.saw_command_capability,
        "command capability was not advertised"
    );
    anyhow::ensure!(
        first.attempt == second.attempt,
        "receipt attempt identity changed"
    );
    anyhow::ensure!(
        std::fs::read_to_string(output_file)?.lines().count() == 1,
        "the temporary command fixture should be written exactly once"
    );
    Ok(())
}

#[cfg(unix)]
#[tokio::test]
async fn cli_executor_receives_receipt_and_gracefully_stops_on_sigterm() -> anyhow::Result<()> {
    use anyhow::Context;
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    use std::process::Stdio;

    let fixture = tempdir()?;
    let token_file = fixture.path().join("executor-token");
    let mut token_options = std::fs::OpenOptions::new();
    token_options.write(true).create_new(true).mode(0o600);
    let mut token = token_options.open(&token_file)?;
    token.write_all(FIXTURE_TOKEN.as_bytes())?;
    token.sync_all()?;
    anyhow::ensure!(
        std::fs::metadata(&token_file)?.permissions().mode() & 0o777 == 0o600,
        "CLI credential fixture must remain private"
    );

    let output_file = fixture.path().join("execution-count.txt");
    let (mut server, mut events) =
        FixtureServer::start(fixture_command().into(), fixture.path().to_path_buf())?;
    let state_dir = fixture.path().join("cli-state");
    let mut child = tokio::process::Command::new(env!("CARGO_BIN_EXE_kodkod-executor"))
        .arg("serve")
        .arg("--server")
        .arg(&server.base_url)
        .arg("--token-file")
        .arg(&token_file)
        .arg("--state-dir")
        .arg(&state_dir)
        .arg("--allow")
        .arg("commands")
        .arg("--name")
        .arg("outbound fixture")
        .arg("--allow-http")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()?;

    let (first, second) = timeout(Duration::from_secs(8), async {
        let (first_ordinal, first) = receive_receipt(&mut events).await?;
        anyhow::ensure!(
            first_ordinal == 1,
            "first CLI receipt ordinal was unexpected"
        );
        let (second_ordinal, second) = receive_receipt(&mut events).await?;
        anyhow::ensure!(
            second_ordinal == 2,
            "CLI did not retry the dropped receipt acknowledgement"
        );
        anyhow::ensure!(
            second.connection == first.connection && second.attempt == first.attempt,
            "CLI receipt retry changed its connection or attempt identity"
        );
        anyhow::ensure!(
            second.result == first.result,
            "CLI receipt retry changed the original command result"
        );
        let output = second
            .result
            .as_ref()
            .map_err(|error| anyhow::anyhow!("CLI command failed: {error}"))?;
        anyhow::ensure!(
            output["exit_code"] == 0,
            "CLI command exit code was not returned"
        );
        anyhow::ensure!(
            output["stdout"]
                .as_str()
                .is_some_and(|stdout| stdout.trim() == "outbound-executor-ok"),
            "CLI command output was not returned"
        );
        Ok::<_, anyhow::Error>((first, second))
    })
    .await
    .expect("CLI executor should receive and retry its receipt within eight seconds")?;

    let pid = child.id().context("CLI process has no pid")? as libc::pid_t;
    let signal_result = unsafe { libc::kill(pid, libc::SIGTERM) };
    anyhow::ensure!(
        signal_result == 0,
        "could not signal CLI process: {}",
        std::io::Error::last_os_error()
    );
    let status = timeout(Duration::from_secs(5), async {
        loop {
            if let Some(status) = child.try_wait()? {
                break Ok::<_, std::io::Error>(status);
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("CLI process should exit gracefully within five seconds")?;
    anyhow::ensure!(
        status.success(),
        "CLI did not exit successfully after SIGTERM"
    );

    server.stop();
    let state = server.snapshot();
    anyhow::ensure!(state.connect_count == 1, "CLI unexpectedly reconnected");
    anyhow::ensure!(state.operation_count == 1, "CLI replayed its command");
    anyhow::ensure!(
        state.receipt_count == 2,
        "CLI did not retry exactly one receipt"
    );
    anyhow::ensure!(state.auth_failures == 0, "CLI sent an invalid bearer token");
    anyhow::ensure!(
        state.protocol_failures == 0,
        "CLI omitted its protocol header"
    );
    anyhow::ensure!(
        first.attempt == second.attempt,
        "CLI receipt attempt identity changed"
    );
    anyhow::ensure!(
        std::fs::read_to_string(output_file)?.lines().count() == 1,
        "CLI command fixture should be written exactly once"
    );
    Ok(())
}

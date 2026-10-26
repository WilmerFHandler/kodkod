//! Bounded command execution for callers that already own an execution root.
//!
//! A command's working directory is checked here, but the command itself is
//! not sandboxed. Callers must authorize the request and choose its directory.

use anyhow::{Context, Result, bail};
use std::{ffi::OsString, path::PathBuf, process::Stdio, time::Duration};
use tokio::{
    io::{AsyncRead, AsyncReadExt},
    process::Command,
};

pub const DEFAULT_COMMAND_TIMEOUT_MS: u64 = 60_000;
pub const MAX_COMMAND_TIMEOUT_MS: u64 = 60_000;
pub const DEFAULT_COMMAND_OUTPUT_BYTES: usize = 64 * 1024;
pub const MAX_COMMAND_OUTPUT_BYTES: usize = 64 * 1024;
const MAX_ENV_REMOVALS: usize = 64;

/// One shell command to run in a caller-selected, existing directory.
#[derive(Clone, Debug)]
pub struct CommandRequest {
    pub command: String,
    pub cwd: PathBuf,
    /// Must be in `1..=MAX_COMMAND_TIMEOUT_MS`.
    pub timeout_ms: u64,
    /// Per-stream limit; must be in `1..=MAX_COMMAND_OUTPUT_BYTES`.
    pub max_output_bytes: usize,
    /// Environment variables the caller wants withheld from the child.
    pub env_remove: Vec<OsString>,
}

impl CommandRequest {
    pub fn shell(command: impl Into<String>, cwd: impl Into<PathBuf>) -> Self {
        Self {
            command: command.into(),
            cwd: cwd.into(),
            timeout_ms: DEFAULT_COMMAND_TIMEOUT_MS,
            max_output_bytes: DEFAULT_COMMAND_OUTPUT_BYTES,
            env_remove: Vec::new(),
        }
    }

    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    pub fn with_output_limit(mut self, max_output_bytes: usize) -> Self {
        self.max_output_bytes = max_output_bytes;
        self
    }

    pub fn without_env(mut self, name: impl Into<OsString>) -> Self {
        self.env_remove.push(name.into());
        self
    }
}

/// Captured process result. Truncation is reported independently for stdout
/// and stderr because each stream has its own byte limit.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CommandOutput {
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub output_limit_bytes: usize,
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
}

/// Run a shell command and drain both streams while retaining at most the
/// requested number of bytes from each. Dropping this future kills the whole
/// process group on Unix and the assigned Job Object on Windows.
pub async fn run(request: CommandRequest) -> Result<CommandOutput> {
    anyhow::ensure!(
        !request.command.trim().is_empty(),
        "Command cannot be empty"
    );
    anyhow::ensure!(
        (1..=MAX_COMMAND_TIMEOUT_MS).contains(&request.timeout_ms),
        "Command timeout must be between 1 and {MAX_COMMAND_TIMEOUT_MS} milliseconds"
    );
    anyhow::ensure!(
        (1..=MAX_COMMAND_OUTPUT_BYTES).contains(&request.max_output_bytes),
        "Command output limit must be between 1 and {MAX_COMMAND_OUTPUT_BYTES} bytes per stream"
    );
    anyhow::ensure!(
        request.env_remove.len() <= MAX_ENV_REMOVALS,
        "Too many environment variables to remove"
    );

    // Resolve and validate cwd locally before spawning so failures are
    // attributable to the requested directory rather than the shell.
    let cwd = tokio::fs::canonicalize(&request.cwd)
        .await
        .with_context(|| {
            format!(
                "Command working directory does not exist: {}",
                request.cwd.display()
            )
        })?;
    anyhow::ensure!(
        tokio::fs::metadata(&cwd).await?.is_dir(),
        "Command working directory is not a directory: {}",
        cwd.display()
    );

    let mut command = shell_command(&request.command);
    command
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    for name in &request.env_remove {
        command.env_remove(name);
    }
    #[cfg(unix)]
    command.process_group(0);
    #[cfg(windows)]
    command.creation_flags(windows_sys::Win32::System::Threading::CREATE_SUSPENDED);

    let mut child = command.spawn().context("Could not start command shell")?;
    #[cfg(unix)]
    let mut process_group = ProcessGroup::new(child.id());
    #[cfg(windows)]
    let mut process_group = match ProcessGroup::attach(&child) {
        Ok(group) => group,
        Err(error) => {
            // The child is still suspended, so stop it before the first await
            // in this setup-failure path. If the Job Object had been assigned,
            // its kill-on-close guard has already stopped its descendants.
            let _ = child.start_kill();
            let _ = child.wait().await;
            return Err(error.context("Could not initialize command process tree cleanup"));
        }
    };
    #[cfg(not(any(unix, windows)))]
    let mut process_group = ProcessGroup::new();
    let stdout = child.stdout.take().context("Missing command stdout")?;
    let stderr = child.stderr.take().context("Missing command stderr")?;
    let timeout = Duration::from_millis(request.timeout_ms);
    let max_output_bytes = request.max_output_bytes;

    let result = tokio::time::timeout(timeout, async {
        tokio::try_join!(
            child.wait(),
            read_limited(stdout, max_output_bytes),
            read_limited(stderr, max_output_bytes),
        )
    })
    .await;

    let (status, stdout, stderr) = match result {
        Ok(result) => result.context("Failed while collecting command output")?,
        Err(_) => {
            process_group.kill();
            // Keep direct-child termination as a fallback if the platform
            // process-tree signal did not finish it.
            let _ = child.start_kill();
            let _ = child.wait().await;
            let duration = timeout_description(request.timeout_ms);
            #[cfg(unix)]
            bail!("Command timed out after {duration}; its process group was stopped");
            #[cfg(windows)]
            bail!("Command timed out after {duration}; its process tree was stopped");
            #[cfg(not(any(unix, windows)))]
            bail!(
                "Command timed out after {duration}; child-process cleanup is not guaranteed on this platform"
            );
        }
    };

    Ok(CommandOutput {
        exit_code: status.code(),
        stdout: String::from_utf8_lossy(&stdout.bytes).into_owned(),
        stderr: String::from_utf8_lossy(&stderr.bytes).into_owned(),
        output_limit_bytes: request.max_output_bytes,
        stdout_truncated: stdout.truncated,
        stderr_truncated: stderr.truncated,
    })
}

fn shell_command(script: &str) -> Command {
    #[cfg(windows)]
    {
        let mut command = Command::new("cmd");
        command.arg("/C").arg(script);
        command
    }
    #[cfg(not(windows))]
    {
        let mut command = Command::new("/bin/sh");
        command.arg("-c").arg(script);
        command
    }
}

fn timeout_description(timeout_ms: u64) -> String {
    if timeout_ms.is_multiple_of(1_000) {
        format!("{} seconds", timeout_ms / 1_000)
    } else {
        format!("{timeout_ms} ms")
    }
}

struct LimitedOutput {
    bytes: Vec<u8>,
    truncated: bool,
}

async fn read_limited(
    reader: impl AsyncRead + Unpin,
    limit: usize,
) -> std::io::Result<LimitedOutput> {
    let mut reader = reader;
    let mut output = Vec::with_capacity(limit.min(8192));
    let mut chunk = [0u8; 8192];
    let mut truncated = false;
    loop {
        let count = reader.read(&mut chunk).await?;
        if count == 0 {
            break;
        }
        let remaining = limit.saturating_sub(output.len());
        let retained = remaining.min(count);
        output.extend_from_slice(&chunk[..retained]);
        truncated |= retained < count;
        // Continue draining after reaching the limit so a verbose process does
        // not block on a full pipe and hide its exit status.
    }
    Ok(LimitedOutput {
        bytes: output,
        truncated,
    })
}

struct ProcessGroup {
    #[cfg(unix)]
    id: Option<u32>,
    #[cfg(windows)]
    job: WindowsJobObject,
    armed: bool,
}

impl ProcessGroup {
    #[cfg(unix)]
    fn new(id: Option<u32>) -> Self {
        Self { id, armed: true }
    }

    #[cfg(windows)]
    fn attach(child: &tokio::process::Child) -> Result<Self> {
        Ok(Self {
            job: WindowsJobObject::assign_suspended(child)?,
            armed: true,
        })
    }

    #[cfg(not(any(unix, windows)))]
    fn new() -> Self {
        Self { armed: true }
    }

    fn kill(&mut self) {
        #[cfg(unix)]
        if let Some(id) = self.id {
            // Tokio assigned the child to a fresh process group before spawn.
            // Ignore ESRCH because the process may already have exited.
            unsafe {
                libc::kill(-(id as libc::pid_t), libc::SIGKILL);
            }
        }
        #[cfg(windows)]
        self.job.terminate();
        self.armed = false;
    }
}

impl Drop for ProcessGroup {
    fn drop(&mut self) {
        if self.armed {
            self.kill();
        }
    }
}

#[cfg(windows)]
struct WindowsJobObject {
    // Kernel handles are process-wide resources, so storing the handle value
    // keeps the command future Send when it moves between Tokio worker threads.
    handle: isize,
}

#[cfg(windows)]
impl WindowsJobObject {
    fn assign_suspended(child: &tokio::process::Child) -> Result<Self> {
        use std::{mem::size_of, ptr};
        use windows_sys::Win32::{
            Foundation::{FALSE, HANDLE, INVALID_HANDLE_VALUE},
            System::{
                Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD},
                JobObjects::{
                    AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
                    JOBOBJECT_BASIC_LIMIT_INFORMATION, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
                    JobObjectExtendedLimitInformation, SetInformationJobObject,
                },
                Threading::{OpenThread, ResumeThread, THREAD_SUSPEND_RESUME},
            },
        };

        let handle = unsafe { CreateJobObjectW(ptr::null(), ptr::null()) };
        anyhow::ensure!(!handle.is_null(), "Could not create a Windows Job Object");
        let job = Self {
            handle: handle as isize,
        };

        let limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
            BasicLimitInformation: JOBOBJECT_BASIC_LIMIT_INFORMATION {
                LimitFlags: JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
                ..JOBOBJECT_BASIC_LIMIT_INFORMATION::default()
            },
            ..JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default()
        };
        let configured = unsafe {
            SetInformationJobObject(
                handle,
                JobObjectExtendedLimitInformation,
                (&limits as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
                size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
        };
        anyhow::ensure!(
            configured != 0,
            "Could not configure the Windows Job Object"
        );

        let assigned = unsafe {
            AssignProcessToJobObject(
                handle,
                child
                    .raw_handle()
                    .context("Command shell has no process handle")? as HANDLE,
            )
        };
        anyhow::ensure!(
            assigned != 0,
            "Could not assign the command to its Windows Job Object"
        );

        let process_id = child.id().context("Command shell has no process id")?;
        let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) };
        anyhow::ensure!(
            !snapshot.is_null() && snapshot != INVALID_HANDLE_VALUE,
            "Could not enumerate the suspended command thread"
        );
        let snapshot = OwnedHandle(snapshot);
        let thread_id = find_primary_thread(snapshot.0, process_id)?;
        let thread = unsafe { OpenThread(THREAD_SUSPEND_RESUME, FALSE, thread_id) };
        anyhow::ensure!(
            !thread.is_null(),
            "Could not open the suspended command thread"
        );
        let thread = OwnedHandle(thread);
        let previous_suspend_count = unsafe { ResumeThread(thread.0) };
        anyhow::ensure!(
            previous_suspend_count == 1,
            "Could not resume the command's primary thread"
        );

        // `job` owns the kill-on-close handle from this point onward.
        Ok(job)
    }

    fn terminate(&self) {
        use windows_sys::Win32::System::JobObjects::TerminateJobObject;
        unsafe {
            TerminateJobObject(self.handle as _, 1);
        }
    }
}

#[cfg(windows)]
impl Drop for WindowsJobObject {
    fn drop(&mut self) {
        use windows_sys::Win32::Foundation::CloseHandle;
        unsafe {
            CloseHandle(self.handle as _);
        }
    }
}

#[cfg(windows)]
struct OwnedHandle(windows_sys::Win32::Foundation::HANDLE);

#[cfg(windows)]
impl Drop for OwnedHandle {
    fn drop(&mut self) {
        use windows_sys::Win32::Foundation::CloseHandle;
        unsafe {
            CloseHandle(self.0);
        }
    }
}

#[cfg(windows)]
fn find_primary_thread(
    snapshot: windows_sys::Win32::Foundation::HANDLE,
    process_id: u32,
) -> Result<u32> {
    use std::mem::size_of;
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        THREADENTRY32, Thread32First, Thread32Next,
    };

    let mut entry = THREADENTRY32 {
        dwSize: size_of::<THREADENTRY32>() as u32,
        ..THREADENTRY32::default()
    };
    let mut found = unsafe { Thread32First(snapshot, &mut entry) } != 0;
    while found {
        if entry.th32OwnerProcessID == process_id {
            return Ok(entry.th32ThreadID);
        }
        entry.dwSize = size_of::<THREADENTRY32>() as u32;
        found = unsafe { Thread32Next(snapshot, &mut entry) } != 0;
    }
    bail!("Could not find the command's suspended primary thread")
}

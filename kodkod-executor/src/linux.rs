//! Linux X11 desktop access through the small `scrot` and `xdotool` helpers.
//!
//! A Wayland session is not supported by this backend. The caller supplies the
//! X11 display and optional authority file so container and VM paths stay out
//! of this reusable crate.

use crate::computer::{
    ComputerInput, DesktopBackend, DesktopPermissions, DesktopScreenshot, MAX_SCREENSHOT_BYTES,
    validate_click_against_frame,
};
use anyhow::{Context, Result, bail};
use futures::future::BoxFuture;
#[cfg(target_os = "linux")]
use std::path::Path;
use std::{
    ffi::{OsStr, OsString},
    fs,
    path::PathBuf,
    process::Stdio,
    time::Duration,
};
use tokio::{
    io::{AsyncRead, AsyncReadExt},
    process::Command,
};

const DESKTOP_COMMAND_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_HELPER_OUTPUT_BYTES: usize = 64 * 1024;
const MAX_GEOMETRY_OUTPUT_BYTES: usize = 64;

/// X11 desktop backend. `display` is normally a value such as `:0`; the
/// optional authority file is passed only to the helper processes.
#[derive(Clone, Debug)]
pub struct LinuxDesktopBackend {
    display: String,
    xauthority: Option<PathBuf>,
}

impl LinuxDesktopBackend {
    pub fn new(display: impl Into<String>, xauthority: Option<PathBuf>) -> Self {
        Self {
            display: display.into(),
            xauthority,
        }
    }

    /// Read the current X11 settings without baking a guest path into the
    /// shared backend. A missing DISPLAY yields no backend to advertise.
    pub fn from_environment() -> Option<Self> {
        let display = std::env::var("DISPLAY").ok()?;
        if display.trim().is_empty() {
            return None;
        }
        Some(Self::new(
            display,
            std::env::var_os("XAUTHORITY").map(PathBuf::from),
        ))
    }

    /// Cheap capability check for registration and polling paths. It checks
    /// the current Linux target, a configured X11 display, the optional auth
    /// file, and both helper executables; actual display access is checked by
    /// each bounded helper call.
    pub fn is_available(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            !self.display.trim().is_empty()
                && std::env::var_os("WAYLAND_DISPLAY").is_none()
                && self.xauthority.as_ref().is_none_or(|path| path.is_file())
                && executable_available(OsStr::new("scrot"))
                && executable_available(OsStr::new("xdotool"))
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }

    async fn capture(&self) -> Result<DesktopScreenshot> {
        self.ensure_x11_supported()?;
        struct Capture(PathBuf);
        impl Drop for Capture {
            fn drop(&mut self) {
                let _ = fs::remove_file(&self.0);
            }
        }

        let capture = Capture(
            std::env::temp_dir().join(format!("kodkod-screen-{}.png", uuid::Uuid::now_v7())),
        );
        let path = capture.0.as_os_str().to_owned();
        self.run_helper(OsStr::new("scrot"), &[OsString::from("-o"), path], 1024)
            .await
            .context("Could not capture the X11 desktop")?
            .ensure_success("scrot")?;

        let mut bytes = Vec::new();
        tokio::fs::File::open(&capture.0)
            .await
            .context("The screenshot helper did not create an image")?
            .take((MAX_SCREENSHOT_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .await?;
        DesktopScreenshot::from_png_limited(bytes, MAX_SCREENSHOT_BYTES)
    }

    async fn send_input(&self, input: ComputerInput) -> Result<()> {
        self.ensure_x11_supported()?;
        input.validate()?;
        if matches!(input, ComputerInput::Click { .. }) {
            validate_click_against_frame(&input, self.display_dimensions().await?)?;
        }

        let args = match input {
            ComputerInput::Click { x, y } => vec![
                OsString::from("mousemove"),
                OsString::from(x.to_string()),
                OsString::from(y.to_string()),
                OsString::from("click"),
                OsString::from("1"),
            ],
            ComputerInput::Type { text } => vec![
                OsString::from("type"),
                OsString::from("--delay"),
                OsString::from("0"),
                OsString::from("--clearmodifiers"),
                OsString::from("--"),
                OsString::from(text),
            ],
            ComputerInput::Key { key } => vec![
                OsString::from("key"),
                OsString::from("--clearmodifiers"),
                OsString::from(linux_key(&key)?),
            ],
        };
        self.run_helper(OsStr::new("xdotool"), &args, MAX_HELPER_OUTPUT_BYTES)
            .await?
            .ensure_success("xdotool")?;
        Ok(())
    }

    async fn display_dimensions(&self) -> Result<(u32, u32)> {
        let output = self
            .run_helper(
                OsStr::new("xdotool"),
                &[OsString::from("getdisplaygeometry")],
                MAX_GEOMETRY_OUTPUT_BYTES,
            )
            .await?
            .ensure_success("xdotool")?;
        anyhow::ensure!(
            !output.stdout._truncated,
            "Desktop geometry output exceeds the size limit"
        );
        let stdout = String::from_utf8_lossy(&output.stdout.bytes);
        let mut values = stdout.split_whitespace();
        let width = values
            .next()
            .context("Desktop geometry output was incomplete")?
            .parse::<u32>()
            .context("Desktop geometry width was invalid")?;
        let height = values
            .next()
            .context("Desktop geometry output was incomplete")?
            .parse::<u32>()
            .context("Desktop geometry height was invalid")?;
        anyhow::ensure!(
            values.next().is_none() && width > 0 && height > 0,
            "Desktop geometry was invalid"
        );
        Ok((width, height))
    }

    fn ensure_x11_supported(&self) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            anyhow::ensure!(
                std::env::var_os("WAYLAND_DISPLAY").is_none(),
                "Wayland desktop control is unsupported; this backend requires X11"
            );
            anyhow::ensure!(
                !self.display.trim().is_empty(),
                "Linux desktop is unavailable because DISPLAY is not configured"
            );
            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        {
            bail!("Linux X11 desktop is unsupported on this platform")
        }
    }

    async fn run_helper(
        &self,
        program: &OsStr,
        args: &[OsString],
        output_limit: usize,
    ) -> Result<HelperOutput> {
        self.ensure_x11_supported()?;
        let mut command = Command::new(program);
        command
            .args(args)
            .env("DISPLAY", &self.display)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if let Some(xauthority) = &self.xauthority {
            command.env("XAUTHORITY", xauthority);
        } else {
            command.env_remove("XAUTHORITY");
        }
        #[cfg(unix)]
        command.process_group(0);

        let mut child = command
            .spawn()
            .with_context(|| format!("Could not start {}", program.to_string_lossy()))?;
        let mut process_group = HelperProcessGroup::new(child.id());
        let stdout = child
            .stdout
            .take()
            .context("Missing desktop helper stdout")?;
        let stderr = child
            .stderr
            .take()
            .context("Missing desktop helper stderr")?;
        let result = tokio::time::timeout(DESKTOP_COMMAND_TIMEOUT, async {
            tokio::try_join!(
                child.wait(),
                read_limited(stdout, output_limit),
                read_limited(stderr, output_limit),
            )
        })
        .await;

        let (status, stdout, stderr) = match result {
            Ok(result) => result.context("Failed while collecting desktop helper output")?,
            Err(_) => {
                process_group.kill();
                let _ = child.start_kill();
                let _ = child.wait().await;
                #[cfg(unix)]
                bail!("Desktop command timed out; its process group was stopped");
                #[cfg(not(unix))]
                bail!(
                    "Desktop command timed out; child-process cleanup is not guaranteed on this platform"
                );
            }
        };
        process_group.disarm();
        Ok(HelperOutput {
            status,
            stdout,
            stderr,
        })
    }
}

impl DesktopBackend for LinuxDesktopBackend {
    fn permissions(&self) -> DesktopPermissions {
        DesktopPermissions::not_required()
    }

    fn screenshot<'a>(&'a self) -> BoxFuture<'a, Result<DesktopScreenshot>> {
        Box::pin(self.capture())
    }

    fn input<'a>(&'a self, input: ComputerInput) -> BoxFuture<'a, Result<()>> {
        Box::pin(self.send_input(input))
    }
}

struct HelperOutput {
    status: std::process::ExitStatus,
    stdout: LimitedOutput,
    stderr: LimitedOutput,
}

impl HelperOutput {
    fn ensure_success(self, program: &str) -> Result<Self> {
        if self.status.success() {
            return Ok(self);
        }
        let stderr = String::from_utf8_lossy(&self.stderr.bytes);
        let code = self
            .status
            .code()
            .map(|code| code.to_string())
            .unwrap_or_else(|| "signal".to_owned());
        if stderr.is_empty() {
            bail!("{program} exited with {code}; the X11 desktop may be unavailable");
        }
        bail!("{program} exited with {code}: {stderr}");
    }
}

struct LimitedOutput {
    bytes: Vec<u8>,
    _truncated: bool,
}

async fn read_limited(
    reader: impl AsyncRead + Unpin,
    limit: usize,
) -> std::io::Result<LimitedOutput> {
    let mut reader = reader;
    let mut bytes = Vec::with_capacity(limit.min(8192));
    let mut chunk = [0u8; 8192];
    let mut truncated = false;
    loop {
        let count = reader.read(&mut chunk).await?;
        if count == 0 {
            break;
        }
        let remaining = limit.saturating_sub(bytes.len());
        let retained = remaining.min(count);
        bytes.extend_from_slice(&chunk[..retained]);
        truncated |= retained < count;
    }
    Ok(LimitedOutput {
        bytes,
        _truncated: truncated,
    })
}

fn linux_key(key: &str) -> Result<&str> {
    anyhow::ensure!(
        !key.starts_with("cmd+"),
        "Command shortcuts are available in macOS sessions; use ctrl+ shortcuts in Linux sessions"
    );
    Ok(key)
}

#[cfg(target_os = "linux")]
fn executable_available(program: &OsStr) -> bool {
    let path = Path::new(program);
    if path.components().count() > 1 {
        return executable_file(path);
    }
    let Some(search_path) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&search_path).any(|directory| executable_file(&directory.join(path)))
}

#[cfg(target_os = "linux")]
fn executable_file(path: &Path) -> bool {
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

struct HelperProcessGroup {
    id: Option<u32>,
    armed: bool,
}

impl HelperProcessGroup {
    fn new(id: Option<u32>) -> Self {
        Self { id, armed: true }
    }

    fn kill(&mut self) {
        #[cfg(unix)]
        if let Some(id) = self.id {
            unsafe {
                libc::kill(-(id as libc::pid_t), libc::SIGKILL);
            }
        }
        self.armed = false;
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for HelperProcessGroup {
    fn drop(&mut self) {
        if self.armed {
            self.kill();
        }
    }
}

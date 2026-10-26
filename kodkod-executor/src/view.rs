//! Read-only media, separate from action dispatch, receipts and observations.
#[cfg(target_os = "linux")]
use anyhow::Context;
use anyhow::{Result, ensure};
use image::{ExtendedColorType, ImageEncoder};
use serde::{Deserialize, Serialize};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering},
    },
    time::Duration,
};

pub const MAX_VIEW_FRAME_BYTES: usize = 1024 * 1024;
pub const VIEW_WIDTH: u32 = 1600;
pub const VIEW_INTERVAL: Duration = Duration::from_millis(100);
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ViewPoll {
    pub connection: String,
    pub grant_generation: u64,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ViewPollResult {
    pub stream: Option<String>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ViewFrame {
    pub codec: String,
    pub width: u32,
    pub height: u32,
    pub sequence: u64,
    pub data: Vec<u8>,
}
impl ViewFrame {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.codec == "jpeg"
                && self.sequence > 0
                && !self.data.is_empty()
                && self.data.len() <= MAX_VIEW_FRAME_BYTES,
            "Invalid desktop media frame"
        );
        let dimensions = image::ImageReader::with_format(
            std::io::Cursor::new(&self.data),
            image::ImageFormat::Jpeg,
        )
        .into_dimensions()?;
        ensure!(
            dimensions == (self.width, self.height)
                && self.width > 0
                && self.width <= VIEW_WIDTH
                && self.height > 0
                && u64::from(self.width) * u64::from(self.height)
                    <= crate::computer::MAX_SCREENSHOT_PIXELS,
            "Invalid desktop media geometry"
        );
        Ok(())
    }
    pub fn encode(image: image::DynamicImage, sequence: u64) -> Result<Self> {
        let image = if image.width() > VIEW_WIDTH {
            image.resize(VIEW_WIDTH, u32::MAX, image::imageops::FilterType::Triangle)
        } else {
            image
        };
        let rgb = image.into_rgb8();
        let mut data = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut data, 75).write_image(
            rgb.as_raw(),
            rgb.width(),
            rgb.height(),
            ExtendedColorType::Rgb8,
        )?;
        let frame = Self {
            codec: "jpeg".into(),
            width: rgb.width(),
            height: rgb.height(),
            sequence,
            data,
        };
        frame.validate()?;
        Ok(frame)
    }
}

/// Aborting this owner also drops its in-flight capture and media request.
pub(crate) struct ViewWorker(tokio::task::JoinHandle<()>);
impl Drop for ViewWorker {
    fn drop(&mut self) {
        self.0.abort();
    }
}
impl ViewWorker {
    pub fn is_finished(&self) -> bool {
        self.0.is_finished()
    }
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        client: reqwest::Client,
        base: String,
        token: String,
        connection: String,
        epoch: u64,
        allowed: Arc<AtomicU8>,
        generation: Arc<AtomicU64>,
        alive: Arc<AtomicBool>,
        mask: u8,
        helper_socket: Option<PathBuf>,
    ) -> Self {
        Self(tokio::spawn(async move {
            let authorized = || {
                alive.load(Ordering::Acquire)
                    && generation.load(Ordering::Acquire) == epoch
                    && allowed.load(Ordering::Acquire) & mask & 4 != 0
            };
            #[cfg(target_os = "linux")]
            let helper = crate::platform::resolve_desktop_helper_socket(helper_socket.as_deref())
                .map(crate::desktop_helper::DesktopHelperClient::new);
            #[cfg(not(target_os = "linux"))]
            let _ = helper_socket;
            let mut sequence = 0u64;
            let mut active_stream = None;
            let mut ticks = tokio::time::interval(VIEW_INTERVAL);
            ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            while authorized() {
                if active_stream.is_none() {
                    active_stream = poll_stream(&client, &base, &token, &connection, epoch)
                        .await
                        .ok()
                        .flatten();
                    if active_stream.is_none() {
                        tokio::time::sleep(Duration::from_millis(500)).await;
                        continue;
                    }
                }
                ticks.tick().await;
                if !authorized() {
                    break;
                }
                sequence = match sequence.checked_add(1) {
                    Some(next) => next,
                    None => break,
                };
                #[cfg(target_os = "linux")]
                let frame = if let Some(helper) = &helper {
                    helper.view_frame().await
                } else {
                    capture(sequence).await
                };
                #[cfg(not(target_os = "linux"))]
                let frame = capture(sequence).await;
                if let Ok(frame) = frame
                    && authorized()
                {
                    let result = client
                        .post(format!(
                            "{}/v1/device-executor/view/frame",
                            base.trim_end_matches('/')
                        ))
                        .bearer_auth(&token)
                        .header("content-type", "image/jpeg")
                        .header("x-executor-connection", &connection)
                        .header("x-grant-generation", epoch)
                        .header("x-view-stream", active_stream.as_ref().unwrap())
                        .header("x-frame-sequence", sequence)
                        .header("x-frame-width", frame.width)
                        .header("x-frame-height", frame.height)
                        .body(frame.data)
                        .send()
                        .await;
                    if !result.is_ok_and(|response| response.status().is_success()) {
                        active_stream = None;
                    }
                } else {
                    active_stream = None;
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
                // One in-flight upload and no capture queue. Missed frames are
                // dropped rather than accumulating delay behind a slow link.
            }
        }))
    }
}
async fn poll_stream(
    client: &reqwest::Client,
    base: &str,
    token: &str,
    connection: &str,
    generation: u64,
) -> Result<Option<String>> {
    let mut response = client
        .post(format!(
            "{}/v1/device-executor/view/poll",
            base.trim_end_matches('/')
        ))
        .bearer_auth(token)
        .json(&ViewPoll {
            connection: connection.into(),
            grant_generation: generation,
        })
        .send()
        .await?
        .error_for_status()?;
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        ensure!(
            bytes.len() + chunk.len() <= 4096,
            "Invalid viewing directive"
        );
        bytes.extend_from_slice(&chunk);
    }
    let directive: ViewPollResult = serde_json::from_slice(&bytes)?;
    if let Some(stream) = &directive.stream {
        uuid::Uuid::parse_str(stream)?;
    }
    Ok(directive.stream)
}

async fn capture(sequence: u64) -> Result<ViewFrame> {
    #[cfg(any(target_os = "linux", windows))]
    use crate::computer::DesktopBackend;
    #[cfg(target_os = "linux")]
    let frame = crate::linux::LinuxDesktopBackend::from_environment()
        .context("No X11 desktop")?
        .screenshot()
        .await?;
    #[cfg(target_os = "macos")]
    let frame = tokio::task::spawn_blocking(crate::macos::read_only_screenshot).await??;
    #[cfg(windows)]
    let frame = crate::windows::WindowsDesktopBackend.screenshot().await?;
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    anyhow::bail!("Desktop viewing is unsupported on this platform");
    #[cfg(any(target_os = "linux", target_os = "macos", windows))]
    {
        let data = frame.png().to_vec();
        tokio::task::spawn_blocking(move || {
            ViewFrame::encode(
                image::load_from_memory_with_format(&data, image::ImageFormat::Png)?,
                sequence,
            )
        })
        .await?
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn media_is_bounded_compressed_and_geometry_checked() {
        let frame = ViewFrame::encode(image::DynamicImage::new_rgb8(2000, 1000), 1).unwrap();
        assert_eq!((frame.width, frame.height), (1600, 800));
        assert!(frame.data.len() < 100_000);
        let mut bad = frame.clone();
        bad.width += 1;
        assert!(bad.validate().is_err());
        bad = frame;
        bad.codec = "png".into();
        assert!(bad.validate().is_err());
    }
}

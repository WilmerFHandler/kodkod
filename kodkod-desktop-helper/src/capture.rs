//! Portal-scoped PipeWire capture. All PipeWire objects live on one worker
//! thread because the PipeWire Rust bindings are thread-affine.

use anyhow::{Context, Result, bail};
use image::{ExtendedColorType, ImageEncoder, codecs::png::PngEncoder};
use kodkod_executor::computer::{DesktopScreenshot, MAX_SCREENSHOT_BYTES, MAX_SCREENSHOT_PIXELS};
use pipewire as pw;
use pw::{properties::properties, spa};
use std::{
    os::fd::OwnedFd,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const CAPTURE_START_TIMEOUT: Duration = Duration::from_secs(10);
const WORKER_POLL_INTERVAL: Duration = Duration::from_millis(100);
const MAX_CAPTURE_RATE: Duration = Duration::from_millis(67);
const MAX_CAPTURE_BYTES: usize = (MAX_SCREENSHOT_PIXELS as usize) * 4;

#[derive(Clone)]
pub struct FrameCapture {
    inner: Arc<CaptureInner>,
}

struct CaptureInner {
    shared: Arc<(Mutex<CaptureState>, Condvar)>,
    stopping: Arc<AtomicBool>,
    ready: Arc<AtomicBool>,
    geometry_changed: Arc<AtomicBool>,
    worker: Mutex<Option<JoinHandle<()>>>,
    jpeg: Mutex<Option<kodkod_executor::view::ViewFrame>>,
}

#[derive(Default)]
struct CaptureState {
    latest: Option<Arc<RawFrame>>,
    failure: Option<String>,
    sequence: u64,
    baseline_dimensions: Option<(u32, u32)>,
}

struct RawFrame {
    width: u32,
    height: u32,
    stride: usize,
    format: spa::param::video::VideoFormat,
    bytes: Vec<u8>,
    sequence: u64,
}

struct StreamData {
    format: spa::param::video::VideoInfoRaw,
    last_frame_copy: Option<Instant>,
}

impl FrameCapture {
    /// Connects only to the file descriptor returned by the ScreenCast portal
    /// and only to its selected node. This never consults the session's normal
    /// PipeWire server or guesses a display/socket path.
    pub fn start(remote_fd: OwnedFd, node_id: u32, pipewire_serial: Option<u64>) -> Result<Self> {
        let shared = Arc::new((Mutex::new(CaptureState::default()), Condvar::new()));
        let stopping = Arc::new(AtomicBool::new(false));
        let ready = Arc::new(AtomicBool::new(false));
        let geometry_changed = Arc::new(AtomicBool::new(false));
        let worker_shared = shared.clone();
        let worker_stopping = stopping.clone();
        let worker_ready = ready.clone();
        let worker_geometry_changed = geometry_changed.clone();
        let worker = thread::Builder::new()
            .name("kodkod-portal-capture".to_owned())
            .spawn(move || {
                let result = run_pipewire(
                    remote_fd,
                    node_id,
                    pipewire_serial,
                    worker_shared.clone(),
                    worker_stopping.clone(),
                    worker_ready.clone(),
                    worker_geometry_changed.clone(),
                );
                if let Err(error) = result {
                    fail(&worker_shared, &worker_ready, error.to_string());
                } else if !worker_stopping.load(Ordering::Acquire) {
                    fail(
                        &worker_shared,
                        &worker_ready,
                        "PipeWire capture worker stopped unexpectedly".to_owned(),
                    );
                }
            })
            .context("Could not start PipeWire capture worker")?;
        let capture = Self {
            inner: Arc::new(CaptureInner {
                shared,
                stopping,
                ready,
                geometry_changed,
                worker: Mutex::new(Some(worker)),
                jpeg: Mutex::new(None),
            }),
        };

        if let Err(error) = capture.wait_for_first_frame(CAPTURE_START_TIMEOUT) {
            capture.stop();
            return Err(error).context("Portal PipeWire stream did not produce a usable frame");
        }
        Ok(capture)
    }

    pub fn is_ready(&self) -> bool {
        self.inner.ready.load(Ordering::Acquire)
    }

    pub fn ready_flag(&self) -> Arc<AtomicBool> {
        self.inner.ready.clone()
    }

    pub fn geometry_changed_flag(&self) -> Arc<AtomicBool> {
        self.inner.geometry_changed.clone()
    }

    pub fn dimensions(&self) -> Result<(u32, u32)> {
        let frame = self.latest_frame()?;
        Ok((frame.width, frame.height))
    }

    pub fn frame_sequence(&self) -> Result<u64> {
        Ok(self.latest_frame()?.sequence)
    }

    /// Waits until PipeWire has delivered a frame newer than `sequence`.
    /// Callers use this after input so the next observation cannot be the
    /// cached pre-action frame.
    pub fn wait_for_newer_frame(
        &self,
        sequence: u64,
        timeout: Duration,
        authorized: Arc<AtomicBool>,
    ) -> Result<bool> {
        let deadline = Instant::now() + timeout;
        let (state, wake) = &*self.inner.shared;
        let mut state = state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        loop {
            if state
                .latest
                .as_ref()
                .is_some_and(|frame| frame.sequence > sequence)
            {
                return Ok(true);
            }
            if let Some(error) = &state.failure {
                bail!("Portal PipeWire capture failed: {error}");
            }
            if !self.is_ready() || self.inner.stopping.load(Ordering::Acquire) {
                bail!("Portal PipeWire capture stopped while waiting for a fresh frame");
            }
            if !authorized.load(Ordering::Acquire) {
                bail!(
                    "Input may have completed before the IPC client disconnected; do not replay it"
                );
            }
            let now = Instant::now();
            if now >= deadline {
                return Ok(false);
            }
            let (next, wait) = wake
                .wait_timeout(
                    state,
                    deadline
                        .saturating_duration_since(now)
                        .min(Duration::from_millis(100)),
                )
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state = next;
            if wait.timed_out() && Instant::now() >= deadline {
                return Ok(false);
            }
        }
    }

    /// Encodes the last bounded raw frame on demand. The PipeWire callback
    /// keeps one raw frame only and does no PNG work.
    pub fn screenshot(&self) -> Result<DesktopScreenshot> {
        let frame = self.latest_frame()?;
        let rgba = frame_to_rgba(&frame)?;
        let mut png = Vec::new();
        PngEncoder::new(&mut png)
            .write_image(&rgba, frame.width, frame.height, ExtendedColorType::Rgba8)
            .context("Could not encode portal frame as PNG")?;
        DesktopScreenshot::from_png_limited(png, MAX_SCREENSHOT_BYTES)
    }

    /// One compressed frame cache shared by every viewer; no input observation.
    pub fn view_frame(&self) -> Result<kodkod_executor::view::ViewFrame> {
        let frame = self.latest_frame()?;
        let mut cache = self
            .inner
            .jpeg
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(cached) = cache
            .as_ref()
            .filter(|cached| cached.sequence == frame.sequence)
        {
            return Ok(cached.clone());
        }
        let rgba = image::RgbaImage::from_raw(frame.width, frame.height, frame_to_rgba(&frame)?)
            .context("Invalid portal frame geometry")?;
        let encoded = kodkod_executor::view::ViewFrame::encode(
            image::DynamicImage::ImageRgba8(rgba),
            frame.sequence,
        )?;
        *cache = Some(encoded.clone());
        Ok(encoded)
    }

    pub fn stop(&self) {
        self.inner.stopping.store(true, Ordering::Release);
        self.inner.ready.store(false, Ordering::Release);
        let (_, wake) = &*self.inner.shared;
        wake.notify_all();
    }

    fn wait_for_first_frame(&self, timeout: Duration) -> Result<()> {
        let deadline = Instant::now() + timeout;
        let (state, wake) = &*self.inner.shared;
        let mut state = state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        loop {
            if state.latest.is_some() {
                return Ok(());
            }
            if let Some(error) = &state.failure {
                bail!("{error}");
            }
            if self.inner.stopping.load(Ordering::Acquire) {
                bail!("PipeWire capture was stopped");
            }
            let now = Instant::now();
            if now >= deadline {
                bail!("Timed out waiting for the first PipeWire frame");
            }
            let (next, wait) = wake
                .wait_timeout(state, deadline.saturating_duration_since(now))
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state = next;
            if wait.timed_out() && state.latest.is_none() {
                bail!("Timed out waiting for the first PipeWire frame");
            }
        }
    }

    fn latest_frame(&self) -> Result<Arc<RawFrame>> {
        let (state, _) = &*self.inner.shared;
        let state = state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(error) = &state.failure {
            bail!("Portal PipeWire capture failed: {error}");
        }
        state
            .latest
            .clone()
            .context("Portal PipeWire stream has not produced a frame")
    }
}

impl Drop for CaptureInner {
    fn drop(&mut self) {
        self.stopping.store(true, Ordering::Release);
        self.ready.store(false, Ordering::Release);
        self.shared.1.notify_all();
        if let Some(worker) = self
            .worker
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
        {
            let _ = worker.join();
        }
    }
}

fn run_pipewire(
    remote_fd: OwnedFd,
    node_id: u32,
    pipewire_serial: Option<u64>,
    shared: Arc<(Mutex<CaptureState>, Condvar)>,
    stopping: Arc<AtomicBool>,
    ready: Arc<AtomicBool>,
    geometry_changed: Arc<AtomicBool>,
) -> Result<()> {
    pw::init();
    let _pipewire_lifetime = PipeWireLifetime;
    let mainloop = pw::main_loop::MainLoop::new(None).context("Could not create PipeWire loop")?;
    let context =
        pw::context::Context::new(&mainloop).context("Could not create PipeWire context")?;
    let core = context
        .connect_fd(remote_fd, None)
        .context("Could not connect to the portal PipeWire remote")?;
    // Even on older portals without serial metadata, removing the selected
    // node ends this grant. Never reconnect to a recycled numeric node ID.
    let registry = core
        .get_registry()
        .context("Could not inspect the granted PipeWire remote")?;
    let removal_shared = shared.clone();
    let removal_ready = ready.clone();
    let removal_stopping = stopping.clone();
    let identity_verified = Arc::new(AtomicBool::new(pipewire_serial.is_none()));
    let identity_global = identity_verified.clone();
    let identity_shared = shared.clone();
    let identity_ready = ready.clone();
    let identity_stopping = stopping.clone();
    let _registry_listener = registry
        .add_listener_local()
        .global(move |object| {
            if object.id != node_id {
                return;
            }
            if let Some(expected) = pipewire_serial {
                let actual = object
                    .props
                    .and_then(|properties| properties.get("object.serial"))
                    .and_then(|value| value.parse::<u64>().ok());
                if actual == Some(expected) {
                    identity_global.store(true, Ordering::Release);
                } else {
                    fail(
                        &identity_shared,
                        &identity_ready,
                        "Portal-selected capture identity changed; explicitly restart sharing"
                            .to_owned(),
                    );
                    identity_stopping.store(true, Ordering::Release);
                }
            }
        })
        .global_remove(move |removed| {
            if removed == node_id {
                fail(
                    &removal_shared,
                    &removal_ready,
                    "Portal-selected capture source was removed; explicitly restart sharing"
                        .to_owned(),
                );
                removal_stopping.store(true, Ordering::Release);
            }
        })
        .register();
    let stream_data = StreamData {
        format: spa::param::video::VideoInfoRaw::default(),
        last_frame_copy: None,
    };
    let mut properties = properties! {
        *pw::keys::MEDIA_TYPE => "Video",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE => "Screen",
        *pw::keys::NODE_DONT_RECONNECT => "true",
    };
    if let Some(serial) = pipewire_serial {
        properties.insert(*pw::keys::TARGET_OBJECT, serial.to_string());
    }
    let stream = pw::stream::Stream::new(&core, "kodkod-portal-capture", properties)
        .context("Could not create PipeWire video stream")?;

    let shared_state = shared.clone();
    let stop_on_error = stopping.clone();
    let ready_on_error = ready.clone();
    let shared_param = shared.clone();
    let ready_param = ready.clone();
    let stop_param = stopping.clone();
    let shared_process = shared.clone();
    let ready_process = ready.clone();
    let stop_process = stopping.clone();
    let geometry_process = geometry_changed.clone();
    let listener = stream
        .add_local_listener_with_user_data(stream_data)
        .state_changed(move |_, _, _, new| match new {
            pw::stream::StreamState::Error(error) => {
                fail(&shared_state, &ready_on_error, error);
                stop_on_error.store(true, Ordering::Release);
            }
            pw::stream::StreamState::Unconnected => {
                fail(
                    &shared_state,
                    &ready_on_error,
                    "PipeWire stream disconnected".to_owned(),
                );
                stop_on_error.store(true, Ordering::Release);
            }
            _ => {}
        })
        .param_changed(move |_, user_data, id, param| {
            if id != spa::param::ParamType::Format.as_raw() {
                return;
            }
            let Some(param) = param else {
                return;
            };
            let Ok((media_type, media_subtype)) = spa::param::format_utils::parse_format(param)
            else {
                return;
            };
            if media_type != spa::param::format::MediaType::Video
                || media_subtype != spa::param::format::MediaSubtype::Raw
            {
                return;
            }
            if let Err(error) = user_data.format.parse(param) {
                fail(
                    &shared_param,
                    &ready_param,
                    format!("Could not parse negotiated PipeWire format: {error}"),
                );
                stop_param.store(true, Ordering::Release);
            }
        })
        .process(move |stream, user_data| {
            let Some(mut buffer) = stream.dequeue_buffer() else {
                return;
            };
            if stop_process.load(Ordering::Acquire) || !identity_verified.load(Ordering::Acquire) {
                return;
            }
            let now = Instant::now();
            if user_data
                .last_frame_copy
                .is_some_and(|last| now.duration_since(last) < MAX_CAPTURE_RATE)
            {
                return;
            }
            let data = buffer.datas_mut();
            if data.is_empty() {
                return;
            }
            let offset = data[0].chunk().offset() as usize;
            let size = data[0].chunk().size() as usize;
            let stride = data[0].chunk().stride();
            let Some(bytes) = data[0].data() else {
                return;
            };
            let frame_size = user_data.format.size();
            let result = copy_frame(
                frame_size.width,
                frame_size.height,
                user_data.format.format(),
                stride,
                bytes,
                offset,
                size,
            );
            match result {
                Ok(frame) => {
                    user_data.last_frame_copy = Some(now);
                    let (state, wake) = &*shared_process;
                    let mut state = state
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    let dimensions = (frame.width, frame.height);
                    if state
                        .baseline_dimensions
                        .is_some_and(|baseline| baseline != dimensions)
                    {
                        drop(state);
                        geometry_process.store(true, Ordering::Release);
                        fail(
                            &shared_process,
                            &ready_process,
                            "Display geometry changed after portal metadata was selected; explicitly restart sharing".to_owned(),
                        );
                        stop_process.store(true, Ordering::Release);
                        return;
                    }
                    state.baseline_dimensions.get_or_insert(dimensions);
                    let Some(sequence) = state.sequence.checked_add(1) else {
                        drop(state);
                        fail(
                            &shared_process,
                            &ready_process,
                            "PipeWire frame sequence overflowed".to_owned(),
                        );
                        return;
                    };
                    state.sequence = sequence;
                    let frame = RawFrame { sequence, ..frame };
                    state.latest = Some(Arc::new(frame));
                    ready_process.store(true, Ordering::Release);
                    wake.notify_all();
                }
                Err(error) => {
                    fail(&shared_process, &ready_process, error.to_string());
                    stop_process.store(true, Ordering::Release);
                }
            }
        })
        .register()
        .context("Could not register PipeWire stream callbacks")?;

    let pod = pw::spa::pod::object!(
        pw::spa::utils::SpaTypes::ObjectParamFormat,
        pw::spa::param::ParamType::EnumFormat,
        pw::spa::pod::property!(
            pw::spa::param::format::FormatProperties::MediaType,
            Id,
            pw::spa::param::format::MediaType::Video
        ),
        pw::spa::pod::property!(
            pw::spa::param::format::FormatProperties::MediaSubtype,
            Id,
            pw::spa::param::format::MediaSubtype::Raw
        ),
        pw::spa::pod::property!(
            pw::spa::param::format::FormatProperties::VideoFormat,
            Choice,
            Enum,
            Id,
            pw::spa::param::video::VideoFormat::RGBA,
            pw::spa::param::video::VideoFormat::RGBA,
            pw::spa::param::video::VideoFormat::BGRA,
            pw::spa::param::video::VideoFormat::RGBx,
            pw::spa::param::video::VideoFormat::BGRx
        ),
        pw::spa::pod::property!(
            pw::spa::param::format::FormatProperties::VideoSize,
            Choice,
            Range,
            Rectangle,
            pw::spa::utils::Rectangle {
                width: 1920,
                height: 1080
            },
            pw::spa::utils::Rectangle {
                width: 1,
                height: 1
            },
            pw::spa::utils::Rectangle {
                width: 8192,
                height: 8192
            }
        ),
    );
    let values: Vec<u8> = pw::spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &pw::spa::pod::Value::Object(pod),
    )
    .context("Could not serialize PipeWire format request")?
    .0
    .into_inner();
    let mut params = [spa::pod::Pod::from_bytes(&values).context("Invalid PipeWire format pod")?];
    stream
        .connect(
            spa::utils::Direction::Input,
            if pipewire_serial.is_some() {
                None
            } else {
                Some(node_id)
            },
            pw::stream::StreamFlags::AUTOCONNECT
                | pw::stream::StreamFlags::MAP_BUFFERS
                | pw::stream::StreamFlags::DONT_RECONNECT,
            &mut params,
        )
        .context("Could not connect to the portal-selected PipeWire stream")?;

    let weak_loop = mainloop.downgrade();
    let stop_timer = mainloop.loop_().add_timer(move |_| {
        if stopping.load(Ordering::Acquire)
            && let Some(mainloop) = weak_loop.upgrade()
        {
            mainloop.quit();
        }
    });
    stop_timer
        .update_timer(Some(WORKER_POLL_INTERVAL), Some(WORKER_POLL_INTERVAL))
        .into_result()
        .context("Could not schedule PipeWire worker shutdown checks")?;

    mainloop.run();
    drop(stop_timer);
    drop(listener);
    drop(stream);
    drop(_registry_listener);
    drop(registry);
    drop(core);
    drop(context);
    drop(mainloop);
    Ok(())
}

struct PipeWireLifetime;

impl Drop for PipeWireLifetime {
    fn drop(&mut self) {
        // This dedicated helper owns the sole PipeWire context, already dropped.
        unsafe { pw::deinit() };
    }
}

fn fail(shared: &(Mutex<CaptureState>, Condvar), ready: &AtomicBool, message: String) {
    ready.store(false, Ordering::Release);
    let (state, wake) = shared;
    let mut state = state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if state.failure.is_none() {
        state.failure = Some(message);
    }
    wake.notify_all();
}

fn copy_frame(
    width: u32,
    height: u32,
    format: spa::param::video::VideoFormat,
    stride: i32,
    source: &[u8],
    offset: usize,
    chunk_size: usize,
) -> Result<RawFrame> {
    anyhow::ensure!(
        width > 0 && height > 0,
        "PipeWire video dimensions are empty"
    );
    anyhow::ensure!(
        u64::from(width) * u64::from(height) <= MAX_SCREENSHOT_PIXELS,
        "PipeWire video dimensions exceed the desktop pixel limit"
    );
    anyhow::ensure!(
        matches!(
            format,
            spa::param::video::VideoFormat::RGBA
                | spa::param::video::VideoFormat::BGRA
                | spa::param::video::VideoFormat::RGBx
                | spa::param::video::VideoFormat::BGRx
        ),
        "PipeWire negotiated an unsupported pixel format"
    );
    let row_bytes = usize::try_from(width)
        .context("PipeWire video width is too large")?
        .checked_mul(4)
        .context("PipeWire video row size overflowed")?;
    let stride = usize::try_from(stride).context("PipeWire video stride is invalid")?;
    anyhow::ensure!(stride >= row_bytes, "PipeWire video stride is too short");
    let required_size = stride
        .checked_mul(height as usize - 1)
        .and_then(|prior_rows| prior_rows.checked_add(row_bytes))
        .context("PipeWire video frame size overflowed")?;
    anyhow::ensure!(
        required_size <= MAX_CAPTURE_BYTES,
        "PipeWire frame exceeds the raw capture limit"
    );
    anyhow::ensure!(
        chunk_size >= required_size,
        "PipeWire frame chunk does not contain all rows"
    );
    anyhow::ensure!(!source.is_empty(), "PipeWire mapped buffer is empty");
    anyhow::ensure!(
        required_size <= source.len(),
        "PipeWire mapped buffer is smaller than the video frame"
    );
    let offset = offset % source.len();
    let mut frame_bytes = Vec::with_capacity(required_size);
    let first_part = required_size.min(source.len() - offset);
    frame_bytes.extend_from_slice(&source[offset..offset + first_part]);
    if first_part < required_size {
        frame_bytes.extend_from_slice(&source[..required_size - first_part]);
    }
    Ok(RawFrame {
        width,
        height,
        stride,
        format,
        bytes: frame_bytes,
        sequence: 0,
    })
}

fn frame_to_rgba(frame: &RawFrame) -> Result<Vec<u8>> {
    let row_bytes = usize::try_from(frame.width)
        .context("PipeWire video width is too large")?
        .checked_mul(4)
        .context("PipeWire video row size overflowed")?;
    let output_size = row_bytes
        .checked_mul(frame.height as usize)
        .context("PipeWire output size overflowed")?;
    anyhow::ensure!(
        output_size <= MAX_CAPTURE_BYTES,
        "PipeWire frame exceeds the raw capture limit"
    );
    let required_size = frame
        .stride
        .checked_mul(frame.height as usize - 1)
        .and_then(|prior_rows| prior_rows.checked_add(row_bytes))
        .context("Stored PipeWire frame size overflowed")?;
    anyhow::ensure!(
        frame.bytes.len() >= required_size,
        "Stored PipeWire frame is incomplete"
    );
    let mut rgba = vec![0; output_size];
    for y in 0..frame.height as usize {
        let input_row = &frame.bytes[y * frame.stride..y * frame.stride + row_bytes];
        let output_row = &mut rgba[y * row_bytes..(y + 1) * row_bytes];
        for (source, destination) in input_row
            .as_chunks::<4>()
            .0
            .iter()
            .zip(output_row.as_chunks_mut::<4>().0.iter_mut())
        {
            match frame.format {
                spa::param::video::VideoFormat::RGBA => destination.copy_from_slice(source),
                spa::param::video::VideoFormat::BGRA => {
                    destination.copy_from_slice(&[source[2], source[1], source[0], source[3]])
                }
                spa::param::video::VideoFormat::RGBx => {
                    destination.copy_from_slice(&[source[0], source[1], source[2], 255])
                }
                spa::param::video::VideoFormat::BGRx => {
                    destination.copy_from_slice(&[source[2], source[1], source[0], 255])
                }
                _ => bail!("PipeWire negotiated an unsupported pixel format"),
            }
        }
    }
    Ok(rgba)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_only_a_bounded_frame_with_valid_stride() {
        let source = [1, 2, 3, 4, 5, 6, 7, 8, 90, 91, 92, 93];
        let frame = copy_frame(
            1,
            2,
            spa::param::video::VideoFormat::RGBA,
            8,
            &source,
            0,
            source.len(),
        )
        .unwrap();
        assert_eq!(frame.bytes, source);
        assert_eq!(frame_to_rgba(&frame).unwrap(), [1, 2, 3, 4, 90, 91, 92, 93]);
        assert!(
            copy_frame(
                2,
                1,
                spa::param::video::VideoFormat::RGBA,
                4,
                &source,
                0,
                source.len(),
            )
            .is_err()
        );
    }

    #[test]
    fn converts_bgra_and_x_channel_formats_to_rgba() {
        let bgra = copy_frame(
            1,
            1,
            spa::param::video::VideoFormat::BGRA,
            4,
            &[3, 2, 1, 4],
            0,
            4,
        )
        .unwrap();
        assert_eq!(frame_to_rgba(&bgra).unwrap(), [1, 2, 3, 4]);
        let bgrx = copy_frame(
            1,
            1,
            spa::param::video::VideoFormat::BGRx,
            4,
            &[3, 2, 1, 0],
            0,
            4,
        )
        .unwrap();
        assert_eq!(frame_to_rgba(&bgrx).unwrap(), [1, 2, 3, 255]);
    }

    #[test]
    fn handles_wrapped_pipewire_chunks_without_copying_other_bytes() {
        let frame = copy_frame(
            1,
            1,
            spa::param::video::VideoFormat::RGBA,
            4,
            &[3, 4, 99, 99, 1, 2],
            4,
            4,
        )
        .unwrap();
        assert_eq!(frame_to_rgba(&frame).unwrap(), [1, 2, 3, 4]);
        assert!(copy_frame(1, 2, spa::param::video::VideoFormat::RGBA, 4, &[0; 8], 0, 4).is_err());
        assert!(
            copy_frame(
                u32::MAX,
                u32::MAX,
                spa::param::video::VideoFormat::RGBA,
                4,
                &[0; 4],
                0,
                4
            )
            .is_err()
        );
    }
}

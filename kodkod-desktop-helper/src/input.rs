// reis Device equality and hashing use stable protocol-object identity.
#![allow(clippy::mutable_key_type)]
//! Portal-scoped EIS input. The actor owns the EI connection for the lifetime
//! of the portal session and never falls back to D-Bus Notify* methods.

use crate::portal::PortalHealth;
use anyhow::{Context, Result, anyhow, bail};
use futures::StreamExt;
use kodkod_executor::{computer::ComputerInput, desktop_helper::HelperInputs};
use reis::{
    PendingRequestResult, ei,
    event::{Device as EisDevice, EiEvent, EiEventConverter},
    tokio::EiEventStream,
};
use std::{
    collections::HashMap,
    os::unix::net::UnixStream,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
    time::Instant,
};

const INITIAL_EVENT_QUIET_PERIOD: Duration = Duration::from_millis(400);
const MAX_EI_TEXT_CHUNK_BYTES: usize = 254;
const LEFT_MOUSE_BUTTON_EVDEV: u32 = 0x110;

/// Atomic capabilities published by the EIS lifecycle actor.
#[derive(Clone)]
pub struct InputHealth {
    open: Arc<AtomicBool>,
    pointer: Arc<AtomicBool>,
    keyboard: Arc<AtomicBool>,
    text: Arc<AtomicBool>,
}

impl InputHealth {
    pub(crate) fn new() -> Self {
        Self {
            open: Arc::new(AtomicBool::new(false)),
            pointer: Arc::new(AtomicBool::new(false)),
            keyboard: Arc::new(AtomicBool::new(false)),
            text: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn is_open(&self) -> bool {
        self.open.load(Ordering::Acquire)
    }

    pub(crate) fn set_open(&self, open: bool) {
        self.open.store(open, Ordering::Release);
    }

    pub fn capabilities(&self) -> HelperInputs {
        HelperInputs {
            pointer: self.pointer.load(Ordering::Acquire),
            keyboard: self.keyboard.load(Ordering::Acquire),
            text: self.text.load(Ordering::Acquire),
        }
    }

    fn clear_capabilities(&self) {
        self.pointer.store(false, Ordering::Release);
        self.keyboard.store(false, Ordering::Release);
        self.text.store(false, Ordering::Release);
    }
}

/// One persistent EIS connection and its single event actor.
pub struct InputActor {
    sender: mpsc::Sender<InputCommand>,
    health: InputHealth,
    task: Option<JoinHandle<()>>,
}

impl InputActor {
    /// Completes the EI sender handshake on the portal-provided EIS FD. The
    /// event actor tracks initial devices and later invalidates the session on
    /// any device or region lifecycle change.
    pub async fn start(fd: std::os::fd::OwnedFd, stream_mapping_id: String) -> Result<Self> {
        let socket = UnixStream::from(fd);
        let context = ei::Context::new(socket).context("Could not create the EI context")?;
        let mut events = EiEventStream::new(context.clone())
            .context("Could not watch the portal EIS connection")?;
        let handshake = reis::tokio::ei_handshake(
            &mut events,
            "Kodkod Desktop Helper",
            ei::handshake::ContextType::Sender,
        )
        .await
        .context("Could not complete the EIS sender handshake")?;
        let text_negotiated = handshake.negotiated_interfaces.contains_key("ei_text");
        let converter = EiEventConverter::new(&context, handshake);
        let connection = converter.connection().clone();
        let health = InputHealth::new();
        health.set_open(true);
        let (sender, receiver) = mpsc::channel(8);
        let task_health = health.clone();
        let task = tokio::task::spawn_local(async move {
            run_actor(
                context,
                connection,
                events,
                converter,
                receiver,
                task_health,
                stream_mapping_id,
                text_negotiated,
            )
            .await;
        });
        Ok(Self {
            sender,
            health,
            task: Some(task),
        })
    }

    pub fn health(&self) -> InputHealth {
        self.health.clone()
    }

    /// Sends one action to the actor. If the caller disconnects after enqueue,
    /// the actor still finishes any paired release before observing its
    /// authorization flag.
    pub async fn send(
        &self,
        input: ComputerInput,
        frame_dimensions: (u32, u32),
        authorized: Arc<AtomicBool>,
        portal_health: PortalHealth,
    ) -> Result<()> {
        let (reply, receive) = oneshot::channel();
        self.sender
            .send(InputCommand::Input {
                input,
                frame_dimensions,
                authorized,
                portal_health,
                reply,
            })
            .await
            .context("The EIS input actor stopped")?;
        receive.await.context("The EIS input actor stopped")?
    }

    /// Stops emulation and joins the lifecycle actor before the portal session
    /// is closed.
    pub async fn close(&mut self) {
        let (reply, receive) = oneshot::channel();
        let _ = self.sender.send(InputCommand::Shutdown { reply }).await;
        let _ = receive.await;
        if let Some(task) = self.task.take() {
            let _ = task.await;
        }
    }
}

enum InputCommand {
    Input {
        input: ComputerInput,
        frame_dimensions: (u32, u32),
        authorized: Arc<AtomicBool>,
        portal_health: PortalHealth,
        reply: oneshot::Sender<Result<()>>,
    },
    Shutdown {
        reply: oneshot::Sender<()>,
    },
}

struct TrackedDevice {
    device: EisDevice,
    active: bool,
    emulating: bool,
}

#[derive(Clone, Copy)]
struct RegionRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

#[derive(Clone)]
struct PointerControl {
    device: EisDevice,
    absolute: ei::PointerAbsolute,
    button: ei::Button,
    region: RegionRect,
}

#[derive(Clone)]
struct KeyboardControl {
    device: EisDevice,
    keyboard: ei::Keyboard,
}

#[derive(Clone)]
struct TextControl {
    device: EisDevice,
    text: ei::Text,
}

#[derive(Clone, Default)]
struct InputControls {
    pointer: Option<PointerControl>,
    keyboard: Option<KeyboardControl>,
    text: Option<TextControl>,
}

#[allow(clippy::too_many_arguments)]
async fn run_actor(
    context: ei::Context,
    connection: reis::event::Connection,
    mut events: EiEventStream,
    mut converter: EiEventConverter,
    mut commands: mpsc::Receiver<InputCommand>,
    health: InputHealth,
    stream_mapping_id: String,
    text_negotiated: bool,
) {
    let mut devices = HashMap::<EisDevice, TrackedDevice>::new();
    let mut controls = InputControls::default();
    let mut settled = false;
    let mut quiet_until = None;
    let mut next_sequence = 1_u32;

    loop {
        let command = if settled {
            tokio::select! {
                biased;
                event = events.next() => {
                    if !handle_event_result(
                        event,
                        &mut converter,
                        &mut devices,
                        &mut next_sequence,
                        &context,
                        &connection,
                        &health,
                        &mut controls,
                        true,
                        &mut quiet_until,
                        text_negotiated,
                    ) {
                        break;
                    }
                    continue;
                }
                command = commands.recv() => command,
            }
        } else if let Some(deadline) = quiet_until {
            tokio::select! {
                biased;
                event = events.next() => {
                    if !handle_event_result(
                        event,
                        &mut converter,
                        &mut devices,
                        &mut next_sequence,
                        &context,
                        &connection,
                        &health,
                        &mut controls,
                        false,
                        &mut quiet_until,
                        text_negotiated,
                    ) {
                        break;
                    }
                    continue;
                }
                _ = tokio::time::sleep_until(deadline) => {
                    controls = select_controls(&devices, &stream_mapping_id, text_negotiated);
                    publish_capabilities(&health, &controls);
                    settled = true;
                    continue;
                }
                command = commands.recv() => command,
            }
        } else {
            tokio::select! {
                biased;
                event = events.next() => {
                    if !handle_event_result(
                        event,
                        &mut converter,
                        &mut devices,
                        &mut next_sequence,
                        &context,
                        &connection,
                        &health,
                        &mut controls,
                        false,
                        &mut quiet_until,
                        text_negotiated,
                    ) {
                        break;
                    }
                    continue;
                }
                command = commands.recv() => command,
            }
        };

        match command {
            Some(InputCommand::Input {
                input,
                frame_dimensions,
                authorized,
                portal_health,
                reply,
            }) => {
                let result = if settled {
                    send_input(
                        &context,
                        &connection,
                        &controls,
                        input,
                        frame_dimensions,
                        &authorized,
                        &portal_health,
                        &health,
                    )
                } else {
                    Err(anyhow!(
                        "EIS device mapping is still initializing; wait and take a fresh screenshot"
                    ))
                };
                let _ = reply.send(result);
            }
            Some(InputCommand::Shutdown { reply }) => {
                let _ = reply.send(());
                break;
            }
            None => break,
        }
    }

    health.clear_capabilities();
    health.open.store(false, Ordering::Release);
    for tracked in devices.values() {
        if tracked.emulating {
            tracked.device.device().stop_emulating(connection.serial());
        }
    }
    let _ = context.flush();
}

#[allow(clippy::too_many_arguments)]
fn handle_event_result(
    result: Option<std::io::Result<PendingRequestResult<ei::Event>>>,
    converter: &mut EiEventConverter,
    devices: &mut HashMap<EisDevice, TrackedDevice>,
    next_sequence: &mut u32,
    context: &ei::Context,
    connection: &reis::event::Connection,
    health: &InputHealth,
    controls: &mut InputControls,
    settled: bool,
    quiet_until: &mut Option<Instant>,
    text_negotiated: bool,
) -> bool {
    let Some(result) = result else {
        health.clear_capabilities();
        health.open.store(false, Ordering::Release);
        return false;
    };
    let result = match result {
        Ok(PendingRequestResult::Request(event)) => event,
        Ok(PendingRequestResult::InvalidObject(_)) => {
            health.clear_capabilities();
            health.open.store(false, Ordering::Release);
            return false;
        }
        Ok(PendingRequestResult::ParseError(_)) | Err(_) => {
            health.clear_capabilities();
            health.open.store(false, Ordering::Release);
            return false;
        }
    };

    if converter.handle_event(result).is_err() {
        health.clear_capabilities();
        health.open.store(false, Ordering::Release);
        return false;
    }

    let mut saw_initial_lifecycle = false;
    let mut lifecycle_changed = false;
    while let Some(event) = converter.next_event() {
        match event {
            EiEvent::SeatAdded(event) => {
                use reis::event::DeviceCapability;
                let mut capabilities = DeviceCapability::PointerAbsolute
                    | DeviceCapability::Button
                    | DeviceCapability::Keyboard;
                if text_negotiated {
                    capabilities |= DeviceCapability::Text;
                }
                event.seat.bind_capabilities(capabilities);
                if context.flush().is_err() {
                    health.clear_capabilities();
                    health.set_open(false);
                    return false;
                }
                saw_initial_lifecycle = true;
                lifecycle_changed = true;
            }
            EiEvent::SeatRemoved(_) => {
                devices.clear();
                saw_initial_lifecycle = true;
                lifecycle_changed = true;
            }
            EiEvent::DeviceAdded(event) => {
                let device = event.device;
                devices.insert(
                    device.clone(),
                    TrackedDevice {
                        device,
                        active: false,
                        emulating: false,
                    },
                );
                saw_initial_lifecycle = true;
                lifecycle_changed = true;
            }
            EiEvent::DeviceResumed(event) => {
                let Some(tracked) = devices.get_mut(&event.device) else {
                    health.clear_capabilities();
                    health.open.store(false, Ordering::Release);
                    return false;
                };
                tracked.active = true;
                if !tracked.emulating {
                    let Some(sequence) = next_sequence.checked_add(1) else {
                        health.clear_capabilities();
                        health.open.store(false, Ordering::Release);
                        return false;
                    };
                    tracked
                        .device
                        .device()
                        .start_emulating(event.serial, *next_sequence);
                    if context.flush().is_err() {
                        health.clear_capabilities();
                        health.open.store(false, Ordering::Release);
                        return false;
                    }
                    tracked.emulating = true;
                    *next_sequence = sequence;
                }
                saw_initial_lifecycle = true;
                lifecycle_changed = true;
            }
            EiEvent::DevicePaused(event) => {
                if let Some(tracked) = devices.get_mut(&event.device) {
                    tracked.active = false;
                    if tracked.emulating {
                        tracked.device.device().stop_emulating(event.serial);
                        let _ = context.flush();
                        tracked.emulating = false;
                    }
                }
                saw_initial_lifecycle = true;
                lifecycle_changed = true;
            }
            EiEvent::DeviceRemoved(event) => {
                if let Some(tracked) = devices.remove(&event.device)
                    && tracked.emulating
                {
                    tracked.device.device().stop_emulating(connection.serial());
                    let _ = context.flush();
                }
                saw_initial_lifecycle = true;
                lifecycle_changed = true;
            }
            EiEvent::Disconnected(_) => {
                health.clear_capabilities();
                health.open.store(false, Ordering::Release);
                return false;
            }
            _ => {}
        }
    }

    if settled && lifecycle_changed {
        health.clear_capabilities();
        health.open.store(false, Ordering::Release);
        return false;
    }
    if !settled && saw_initial_lifecycle {
        *quiet_until = Some(Instant::now() + INITIAL_EVENT_QUIET_PERIOD);
    }

    if settled && !controls_alive(controls) {
        health.clear_capabilities();
        health.open.store(false, Ordering::Release);
        return false;
    }
    true
}

fn select_controls(
    devices: &HashMap<EisDevice, TrackedDevice>,
    mapping_id: &str,
    text_negotiated: bool,
) -> InputControls {
    let mut pointer_candidates = Vec::new();
    let mut keyboard_candidates = Vec::new();
    let mut text_candidates = Vec::new();

    if mapping_id.is_empty() {
        return InputControls::default();
    }

    for tracked in devices.values().filter(|tracked| tracked.active) {
        let device = &tracked.device;
        if device.device_type() != ei::device::DeviceType::Virtual {
            continue;
        }
        if let (Some(absolute), Some(button)) = (
            device.interface::<ei::PointerAbsolute>(),
            device.interface::<ei::Button>(),
        ) {
            for region in device
                .regions()
                .iter()
                .filter(|region| region.mapping_id.as_deref() == Some(mapping_id))
            {
                if region.width > 0 && region.height > 0 {
                    pointer_candidates.push(PointerControl {
                        device: device.clone(),
                        absolute: absolute.clone(),
                        button: button.clone(),
                        region: RegionRect {
                            x: region.x,
                            y: region.y,
                            width: region.width,
                            height: region.height,
                        },
                    });
                }
            }
        }
        if let Some(keyboard) = device.interface::<ei::Keyboard>() {
            keyboard_candidates.push(KeyboardControl {
                device: device.clone(),
                keyboard,
            });
        }
        if text_negotiated && let Some(text) = device.interface::<ei::Text>() {
            text_candidates.push(TextControl {
                device: device.clone(),
                text,
            });
        }
    }

    InputControls {
        pointer: (pointer_candidates.len() == 1).then(|| pointer_candidates.remove(0)),
        keyboard: (keyboard_candidates.len() == 1).then(|| keyboard_candidates.remove(0)),
        text: (text_candidates.len() == 1).then(|| text_candidates.remove(0)),
    }
}

fn publish_capabilities(health: &InputHealth, controls: &InputControls) {
    health
        .pointer
        .store(controls.pointer.is_some(), Ordering::Release);
    health
        .keyboard
        .store(controls.keyboard.is_some(), Ordering::Release);
    health
        .text
        .store(controls.text.is_some(), Ordering::Release);
}

fn controls_alive(controls: &InputControls) -> bool {
    controls.pointer.as_ref().is_none_or(|pointer| {
        pointer.device.device().is_alive()
            && pointer.absolute.is_alive()
            && pointer.button.is_alive()
    }) && controls
        .keyboard
        .as_ref()
        .is_none_or(|keyboard| keyboard.device.device().is_alive() && keyboard.keyboard.is_alive())
        && controls
            .text
            .as_ref()
            .is_none_or(|text| text.device.device().is_alive() && text.text.is_alive())
}

#[allow(clippy::too_many_arguments)]
fn send_input(
    context: &ei::Context,
    connection: &reis::event::Connection,
    controls: &InputControls,
    input: ComputerInput,
    frame_dimensions: (u32, u32),
    authorized: &AtomicBool,
    portal_health: &PortalHealth,
    health: &InputHealth,
) -> Result<()> {
    ensure_authorized(authorized, portal_health, health)?;
    input.validate()?;
    send_input_capability_error(&input, health.capabilities())?;
    match input {
        ComputerInput::Click { x, y } => {
            let pointer = controls
                .pointer
                .as_ref()
                .context("No unique active absolute-pointer region maps to this screen stream")?;
            let (x, y) = pixel_to_region(x, y, frame_dimensions, pointer.region)?;
            ensure_authorized(authorized, portal_health, health)?;
            pointer.absolute.motion_absolute(x, y);
            frame(context, &pointer.device, connection.serial())?;
            ensure_authorized(authorized, portal_health, health)?;
            pointer
                .button
                .button(LEFT_MOUSE_BUTTON_EVDEV, ei::button::ButtonState::Press);
            if let Err(error) = frame(context, &pointer.device, connection.serial()) {
                let _ = release_button(context, connection, pointer);
                return Err(error).context("Click may have started; do not replay it");
            }
            if let Err(error) = release_button(context, connection, pointer) {
                bail!(
                    "Click may have completed but the button release failed; do not replay it: {error}"
                );
            }
        }
        ComputerInput::Type { text } => {
            let text_control = controls
                .text
                .as_ref()
                .context("The active keyboard does not provide negotiated ei_text support")?;
            let chunks = prepare_utf8_chunks(&text)?;
            for chunk in chunks {
                ensure_authorized(authorized, portal_health, health)?;
                text_control.text.utf8(chunk);
                frame(context, &text_control.device, connection.serial())
                    .context("Text may have been inserted; do not replay it")?;
            }
        }
        ComputerInput::Key { key } => {
            if let Some(evdev_key) = layout_independent_key(&key) {
                let keyboard = controls
                    .keyboard
                    .as_ref()
                    .context("No unique active virtual keyboard is available")?;
                ensure_authorized(authorized, portal_health, health)?;
                keyboard
                    .keyboard
                    .key(evdev_key, ei::keyboard::KeyState::Press);
                if let Err(error) = frame(context, &keyboard.device, connection.serial()) {
                    let _ = release_evdev_key(context, connection, keyboard, evdev_key);
                    return Err(error).context("Key may have been pressed; do not replay it");
                }
                if let Err(error) = release_evdev_key(context, connection, keyboard, evdev_key) {
                    bail!("Key may have completed but release failed; do not replay it: {error}");
                }
            } else {
                let text_control = controls
                    .text
                    .as_ref()
                    .context("This named key chord requires negotiated ei_text keysym support")?;
                let (modifier, keysym) = named_keysym(&key)?;
                send_keysym(
                    context,
                    connection,
                    text_control,
                    modifier,
                    keysym,
                    authorized,
                    portal_health,
                    health,
                )?;
            }
        }
    }
    anyhow::ensure!(
        authorized.load(Ordering::Acquire),
        "Input may have completed before the IPC client disconnected; do not replay it"
    );
    Ok(())
}

fn ensure_authorized(
    authorized: &AtomicBool,
    portal_health: &PortalHealth,
    health: &InputHealth,
) -> Result<()> {
    anyhow::ensure!(
        authorized.load(Ordering::Acquire),
        "IPC client disconnected; stop sending new desktop input"
    );
    anyhow::ensure!(
        portal_health.is_ready() && health.is_open(),
        "Portal capture or EIS input session is no longer active"
    );
    Ok(())
}

fn frame(context: &ei::Context, device: &EisDevice, serial: u32) -> Result<()> {
    device.device().frame(serial, monotonic_micros()?);
    context
        .flush()
        .context("Could not flush the EIS input frame")?;
    Ok(())
}

fn release_button(
    context: &ei::Context,
    connection: &reis::event::Connection,
    pointer: &PointerControl,
) -> Result<()> {
    pointer
        .button
        .button(LEFT_MOUSE_BUTTON_EVDEV, ei::button::ButtonState::Released);
    frame(context, &pointer.device, connection.serial())
}

fn release_evdev_key(
    context: &ei::Context,
    connection: &reis::event::Connection,
    keyboard: &KeyboardControl,
    key: u32,
) -> Result<()> {
    keyboard.keyboard.key(key, ei::keyboard::KeyState::Released);
    frame(context, &keyboard.device, connection.serial())
}

#[allow(clippy::too_many_arguments)]
fn send_keysym(
    context: &ei::Context,
    connection: &reis::event::Connection,
    control: &TextControl,
    modifier: Option<u32>,
    keysym: u32,
    authorized: &AtomicBool,
    portal_health: &PortalHealth,
    health: &InputHealth,
) -> Result<()> {
    if let Some(modifier) = modifier {
        ensure_authorized(authorized, portal_health, health)?;
        control.text.keysym(modifier, ei::keyboard::KeyState::Press);
        if let Err(error) = frame(context, &control.device, connection.serial()) {
            let _ = release_keysym(context, connection, control, modifier);
            return Err(error)
                .context("Modifier may have been pressed; do not replay the key chord");
        }
        if let Err(error) = ensure_authorized(authorized, portal_health, health) {
            let _ = release_keysym(context, connection, control, modifier);
            return Err(error);
        }
    } else {
        ensure_authorized(authorized, portal_health, health)?;
    }

    control.text.keysym(keysym, ei::keyboard::KeyState::Press);
    if let Err(error) = frame(context, &control.device, connection.serial()) {
        let _ = release_keysym(context, connection, control, keysym);
        if let Some(modifier) = modifier {
            let _ = release_keysym(context, connection, control, modifier);
        }
        return Err(error).context("Keysym may have been pressed; do not replay it");
    }

    if let Err(error) = release_keysym(context, connection, control, keysym) {
        if let Some(modifier) = modifier {
            let _ = release_keysym(context, connection, control, modifier);
        }
        return Err(error).context("Keysym may have completed; do not replay it");
    }
    if let Some(modifier) = modifier {
        // Release modifiers even when the IPC caller disconnected or the
        // portal session changed after the target key went down.
        release_keysym(context, connection, control, modifier)
            .context("Modifier release failed; do not replay the key chord")?;
    }
    Ok(())
}

fn release_keysym(
    context: &ei::Context,
    connection: &reis::event::Connection,
    control: &TextControl,
    keysym: u32,
) -> Result<()> {
    control
        .text
        .keysym(keysym, ei::keyboard::KeyState::Released);
    frame(context, &control.device, connection.serial())
}

fn layout_independent_key(key: &str) -> Option<u32> {
    // These evdev codes identify dedicated control keys across layouts. Letter
    // chords are sent as XKB keysyms only through negotiated ei_text.
    match key {
        "Return" => Some(28),
        "Tab" => Some(15),
        "Escape" => Some(1),
        "BackSpace" => Some(14),
        "Up" => Some(103),
        "Down" => Some(108),
        "Left" => Some(105),
        "Right" => Some(106),
        _ => None,
    }
}

fn named_keysym(key: &str) -> Result<(Option<u32>, u32)> {
    let (modifier, key) = match key {
        "ctrl+l" => (Some(0xffe3), u32::from(b'l')),
        "ctrl+a" => (Some(0xffe3), u32::from(b'a')),
        "ctrl+c" => (Some(0xffe3), u32::from(b'c')),
        "ctrl+v" => (Some(0xffe3), u32::from(b'v')),
        "cmd+l" => (Some(0xffeb), u32::from(b'l')),
        "cmd+a" => (Some(0xffeb), u32::from(b'a')),
        "cmd+c" => (Some(0xffeb), u32::from(b'c')),
        "cmd+v" => (Some(0xffeb), u32::from(b'v')),
        other => bail!("Unsupported portal key: {other}"),
    };
    Ok((modifier, key))
}

fn prepare_utf8_chunks(text: &str) -> Result<Vec<&str>> {
    anyhow::ensure!(!text.is_empty(), "Desktop text input cannot be empty");
    anyhow::ensure!(
        !text.contains('\0'),
        "Desktop text input cannot contain a NUL character"
    );
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let mut end = start;
        for (offset, character) in text[start..].char_indices() {
            let next = start + offset + character.len_utf8();
            if next - start > MAX_EI_TEXT_CHUNK_BYTES {
                break;
            }
            end = next;
        }
        anyhow::ensure!(end > start, "Could not split UTF-8 at an EI text boundary");
        chunks.push(&text[start..end]);
        start = end;
    }
    anyhow::ensure!(
        chunks
            .iter()
            .all(|chunk| !chunk.is_empty() && chunk.len() <= MAX_EI_TEXT_CHUNK_BYTES),
        "EI text chunk exceeds the protocol limit"
    );
    Ok(chunks)
}

fn pixel_to_region(x: u32, y: u32, frame: (u32, u32), region: RegionRect) -> Result<(f32, f32)> {
    anyhow::ensure!(
        frame.0 > 0 && frame.1 > 0 && x < frame.0 && y < frame.1,
        "Desktop coordinates are outside the observed frame"
    );
    anyhow::ensure!(
        region.width > 0 && region.height > 0,
        "Mapped EIS region has invalid dimensions"
    );
    let x =
        f64::from(region.x) + (f64::from(x) + 0.5) * f64::from(region.width) / f64::from(frame.0);
    let y =
        f64::from(region.y) + (f64::from(y) + 0.5) * f64::from(region.height) / f64::from(frame.1);
    anyhow::ensure!(
        x.is_finite() && y.is_finite(),
        "Mapped EIS coordinates are invalid"
    );
    Ok((x as f32, y as f32))
}

fn monotonic_micros() -> Result<u64> {
    let mut value = std::mem::MaybeUninit::<libc::timespec>::uninit();
    // SAFETY: clock_gettime writes a complete timespec on success.
    let result = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, value.as_mut_ptr()) };
    anyhow::ensure!(result == 0, "Could not read CLOCK_MONOTONIC for EIS input");
    // SAFETY: the successful call above initialized the timespec.
    let value = unsafe { value.assume_init() };
    let seconds = u64::try_from(value.tv_sec).context("CLOCK_MONOTONIC seconds were invalid")?;
    let nanos = u64::try_from(value.tv_nsec).context("CLOCK_MONOTONIC nanoseconds were invalid")?;
    seconds
        .checked_mul(1_000_000)
        .and_then(|micros| micros.checked_add(nanos / 1_000))
        .context("CLOCK_MONOTONIC timestamp overflowed")
}

fn send_input_capability_error(input: &ComputerInput, capabilities: HelperInputs) -> Result<()> {
    match input {
        ComputerInput::Click { .. } if !capabilities.pointer => {
            bail!("No unique active absolute-pointer region maps to this screen stream")
        }
        ComputerInput::Type { .. } if !capabilities.text => {
            bail!("The active keyboard does not provide negotiated ei_text UTF-8 support")
        }
        ComputerInput::Key { key }
            if layout_independent_key(key).is_none() && !capabilities.text =>
        {
            bail!("This named key chord requires negotiated ei_text keysym support")
        }
        ComputerInput::Key { key }
            if layout_independent_key(key).is_some() && !capabilities.keyboard =>
        {
            bail!("No unique active virtual keyboard is available")
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn screenshot_pixels_map_to_current_logical_region() {
        let region = RegionRect {
            x: 100,
            y: 200,
            width: 1000,
            height: 500,
        };
        assert_eq!(
            pixel_to_region(999, 499, (2000, 1000), region).unwrap(),
            (599.75, 449.75)
        );
        assert_eq!(
            pixel_to_region(0, 0, (2000, 1000), region).unwrap(),
            (100.25, 200.25)
        );
        let scaled = RegionRect {
            width: 2000,
            ..region
        };
        assert_eq!(
            pixel_to_region(0, 0, (2000, 1000), scaled).unwrap(),
            (100.5, 200.25)
        );
        assert!(pixel_to_region(2000, 0, (2000, 1000), region).is_err());
        assert!(pixel_to_region(0, 0, (0, 1000), region).is_err());
    }

    #[test]
    fn unicode_chunks_preserve_text_and_protocol_limits() {
        let text = "å🦊漢".repeat(100);
        let chunks = prepare_utf8_chunks(&text).unwrap();
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|chunk| chunk.len() <= 254));
        assert_eq!(chunks.concat(), text);
        assert!(prepare_utf8_chunks("").is_err());
        assert!(prepare_utf8_chunks("before\0after").is_err());
    }

    #[test]
    fn text_negotiation_is_independent_of_physical_keyboard() {
        let physical = HelperInputs {
            pointer: true,
            keyboard: true,
            text: false,
        };
        assert!(
            send_input_capability_error(
                &ComputerInput::Key {
                    key: "Return".into()
                },
                physical
            )
            .is_ok()
        );
        assert!(
            send_input_capability_error(
                &ComputerInput::Type {
                    text: "test".into()
                },
                physical
            )
            .is_err()
        );
        assert!(
            send_input_capability_error(
                &ComputerInput::Key {
                    key: "ctrl+l".into()
                },
                physical
            )
            .is_err()
        );
        let text_only = HelperInputs {
            pointer: false,
            keyboard: false,
            text: true,
        };
        assert!(
            send_input_capability_error(
                &ComputerInput::Key {
                    key: "ctrl+l".into()
                },
                text_only
            )
            .is_ok()
        );
        assert!(
            send_input_capability_error(
                &ComputerInput::Key {
                    key: "Return".into()
                },
                text_only
            )
            .is_err()
        );
    }
}

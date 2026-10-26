//! Serialized native macOS app automation.
//!
//! AXUIElement calls synchronously message another process and the handles are
//! opaque Core Foundation objects.  This module therefore keeps every handle
//! on one worker thread and exposes only bounded, UUID-addressed runtime data
//! to the async side.

use super::{ax, bounded_ns_string, capture};
use crate::computer::{
    APP_OBSERVATION_MAX_AGE, AppAction, AppActionRequest, AppAutomationSession, AppList,
    AppObservation, AppSummary, DesktopScreenshot, MAX_LOCAL_APP_ACTIONS,
    MAX_LOCAL_APP_BUNDLE_ID_BYTES, MAX_LOCAL_APP_ELEMENTS, MAX_LOCAL_APP_LABEL_BYTES,
    MAX_LOCAL_APP_NAME_BYTES, MAX_LOCAL_APP_ROLE_BYTES, MAX_LOCAL_APP_WINDOW_TITLE_BYTES,
    MAX_LOCAL_APPS, ObserveAppRequest, ObservedAppElement, ObservedAppTarget,
};
use anyhow::{Context, Result, bail};
use futures::future::BoxFuture;
use objc::{
    class, msg_send,
    runtime::{Class, Object},
    sel, sel_impl,
};
use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
    sync::mpsc::{self, Receiver, SyncSender, TrySendError},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};
use tokio::sync::oneshot;
use uuid::Uuid;

const MAX_TREE_DEPTH: usize = 32;
const MAX_TREE_NODES: usize = 4096;
const MAX_WINDOW_CHOICES: usize = 8;
const AX_TIMEOUT_SECONDS: f32 = 0.75;
const TREE_DEADLINE: Duration = Duration::from_secs(3);

pub struct NativeAppAutomation {
    sender: SyncSender<Command>,
}

impl NativeAppAutomation {
    pub fn new(gate: Arc<Mutex<()>>) -> Self {
        let (sender, receiver) = mpsc::sync_channel(32);
        let _ = thread::Builder::new()
            .name("kodkod-macos-app-automation".to_owned())
            .spawn(move || Worker::new(gate).run(receiver));
        Self { sender }
    }

    fn try_send<T>(
        &self,
        command: Command,
        receiver: oneshot::Receiver<Result<T>>,
    ) -> BoxFuture<'static, Result<T>>
    where
        T: Send + 'static,
    {
        let result = self.sender.try_send(command).map_err(|error| match error {
            TrySendError::Full(_) => anyhow::anyhow!(
                "The macOS app automation worker is busy; retry after the current operation"
            ),
            TrySendError::Disconnected(_) => {
                anyhow::anyhow!("The macOS app automation worker stopped")
            }
        });
        Box::pin(async move {
            result?;
            receiver
                .await
                .map_err(|_| anyhow::anyhow!("The macOS app automation worker stopped"))?
        })
    }
}

impl AppAutomationSession for NativeAppAutomation {
    fn list_apps<'a>(&'a self) -> BoxFuture<'a, Result<AppList>> {
        let (sender, receiver) = oneshot::channel();
        self.try_send(Command::List { responder: sender }, receiver)
    }

    fn observe_app<'a>(
        &'a self,
        request: ObserveAppRequest,
    ) -> BoxFuture<'a, Result<AppObservation>> {
        let (sender, receiver) = oneshot::channel();
        self.try_send(
            Command::Observe {
                request,
                responder: sender,
            },
            receiver,
        )
    }

    fn act_on_app<'a>(&'a self, request: AppActionRequest) -> BoxFuture<'a, Result<()>> {
        let (sender, receiver) = oneshot::channel();
        self.try_send(
            Command::Act {
                request,
                responder: sender,
            },
            receiver,
        )
    }
}

enum Command {
    List {
        responder: oneshot::Sender<Result<AppList>>,
    },
    Observe {
        request: ObserveAppRequest,
        responder: oneshot::Sender<Result<AppObservation>>,
    },
    Act {
        request: AppActionRequest,
        responder: oneshot::Sender<Result<()>>,
    },
}

struct Worker {
    gate: Arc<Mutex<()>>,
    apps: HashMap<String, AppRecord>,
    observation: Option<ObservationState>,
}

impl Worker {
    fn new(gate: Arc<Mutex<()>>) -> Self {
        Self {
            gate,
            apps: HashMap::new(),
            observation: None,
        }
    }

    fn run(mut self, receiver: Receiver<Command>) {
        for command in receiver {
            let gate = self.gate.clone();
            let Ok(_gate) = gate.lock() else {
                break;
            };
            objc::rc::autoreleasepool(|| match command {
                Command::List { responder } => {
                    if !responder.is_closed() {
                        let _ = responder.send(self.list_apps());
                    }
                }
                Command::Observe { request, responder } => {
                    if !responder.is_closed() {
                        let _ = responder.send(self.observe_app(request));
                    }
                }
                Command::Act { request, responder } => {
                    if !responder.is_closed() {
                        let result = self.act_on_app(request, &responder);
                        let _ = responder.send(result);
                    }
                }
            });
        }
    }

    fn list_apps(&mut self) -> Result<AppList> {
        self.observation = None;
        if !ax::trusted() {
            bail!("Accessibility access is required to list local applications");
        }
        let mut identities = running_applications()?;
        identities.sort_by(|first, second| {
            first
                .name
                .cmp(&second.name)
                .then_with(|| first.bundle_id.cmp(&second.bundle_id))
                .then_with(|| first.pid.cmp(&second.pid))
        });
        identities.truncate(MAX_LOCAL_APPS);

        // App IDs are per-list handles.  A fresh list also invalidates IDs
        // from a prior snapshot, which prevents stale process/window choices
        // from being silently reused.
        self.apps.clear();
        for identity in identities {
            let id = Uuid::now_v7().to_string();
            self.apps.insert(id, AppRecord { identity });
        }
        let mut apps: Vec<_> = self
            .apps
            .iter()
            .map(|(id, record)| AppSummary {
                id: id.clone(),
                bundle_id: record.identity.bundle_id.clone(),
                name: record.identity.name.clone(),
            })
            .collect();
        apps.sort_by(|first, second| {
            first
                .name
                .cmp(&second.name)
                .then_with(|| first.bundle_id.cmp(&second.bundle_id))
                .then_with(|| first.id.cmp(&second.id))
        });
        Ok(AppList { apps })
    }

    fn observe_app(&mut self, request: ObserveAppRequest) -> Result<AppObservation> {
        self.observation = None;
        request.validate()?;
        if !ax::trusted() {
            bail!("Accessibility access is required to inspect a local application");
        }
        let record = self
            .apps
            .get(&request.app_id)
            .ok_or_else(|| anyhow::anyhow!("The app handle is stale; list local apps again"))?;
        let identity = current_identity(&record.identity)?;
        let application = ax::Element::application(identity.pid)?;
        ax::set_messaging_timeout(&application, AX_TIMEOUT_SECONDS)?;
        anyhow::ensure!(
            ax::process_id(&application)? == identity.pid,
            "The application process changed while it was being observed"
        );

        let selected = select_window(&application, request.window_title.as_deref())?;
        let mut collector = Collector::default();
        let deadline = Instant::now() + TREE_DEADLINE;
        collect_tree(
            &selected.element,
            0,
            &mut collector,
            &mut VisitedElements::default(),
            deadline,
        )?;

        let observation_id = Uuid::now_v7().to_string();
        let mut observed_elements: Vec<_> = collector.elements.iter().collect();
        observed_elements.sort_by_key(|(id, _)| *id);
        let elements = observed_elements
            .into_iter()
            .map(|(id, element)| ObservedAppElement {
                id: id.clone(),
                role: element.role.clone(),
                label: element.label.clone(),
                enabled: element.enabled,
                editable: element.editable,
                actions: element.actions.clone(),
                value: element.value.clone(),
            })
            .collect();
        let screenshot = capture::capture_window_png(identity.pid, &selected.title)
            .ok()
            .flatten()
            .and_then(|png| DesktopScreenshot::from_png(png).ok());

        // A fresh observation supersedes every prior handle.  Keeping one
        // snapshot bounds retained AX references and stale action surface.
        self.observation = Some(ObservationState {
            id: observation_id.clone(),
            identity: identity.clone(),
            window: selected.element,
            window_title: selected.title.clone(),
            elements: collector.elements,
            created: Instant::now(),
        });

        Ok(AppObservation {
            id: observation_id,
            target: ObservedAppTarget {
                bundle_id: identity.bundle_id,
                name: identity.name,
                window_title: selected.title,
            },
            elements,
            screenshot,
        })
    }

    fn act_on_app(
        &mut self,
        request: AppActionRequest,
        responder: &oneshot::Sender<Result<()>>,
    ) -> Result<()> {
        request.validate()?;
        if responder.is_closed() {
            return Ok(());
        }
        // Removing before any validation makes every admitted action attempt
        // consume its observation, including stale or failed attempts.
        let observation = self
            .observation
            .take()
            .ok_or_else(|| anyhow::anyhow!("The app observation is stale or already used"))?;
        anyhow::ensure!(
            request.observation_id == observation.id,
            "The app observation is stale or already used"
        );
        anyhow::ensure!(
            observation.created.elapsed() <= APP_OBSERVATION_MAX_AGE,
            "The app observation expired; observe the application again"
        );
        let identity = current_identity(&observation.identity)?;
        let application = ax::Element::application(identity.pid)?;
        ax::set_messaging_timeout(&application, AX_TIMEOUT_SECONDS)?;
        let selected = select_window(&application, Some(&observation.window_title))?;
        anyhow::ensure!(
            ax::same(&selected.element, &observation.window),
            "The target window changed; observe the application again"
        );
        anyhow::ensure!(
            ax::process_id(&selected.element)? == identity.pid,
            "The target window belongs to a different process"
        );

        let element_state = observation
            .elements
            .get(&request.element_id)
            .ok_or_else(|| anyhow::anyhow!("The app element is not in this observation"))?;
        let live = find_live_element(
            &selected.element,
            &element_state.element,
            Instant::now() + TREE_DEADLINE,
            &mut VisitedElements::default(),
            0,
        )?
        .ok_or_else(|| anyhow::anyhow!("The app element changed; observe the application again"))?;
        anyhow::ensure!(
            ax::process_id(&live)? == identity.pid,
            "The app element belongs to a different process; observe the application again"
        );
        let current = read_semantics(&live)?;
        anyhow::ensure!(
            current.role == element_state.role,
            "The app element role changed; observe the application again"
        );
        anyhow::ensure!(
            current.label == element_state.label,
            "The app element label changed; observe the application again"
        );
        if let Some(expected) = element_state.value_digest {
            anyhow::ensure!(
                current.value_digest == Some(expected),
                "The app element value changed; observe the application again"
            );
        } else if matches!(request.action, AppAction::SetText { .. }) {
            bail!("The editable value could not be verified; observe the application again");
        }
        anyhow::ensure!(
            current.enabled,
            "The app element is disabled; observe the application again"
        );
        if responder.is_closed() {
            return Ok(());
        }

        match request.action {
            AppAction::Press {} => {
                anyhow::ensure!(
                    current.actions.iter().any(|action| action == "press"),
                    "The app element does not currently support AXPress"
                );
                if responder.is_closed() {
                    return Ok(());
                }
                ax::perform_action(&live, ax::PRESS_ACTION)
            }
            AppAction::SetText { text } => {
                anyhow::ensure!(
                    current.editable && is_editable_role(&current.role),
                    "The app element is not an editable text field"
                );
                anyhow::ensure!(
                    ax::is_attribute_settable(&live, ax::VALUE_ATTRIBUTE)?,
                    "The app element does not currently allow text changes"
                );
                if responder.is_closed() {
                    return Ok(());
                }
                ax::set_string_attribute(&live, ax::VALUE_ATTRIBUTE, &text)
            }
        }
    }
}

#[derive(Clone)]
struct Identity {
    pid: i32,
    bundle_id: String,
    name: String,
    launch_date: f64,
}

impl Identity {
    fn same_process(&self, other: &Self) -> bool {
        self.pid == other.pid
            && self.bundle_id == other.bundle_id
            && self.name == other.name
            && self.launch_date.to_bits() == other.launch_date.to_bits()
    }
}

struct AppRecord {
    identity: Identity,
}

struct ObservationState {
    id: String,
    identity: Identity,
    window: ax::Element,
    window_title: String,
    elements: HashMap<String, ElementState>,
    created: Instant,
}

struct ElementState {
    element: ax::Element,
    role: String,
    label: String,
    value: Option<String>,
    enabled: bool,
    editable: bool,
    actions: Vec<String>,
    value_digest: Option<u64>,
}

#[derive(Default)]
struct Collector {
    elements: HashMap<String, ElementState>,
    nodes: usize,
}

struct WindowSelection {
    element: ax::Element,
    title: String,
}

#[derive(Default)]
struct VisitedElements {
    elements: HashMap<usize, ax::Element>,
}

impl VisitedElements {
    fn insert(&mut self, element: &ax::Element) -> bool {
        let token = element.identity_token();
        if let std::collections::hash_map::Entry::Vacant(entry) = self.elements.entry(token) {
            // Keep the first AX wrapper alive for this traversal. Core
            // Foundation may reuse a released wrapper address for a later
            // element otherwise.
            entry.insert(element.clone());
            true
        } else {
            false
        }
    }

    fn len(&self) -> usize {
        self.elements.len()
    }
}

fn running_applications() -> Result<Vec<Identity>> {
    let workspace: *mut Object = unsafe { msg_send![class!(NSWorkspace), sharedWorkspace] };
    if workspace.is_null() {
        bail!("NSWorkspace is unavailable")
    }
    let applications: *mut Object = unsafe { msg_send![workspace, runningApplications] };
    if applications.is_null() {
        bail!("NSWorkspace returned no running applications")
    }
    let count: usize = unsafe { msg_send![applications, count] };
    anyhow::ensure!(count <= 4096, "NSWorkspace returned too many applications");
    let mut result = Vec::new();
    for index in 0..count {
        let application: *mut Object = unsafe { msg_send![applications, objectAtIndex: index] };
        if let Some(identity) = identity_from_running_application(application) {
            result.push(identity);
        }
    }
    Ok(result)
}

fn current_identity(expected: &Identity) -> Result<Identity> {
    let Some(class) = Class::get("NSRunningApplication") else {
        bail!("NSRunningApplication is unavailable")
    };
    let application: *mut Object =
        unsafe { msg_send![class, runningApplicationWithProcessIdentifier: expected.pid] };
    let Some(identity) = identity_from_running_application(application) else {
        bail!("The application has exited; list local apps again")
    };
    anyhow::ensure!(
        identity.same_process(expected),
        "The application process was replaced; list local apps again"
    );
    Ok(identity)
}

fn identity_from_running_application(application: *mut Object) -> Option<Identity> {
    if application.is_null() {
        return None;
    }
    let terminated: u8 = unsafe { msg_send![application, isTerminated] };
    if terminated != 0 {
        return None;
    }
    let activation_policy: isize = unsafe { msg_send![application, activationPolicy] };
    if activation_policy != 0 {
        return None;
    }
    let pid: i32 = unsafe { msg_send![application, processIdentifier] };
    if pid <= 0 {
        return None;
    }
    let bundle_id = bounded_ns_string(unsafe { msg_send![application, bundleIdentifier] })?;
    let name = bounded_ns_string(unsafe { msg_send![application, localizedName] })?;
    if !valid_metadata(&bundle_id, MAX_LOCAL_APP_BUNDLE_ID_BYTES)
        || !valid_metadata(&name, MAX_LOCAL_APP_NAME_BYTES)
    {
        return None;
    }
    let launch_date: *mut Object = unsafe { msg_send![application, launchDate] };
    if launch_date.is_null() {
        // Without LaunchServices launch time, PID reuse cannot be rejected
        // reliably, so the process is deliberately not exposed.
        return None;
    }
    let launch_date: f64 = unsafe { msg_send![launch_date, timeIntervalSince1970] };
    launch_date.is_finite().then_some(Identity {
        pid,
        bundle_id,
        name,
        launch_date,
    })
}

fn select_window(
    application: &ax::Element,
    requested_title: Option<&str>,
) -> Result<WindowSelection> {
    let windows = ax::windows(application)?;
    anyhow::ensure!(
        !windows.is_empty(),
        "The application has no accessible windows"
    );
    let mut titled = Vec::with_capacity(windows.len());
    for window in &windows {
        let _ = ax::set_messaging_timeout(window, AX_TIMEOUT_SECONDS);
        titled.push((window.clone(), window_title(window)?));
    }

    if let Some(requested_title) = requested_title {
        let matches: Vec<_> = titled
            .iter()
            .filter(|(_, title)| title == requested_title)
            .collect();
        anyhow::ensure!(
            matches.len() == 1,
            if matches.is_empty() {
                format!(
                    "No window matched the exact title {requested_title:?}; available titles: {}",
                    format_window_choices(&titled)
                )
            } else {
                format!(
                    "Multiple windows have the exact title {requested_title:?}; choose a more specific app state"
                )
            }
        );
        return Ok(WindowSelection {
            element: matches[0].0.clone(),
            title: matches[0].1.clone(),
        });
    }

    anyhow::ensure!(
        titled.len() == 1,
        "The application has multiple windows; provide one exact title: {}",
        format_window_choices(&titled)
    );
    let (element, title) = titled.into_iter().next().expect("window count checked");
    Ok(WindowSelection { element, title })
}

fn window_title(window: &ax::Element) -> Result<String> {
    let title = ax::string_attribute(window, ax::TITLE_ATTRIBUTE)?.unwrap_or_default();
    anyhow::ensure!(
        valid_window_title(&title),
        "The target window title is invalid or too long"
    );
    Ok(title)
}

fn format_window_choices(windows: &[(ax::Element, String)]) -> String {
    let mut choices = Vec::new();
    for (_, title) in windows.iter().take(MAX_WINDOW_CHOICES) {
        choices.push(format!("{title:?}"));
    }
    if windows.len() > MAX_WINDOW_CHOICES {
        choices.push(format!("and {} more", windows.len() - MAX_WINDOW_CHOICES));
    }
    choices.join(", ")
}

fn collect_tree(
    element: &ax::Element,
    depth: usize,
    collector: &mut Collector,
    seen: &mut VisitedElements,
    deadline: Instant,
) -> Result<()> {
    if Instant::now() >= deadline {
        bail!("The accessibility tree exceeded the inspection time limit")
    }
    if depth > MAX_TREE_DEPTH || collector.nodes >= MAX_TREE_NODES {
        return Ok(());
    }
    if !seen.insert(element) {
        return Ok(());
    }
    collector.nodes += 1;
    let _ = ax::set_messaging_timeout(element, AX_TIMEOUT_SECONDS);
    let semantics = read_semantics(element)?;
    if !semantics.role.is_empty()
        && (!semantics.label.is_empty()
            || semantics.value.is_some()
            || !semantics.actions.is_empty()
            || semantics.editable)
        && collector.elements.len() < MAX_LOCAL_APP_ELEMENTS
    {
        let id = Uuid::now_v7().to_string();
        collector.elements.insert(
            id,
            ElementState {
                element: element.clone(),
                role: semantics.role,
                label: semantics.label,
                value: semantics.value,
                enabled: semantics.enabled,
                editable: semantics.editable,
                actions: semantics.actions,
                value_digest: semantics.value_digest,
            },
        );
    }
    if collector.elements.len() >= MAX_LOCAL_APP_ELEMENTS {
        return Ok(());
    }
    for child in tree_children(element, depth)? {
        collect_tree(&child, depth + 1, collector, seen, deadline)?;
    }
    Ok(())
}

fn find_live_element(
    element: &ax::Element,
    target: &ax::Element,
    deadline: Instant,
    seen: &mut VisitedElements,
    depth: usize,
) -> Result<Option<ax::Element>> {
    if Instant::now() >= deadline {
        bail!("The accessibility tree exceeded the action time limit")
    }
    if depth > MAX_TREE_DEPTH || seen.len() >= MAX_TREE_NODES {
        return Ok(None);
    }
    if !seen.insert(element) {
        return Ok(None);
    }
    let _ = ax::set_messaging_timeout(element, AX_TIMEOUT_SECONDS);
    if ax::same(element, target) {
        return Ok(Some(element.clone()));
    }
    for child in tree_children(element, depth)? {
        if let Some(found) = find_live_element(&child, target, deadline, seen, depth + 1)? {
            return Ok(Some(found));
        }
    }
    Ok(None)
}

fn tree_children(element: &ax::Element, depth: usize) -> Result<Vec<ax::Element>> {
    let mut children = ax::children(element)?;
    if depth == 0
        && let Some(toolbar) = ax::toolbar(element)?
        && !children.iter().any(|child| ax::same(child, &toolbar))
    {
        children.push(toolbar);
    }
    Ok(children)
}

struct Semantics {
    role: String,
    label: String,
    value: Option<String>,
    enabled: bool,
    editable: bool,
    actions: Vec<String>,
    value_digest: Option<u64>,
}

fn read_semantics(element: &ax::Element) -> Result<Semantics> {
    let _ = ax::set_messaging_timeout(element, AX_TIMEOUT_SECONDS);
    let role = ax::string_attribute(element, ax::ROLE_ATTRIBUTE)?.unwrap_or_default();
    if !valid_metadata(&role, MAX_LOCAL_APP_ROLE_BYTES) && !role.is_empty() {
        return Ok(Semantics {
            role: String::new(),
            label: String::new(),
            value: None,
            enabled: false,
            editable: false,
            actions: Vec::new(),
            value_digest: None,
        });
    }
    let subrole = if role == "AXSecureTextField" {
        String::new()
    } else if subrole_requires_strict_read(&role) {
        ax::string_attribute(element, ax::SUBROLE_ATTRIBUTE)
            .with_context(|| format!("Could not read AXSubrole for role {role:?}"))?
            .unwrap_or_default()
    } else {
        ax::optional_subrole(element)
            .with_context(|| format!("Could not read AXSubrole for role {role:?}"))?
            .unwrap_or_default()
    };
    let secure = is_secure_field(&role, &subrole);
    if secure {
        return Ok(Semantics {
            role: String::new(),
            label: String::new(),
            value: None,
            enabled: false,
            editable: false,
            actions: Vec::new(),
            value_digest: None,
        });
    }
    let value = ax::string_attribute(element, ax::VALUE_ATTRIBUTE)?;
    let title = ax::string_attribute(element, ax::TITLE_ATTRIBUTE)?;
    let description = ax::string_attribute(element, ax::DESCRIPTION_ATTRIBUTE)?;
    let label = title
        .into_iter()
        .chain(description)
        .find(|value| !value.is_empty())
        .or_else(|| {
            (!is_editable_role(&role))
                .then_some(value.clone())
                .flatten()
        })
        .unwrap_or_default();
    let label = bounded_label(&label);
    let value = value.filter(|value| value.len() <= crate::computer::MAX_LOCAL_APP_TEXT_BYTES);
    let value_digest = value.as_deref().map(digest);
    let editable = is_editable_role(&role)
        && value.is_some()
        && ax::is_attribute_settable(element, ax::VALUE_ATTRIBUTE).unwrap_or(false);
    let enabled = effective_enabled(
        ax::bool_attribute(element, ax::ENABLED_ATTRIBUTE)?,
        editable,
    );
    let raw_actions = ax::actions(element)?;
    let mut actions = Vec::with_capacity(MAX_LOCAL_APP_ACTIONS.min(2));
    if raw_actions.iter().any(|action| action == ax::PRESS_ACTION) {
        actions.push("press".to_owned());
    }
    if editable {
        actions.push("set_text".to_owned());
    }
    Ok(Semantics {
        role,
        label,
        value,
        enabled,
        editable,
        actions,
        value_digest,
    })
}

fn effective_enabled(reported: Option<bool>, editable: bool) -> bool {
    reported.unwrap_or(editable)
}

fn is_editable_role(role: &str) -> bool {
    matches!(role, "AXTextField" | "AXTextArea" | "AXSearchField")
}

fn subrole_requires_strict_read(role: &str) -> bool {
    role.is_empty()
        || matches!(
            role,
            "AXTextField" | "AXTextArea" | "AXSearchField" | "AXSecureTextField"
        )
}

fn is_secure_field(role: &str, subrole: &str) -> bool {
    role == "AXSecureTextField" || subrole == "AXSecureTextField"
}

fn bounded_label(value: &str) -> String {
    let sanitized: String = value
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .collect();
    if sanitized.len() <= MAX_LOCAL_APP_LABEL_BYTES {
        return sanitized;
    }
    let mut end = MAX_LOCAL_APP_LABEL_BYTES;
    while end > 0 && !sanitized.is_char_boundary(end) {
        end -= 1;
    }
    sanitized[..end].to_owned()
}

fn valid_metadata(value: &str, max_bytes: usize) -> bool {
    !value.is_empty() && value.len() <= max_bytes && !value.chars().any(char::is_control)
}

fn valid_window_title(value: &str) -> bool {
    value.len() <= MAX_LOCAL_APP_WINDOW_TITLE_BYTES && !value.chars().any(char::is_control)
}

fn digest(value: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_rejects_pid_and_launch_date_reuse() {
        let identity = Identity {
            pid: 42,
            bundle_id: "com.example.App".to_owned(),
            name: "App".to_owned(),
            launch_date: 1234.5,
        };
        let mut replaced_pid = identity.clone();
        replaced_pid.pid += 1;
        assert!(!identity.same_process(&replaced_pid));
        let mut relaunched = identity.clone();
        relaunched.launch_date += 1.0;
        assert!(!identity.same_process(&relaunched));
        assert!(identity.same_process(&identity));
    }

    #[test]
    fn secure_fields_are_detected_by_role_or_subrole() {
        assert!(is_secure_field("AXSecureTextField", ""));
        assert!(is_secure_field("AXTextField", "AXSecureTextField"));
        assert!(!is_secure_field("AXTextField", "AXTextField"));
    }

    #[test]
    fn subrole_failures_stay_strict_for_text_roles() {
        assert!(subrole_requires_strict_read(""));
        assert!(subrole_requires_strict_read("AXTextField"));
        assert!(subrole_requires_strict_read("AXTextArea"));
        assert!(subrole_requires_strict_read("AXSearchField"));
        assert!(subrole_requires_strict_read("AXSecureTextField"));
        assert!(!subrole_requires_strict_read("AXToolbar"));
        assert!(!subrole_requires_strict_read("AXButton"));
    }

    #[test]
    fn labels_are_bounded_without_discarding_multiline_shape() {
        assert_eq!(bounded_label("first\nsecond\tvalue"), "first second value");
        let value = "x".repeat(MAX_LOCAL_APP_LABEL_BYTES + 1);
        assert_eq!(bounded_label(&value).len(), MAX_LOCAL_APP_LABEL_BYTES);
    }

    #[test]
    fn untitled_windows_are_valid_but_control_titles_are_not() {
        assert!(valid_window_title(""));
        assert!(valid_window_title("Document"));
        assert!(!valid_window_title("bad\ntitle"));
    }

    #[test]
    fn missing_enabled_is_inferred_only_for_editable_controls() {
        assert!(effective_enabled(None, true));
        assert!(!effective_enabled(None, false));
        assert!(!effective_enabled(Some(false), true));
        assert!(effective_enabled(Some(true), false));
    }
}

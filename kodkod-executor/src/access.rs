//! Explicit local consent, separate from enrollment and OS permission grants.
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicU8, AtomicU64, Ordering},
};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedChoice {
    version: u8,
    binding: String,
    enabled: bool,
}

// A transport retry must not reload an old positive file after a failed save.
// This is process-local fencing, not another source of saved consent.
type FailedSaves = std::sync::Mutex<std::collections::HashSet<(PathBuf, String)>>;
fn failed_saves() -> &'static FailedSaves {
    static FAILED: std::sync::OnceLock<FailedSaves> = std::sync::OnceLock::new();
    FAILED.get_or_init(Default::default)
}

pub struct DeviceAccess {
    path: PathBuf,
    binding: String,
    pub allowed: std::sync::Arc<AtomicU8>,
    pub generation: std::sync::Arc<AtomicU64>,
    error: std::sync::Mutex<Option<String>>,
}
impl DeviceAccess {
    pub fn load(path: PathBuf, base: &str, token: &str) -> Self {
        // Exact endpoint + authenticated credential scopes both owner and enrolled
        // device (including the legacy master-token identity). Never store a token.
        let mut hash = Sha256::new();
        hash.update(b"handler-mac-tool-consent-v1\0");
        hash.update(base.as_bytes());
        hash.update([0]);
        hash.update(token.as_bytes());
        let binding = format!("{:x}", hash.finalize());
        let loaded = (|| -> Result<bool> {
            let bytes = match std::fs::read(&path) {
                Ok(bytes) => bytes,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
                Err(e) => return Err(e.into()),
            };
            let saved: SavedChoice = serde_json::from_slice(&bytes)?;
            anyhow::ensure!(saved.version == 1, "Unsupported saved choice");
            Ok(saved.enabled && saved.binding == binding)
        })();
        let blocked = failed_saves()
            .lock()
            .unwrap()
            .contains(&(path.clone(), binding.clone()));
        let enabled = !blocked && loaded.as_ref().is_ok_and(|enabled| *enabled);
        Self {
            path,
            binding,
            allowed: std::sync::Arc::new(AtomicU8::new(if enabled { 7 } else { 0 })),
            generation: std::sync::Arc::new(AtomicU64::new(0)),
            error: std::sync::Mutex::new(if blocked {
                Some(
                    "Executor tools are off: a prior save failed. Retry saving disabled access."
                        .into(),
                )
            } else {
                loaded.err().map(|_| {
                "Executor tools are off: saved choice could not be read. Disable to save a new off choice.".into()
            })
            }),
        }
    }
    pub fn enabled(&self) -> bool {
        self.allowed.load(Ordering::Acquire) != 0
    }
    pub fn error(&self) -> Option<String> {
        self.error.lock().unwrap().clone()
    }
    pub fn revoke(&self) {
        self.allowed.store(0, Ordering::Release);
        self.generation.fetch_add(1, Ordering::AcqRel);
    }
    pub fn set_enabled(&self, enabled: bool) -> Result<()> {
        self.set_enabled_with(enabled, |bytes| self.save_choice(bytes))
    }
    fn set_enabled_with(
        &self,
        enabled: bool,
        save: impl FnOnce(&[u8]) -> Result<()>,
    ) -> Result<()> {
        // Fence queued/in-flight work immediately, even if storage fails. No
        // network callback reads the disk or reinstates an earlier positive choice.
        self.revoke();
        let result = (|| -> Result<()> {
            anyhow::ensure!(
                !enabled || self.error().is_none(),
                "Save an off choice first"
            );
            let bytes = serde_json::to_vec(&SavedChoice {
                version: 1,
                binding: self.binding.clone(),
                enabled,
            })?;
            save(&bytes)
        })();
        if result.is_err() {
            failed_saves()
                .lock()
                .unwrap()
                .insert((self.path.clone(), self.binding.clone()));
            let message = "Executor tools are off now, but the choice could not be saved. Retry disabling before quitting; a previously saved choice may return after restart.";
            *self.error.lock().unwrap() = Some(message.into());
            anyhow::bail!(message);
        }
        failed_saves()
            .lock()
            .unwrap()
            .remove(&(self.path.clone(), self.binding.clone()));
        *self.error.lock().unwrap() = None;
        if enabled {
            self.allowed.store(7, Ordering::Release);
        }
        Ok(())
    }
    fn save_choice(&self, bytes: &[u8]) -> Result<()> {
        // Remove a former positive choice first. If the subsequent write
        // fails (for example disk full), restart sees no grant, not stale on.
        match std::fs::remove_file(&self.path) {
            Ok(()) => sync_directory(self.path.parent().context("Missing settings directory")?)?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
        if let Err(error) = save_private(&self.path, bytes) {
            // A directory-sync failure may follow rename. Best-effort revoke
            // that file too; never claim durable disable if storage denies it.
            let _ = std::fs::remove_file(&self.path);
            return Err(error);
        }
        Ok(())
    }
}
/// Preflight only: this never requests native OS permissions.
pub fn capabilities(
    bits: u8,
    permissions: crate::computer::DesktopPermissions,
) -> Vec<crate::protocol::DeviceCapability> {
    use crate::computer::DesktopPermissionStatus::Granted;
    use crate::protocol::DeviceCapability;
    let mut caps = Vec::new();
    if bits & 1 != 0 {
        caps.push(DeviceCapability::Commands);
    }
    if bits & 2 != 0 && permissions.accessibility == Granted {
        caps.push(DeviceCapability::AppAutomation);
    }
    if bits & 4 != 0
        && permissions.accessibility == Granted
        && permissions.screen_recording == Granted
    {
        caps.push(DeviceCapability::Desktop);
    }
    caps
}

/// One installation identity per authenticated server binding, not a permission grant.
pub fn executor_id(path: &Path) -> Result<uuid::Uuid> {
    let parent = path
        .parent()
        .context("Missing executor identity directory")?;
    std::fs::create_dir_all(parent)?;
    let mut options = std::fs::OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let lock = options.open(parent.join("executor-identity.lock"))?;
    lock.lock()?;
    match std::fs::read_to_string(path) {
        Ok(saved) => uuid::Uuid::parse_str(saved.trim()).context("Invalid executor identity"),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let id = uuid::Uuid::new_v4();
            save_private(path, id.to_string().as_bytes())?;
            Ok(id)
        }
        Err(error) => Err(error.into()),
    }
}

pub fn save_private(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let parent = path
        .parent()
        .context("Missing private settings directory")?;
    std::fs::create_dir_all(parent)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700))?;
    }
    let temporary = parent.join(format!(".{}.tmp", uuid::Uuid::new_v4()));
    let result = (|| -> Result<()> {
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        std::fs::rename(&temporary, path)?;
        sync_directory(parent)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(temporary);
    }
    result
}

/// Directory fsync is supported on Unix; Windows file flushes use File::sync_all.
pub(crate) fn sync_directory(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        std::fs::File::open(path)?.sync_all()?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn executor_identity_survives_restart_and_concurrent_initialization() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("private/executor-id");
        let threads: Vec<_> = (0..4)
            .map(|_| {
                let path = path.clone();
                std::thread::spawn(move || executor_id(&path).unwrap())
            })
            .collect();
        let ids: Vec<_> = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();
        assert!(ids.iter().all(|id| *id == ids[0]));
        assert_eq!(executor_id(&path).unwrap(), ids[0]);
        assert_ne!(
            executor_id(&dir.path().join("other/executor-id")).unwrap(),
            ids[0]
        );
    }

    #[test]
    fn explicit_choice_survives_restart_only_for_exact_binding() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("private/mac-tools.json");
        let load = |base: &str, token: &str| DeviceAccess::load(path.clone(), base, token);
        let access = load("https://server", "device-secret");
        assert!(!access.enabled());
        access.set_enabled(true).unwrap();
        assert!(load("https://server", "device-secret").enabled());
        for (base, token) in [
            ("http://server", "device-secret"),
            ("https://new", "device-secret"),
            ("https://server", "new-account-or-enrollment"),
        ] {
            assert!(!load(base, token).enabled());
        }
        let payload = std::fs::read_to_string(&path).unwrap();
        assert!(!payload.contains("device-secret"));
        assert!(!payload.contains("server"));
        access.set_enabled(false).unwrap();
        assert!(!load("https://server", "device-secret").enabled());
        std::fs::remove_file(&path).unwrap();
        assert!(!load("https://server", "device-secret").enabled());
    }
    #[test]
    fn saved_consent_does_not_replace_os_grants() {
        use crate::computer::{DesktopPermissionStatus::*, DesktopPermissions};
        use crate::protocol::DeviceCapability::*;
        assert_eq!(
            capabilities(
                7,
                DesktopPermissions {
                    accessibility: Required,
                    screen_recording: Required
                }
            ),
            vec![Commands]
        );
        assert_eq!(
            capabilities(
                7,
                DesktopPermissions {
                    accessibility: Granted,
                    screen_recording: Required
                }
            ),
            vec![Commands, AppAutomation]
        );
        assert!(
            capabilities(
                0,
                DesktopPermissions {
                    accessibility: Granted,
                    screen_recording: Granted
                }
            )
            .is_empty()
        );
    }
    #[test]
    fn transient_authority_is_not_migrated_and_late_epoch_is_fenced() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("choice");
        let access = DeviceAccess::load(path.clone(), "https://server", "secret");
        // Existing in-memory consent is not an explicit persistence operation.
        access.allowed.store(7, Ordering::Release);
        assert!(!DeviceAccess::load(path.clone(), "https://server", "secret").enabled());
        access.set_enabled(true).unwrap();
        let captured_before_poll = access.generation.load(Ordering::Acquire);
        access.set_enabled(false).unwrap();
        // The executor compares exactly this epoch before effect dispatch and
        // while an action runs. A delayed response cannot inherit later consent.
        assert_ne!(
            captured_before_poll,
            access.generation.load(Ordering::Acquire)
        );
        assert!(!access.enabled());
        access.set_enabled(true).unwrap();
        assert_ne!(
            captured_before_poll,
            access.generation.load(Ordering::Acquire)
        );
        // Closing/replacing transport revokes only transient authority.
        access.revoke();
        assert!(!access.enabled());
        assert!(DeviceAccess::load(path, "https://server", "secret").enabled());
    }
    #[test]
    fn failed_disable_cannot_regrant_on_transport_retry() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("private/choice");
        let access = DeviceAccess::load(path.clone(), "https://server", "secret");
        access.set_enabled(true).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        // Inject only the storage error, exercising the same production transition
        // with the earlier positive file deliberately still present.
        assert!(
            access
                .set_enabled_with(false, |_| anyhow::bail!("read-only storage"))
                .is_err()
        );
        let retry = DeviceAccess::load(path.clone(), "https://server", "secret");
        assert!(!retry.enabled());
        assert!(retry.set_enabled(true).is_err());
        retry.set_enabled(false).unwrap();
        assert!(!DeviceAccess::load(path, "https://server", "secret").enabled());
    }
    #[test]
    fn invalid_config_and_write_failure_fail_closed_without_retry_regrant() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("choice");
        std::fs::write(&path, "invalid").unwrap();
        let access = DeviceAccess::load(path.clone(), "https://server", "secret");
        assert!(!access.enabled());
        assert!(access.error().is_some());
        assert!(access.set_enabled(true).is_err());
        access.set_enabled(false).unwrap();
        access.set_enabled(true).unwrap();
        let epoch = access.generation.load(Ordering::Acquire);
        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir(&path).unwrap();
        assert!(access.set_enabled(false).is_err());
        assert!(!access.enabled());
        assert_ne!(epoch, access.generation.load(Ordering::Acquire));
        assert!(access.set_enabled(true).is_err());
        assert!(!access.enabled());
    }
}

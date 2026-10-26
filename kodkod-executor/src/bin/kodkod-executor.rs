//! Explicitly authorized, user-owned execution endpoint. No model runs here.
use anyhow::{Context, Result, bail};
use kodkod_executor::runner::{Executor, ExecutorConfig};
use std::path::PathBuf;

const USAGE: &str = "Usage: kodkod-executor serve --server URL --token-file FILE --state-dir DIR --allow commands[,desktop,apps] [--name NAME] [--allow-http] [--desktop-helper-socket PATH]\n       kodkod-executor capabilities [--desktop-helper-socket PATH]\n\nRuns as the current OS user. Command access grants that user's account authority.\nUse a dedicated user/VM for isolation; a working directory is not a sandbox.\nOn Linux, --desktop-helper-socket opts into an already-running Wayland helper.\nThe token file must contain an enrolled executor credential, never a model API key.";

fn parse(args: impl IntoIterator<Item = String>) -> Result<Option<ExecutorConfig>> {
    let mut args = args.into_iter();
    let Some(command) = args.next() else {
        bail!("{USAGE}");
    };
    if matches!(command.as_str(), "--help" | "-h") {
        println!("{USAGE}");
        return Ok(None);
    }
    anyhow::ensure!(command == "serve", "Unknown command. {USAGE}");
    let mut server = None;
    let mut token_file = None;
    let mut state_dir = None;
    let mut name = None;
    let mut allowed = None;
    let mut desktop_helper_socket = None;
    let mut allow_http = false;
    while let Some(flag) = args.next() {
        if flag == "--allow-http" {
            allow_http = true;
            continue;
        }
        let value = args
            .next()
            .with_context(|| format!("Missing value for {flag}"))?;
        match flag.as_str() {
            "--server" => {
                anyhow::ensure!(server.is_none(), "Duplicate --server");
                server = Some(value);
            }
            "--token-file" => {
                anyhow::ensure!(token_file.is_none(), "Duplicate --token-file");
                token_file = Some(PathBuf::from(value));
            }
            "--state-dir" => {
                anyhow::ensure!(state_dir.is_none(), "Duplicate --state-dir");
                state_dir = Some(PathBuf::from(value));
            }
            "--name" => {
                anyhow::ensure!(name.is_none(), "Duplicate --name");
                name = Some(value);
            }
            "--allow" => {
                anyhow::ensure!(allowed.is_none(), "Duplicate --allow");
                let mut mask = 0;
                for capability in value.split(',') {
                    mask |= match capability {
                        "commands" => 1,
                        "apps" => 2,
                        "desktop" => 4,
                        _ => bail!("Unknown capability: {capability}"),
                    };
                }
                allowed = Some(mask);
            }
            "--desktop-helper-socket" => {
                anyhow::ensure!(
                    desktop_helper_socket.is_none(),
                    "Duplicate --desktop-helper-socket"
                );
                desktop_helper_socket = Some(PathBuf::from(value));
            }
            _ => bail!("Unknown option: {flag}"),
        }
    }
    let token_path = token_file.context("--token-file is required")?;
    let metadata =
        std::fs::metadata(&token_path).context("Cannot read executor credential file")?;
    anyhow::ensure!(
        metadata.is_file() && metadata.len() <= 8192,
        "Invalid executor credential file"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        anyhow::ensure!(
            metadata.uid() == unsafe { libc::geteuid() }
                && metadata.permissions().mode() & 0o077 == 0,
            "Executor credential file must be owned by this user and private (chmod 600)"
        );
    }
    let token =
        std::fs::read_to_string(token_path).context("Cannot read executor credential file")?;
    let mut config = ExecutorConfig::new(
        server
            .context("--server is required")?
            .trim_end_matches('/')
            .into(),
        token.trim().into(),
        state_dir.context("--state-dir is required")?,
    );
    config.capability_mask = allowed.context("Explicit --allow capabilities are required")?;
    config.allow_insecure_http = allow_http;
    config.desktop_helper_socket = desktop_helper_socket;
    if let Some(name) = name {
        config.name = name;
    }
    config.validate()?;
    Ok(Some(config))
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.first().is_some_and(|arg| arg == "capabilities") {
        let helper_socket = if args.len() == 1 {
            None
        } else {
            anyhow::ensure!(
                args.len() == 3 && args[1] == "--desktop-helper-socket",
                "Invalid capabilities arguments. {USAGE}"
            );
            Some(PathBuf::from(&args[2]))
        };
        #[cfg(not(target_os = "linux"))]
        anyhow::ensure!(
            helper_socket.is_none(),
            "The Wayland desktop helper is supported only on Linux"
        );
        println!(
            "{}",
            serde_json::to_string(
                &kodkod_executor::runner::available_capabilities_with_helper(
                    helper_socket.as_deref(),
                )
            )?
        );
        return Ok(());
    }
    let Some(config) = parse(args)? else {
        return Ok(());
    };
    let executor = Executor::start(config);
    executor.set_enabled(true)?;
    println!("Executor running. Press Ctrl-C to disconnect.");
    tokio::select! {
        result = shutdown_signal() => { result?; },
        _ = async { while executor.is_running() { tokio::time::sleep(std::time::Duration::from_millis(250)).await; } } => {
            anyhow::bail!("{}", executor.connection_error().unwrap_or_else(|| "Executor stopped".into()));
        }
    }
    executor.shutdown();
    executor.join().await;
    Ok(())
}

async fn shutdown_signal() -> Result<()> {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        tokio::select! { result=tokio::signal::ctrl_c()=>{result?;}, _=terminate.recv()=>{} }
    }
    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c().await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn usage_does_not_enable_remote_control() {
        assert!(parse(Vec::<String>::new()).is_err());
        assert!(parse(["serve".into()]).is_err());
        assert!(parse(["bad".into()]).is_err());
    }
}

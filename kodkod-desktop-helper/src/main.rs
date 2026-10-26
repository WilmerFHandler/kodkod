//! Optional native desktop process. Headless executors never link this backend.
#[cfg(target_os = "linux")]
mod capture;
#[cfg(target_os = "linux")]
mod input;
#[cfg(target_os = "linux")]
mod portal;
#[cfg(target_os = "linux")]
mod server;

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    #[cfg(target_os = "linux")]
    {
        tokio::task::LocalSet::new()
            .run_until(server::run(std::env::args().skip(1)))
            .await
    }
    #[cfg(not(target_os = "linux"))]
    {
        anyhow::bail!("The Wayland desktop helper is supported only on Linux")
    }
}

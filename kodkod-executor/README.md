# Kodkod Executor

`kodkod-executor` is a companion library and CLI for machines that an existing
server is authorized to use. The `serve` command connects outward to that
server; it does not run an agent loop, start a server, or manage a VM. VM
integrations can call the library backends in-process instead of launching the
CLI inside a guest.

## Build and inspect

Build the CLI on the operating system that will host it. The `platform` module
selects that build's native backend:

```sh
cargo build --release -p kodkod-executor
./target/release/kodkod-executor --help
./target/release/kodkod-executor capabilities
```

On Windows, Cargo names the executable
`target/release/kodkod-executor.exe`.

`capabilities` is a preflight report. It does not request OS permissions or
create a connection, and it may omit desktop or app access until the required
environment and permissions are available.

| Build target | Commands | Desktop | Native app automation |
| --- | --- | --- | --- |
| Linux | Runs as the invoking user. | X11 requires `DISPLAY`, `scrot`, and `xdotool` in `PATH`, and uses `XAUTHORITY` when set. Wayland uses the optional user-session desktop helper with portal approval. | Unavailable. |
| macOS | Runs as the invoking user. | Requires Screen Recording and Accessibility permissions. | App listing, observation, and actions require Accessibility permission. |
| Windows | Runs as the invoking user. | The implementation uses GDI capture and `SendInput` on the primary desktop; live Windows runtime behavior is not yet verified. | Unavailable. |

The CLI never prompts for macOS permissions. Grant Screen Recording and
Accessibility access manually in System Settings for the application macOS
identifies as requesting access. The library checks those grants before
advertising the corresponding capabilities.

## Run the executor

Enrollment happens through the existing server flow. Put its executor
credential in a private file; this CLI does not mint credentials or pair a
machine. On Unix, the file must belong to the current user and have no group or
other permissions. Keep it readable only by the service user on Windows too.

Run in the foreground for a headless command endpoint:

```sh
kodkod-executor serve \
  --server https://server.example \
  --token-file "$HOME/.config/kodkod-executor/token" \
  --state-dir "$HOME/.local/state/kodkod-executor" \
  --allow commands
```

`--server`, `--token-file`, `--state-dir`, and `--allow` are required. `--allow`
accepts a comma-separated selection from `commands`, `desktop`, and `apps`; use
only the capabilities this machine should expose. `--name` optionally changes
the displayed machine name. HTTPS is required by default, with loopback HTTP
allowed; use `--allow-http` only for a trusted private HTTP endpoint.

Commands run through the target OS shell with a 60-second timeout. The executor
retains at most 64 KiB of stdout and stderr each and reports when either stream
was truncated.

The state directory must be an absolute path. It stores local consent,
executor identity, and pending receipts; an unacknowledged command receipt can
contain command output, so keep the directory private. The executor keeps the
server-issued credential out of that directory.

### Linux systemd user service

Save this as `~/.config/systemd/user/kodkod-executor.service`, replacing the
paths and server URL. The service manager runs it as your normal user.

```ini
[Unit]
Description=Kodkod Executor
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=%h/.local/bin/kodkod-executor serve --server https://server.example --token-file %h/.config/kodkod-executor/token --state-dir %h/.local/state/kodkod-executor --allow commands
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

```sh
systemctl --user daemon-reload
systemctl --user enable --now kodkod-executor.service
```

This headless example grants command access only. For Linux desktop access, run
inside the X11 user session with `DISPLAY` and optional `XAUTHORITY` set, and
with `scrot` and `xdotool` installed; do not set `WAYLAND_DISPLAY`.

For Wayland, build and start the optional
[`kodkod-desktop-helper`](../kodkod-desktop-helper/README.md) inside the graphical
session, approve sharing there, and pass its private socket path with
`--desktop-helper-socket PATH`. The headless executable does not link PipeWire
or request graphical permission. Status checks discover an already-approved
helper; they never start one or open a permission dialog.

### macOS LaunchAgent

Save a plist like this as
`~/Library/LaunchAgents/org.kodkod.executor.plist`, replacing paths and the
server URL. It runs at login in the current user's session.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>org.kodkod.executor</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/alice/.local/bin/kodkod-executor</string>
    <string>serve</string>
    <string>--server</string><string>https://server.example</string>
    <string>--token-file</string><string>/Users/alice/.config/kodkod-executor/token</string>
    <string>--state-dir</string><string>/Users/alice/Library/Application Support/kodkod-executor</string>
    <string>--allow</string><string>commands</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
</dict>
</plist>
```

```sh
launchctl bootstrap gui/$(id -u) "$HOME/Library/LaunchAgents/org.kodkod.executor.plist"
```

Add `desktop` or `apps` to `--allow` only after granting the required macOS
permissions manually.

## Trust and protocol

Commands run with the permissions of the account that started the executor.
The requested working directory is not a sandbox, and the executor does not
elevate privileges. Use an unprivileged account or an isolated VM when command
access needs a stronger boundary.

The executor uses protocol version 1 over outbound HTTP requests to
`/v1/device-executor/connect`, `/v1/device-executor/poll`, and
`/v1/device-executor/receipt`, with a bearer credential and the
`X-Kodkod-Executor-Protocol: 1` header. The executor sends its current local
grant generation while polling; returned operations carry opaque attempt and
attachment identifiers. A compatible server must already implement this
protocol; the executor adds no project or agent-session API of its own.

## Checks

The executor workflow runs formatting, package tests, and Clippy on Ubuntu,
macOS, and Windows. Tests use local fixtures and do not open a desktop, make
network calls, or require credentials.

The Linux helper has a separate CI job that installs its native development
libraries and tests protocol, input planning, and capture conversion without a
desktop session. Real compositor compatibility requires separate live tests.

```sh
cargo fmt --package kodkod-executor -- --check
cargo test -p kodkod-executor
cargo clippy -p kodkod-executor --all-targets -- -D warnings
```

## Read-only viewing and scrolling

Desktop-enabled executors poll a separate view channel. An active viewer starts
one latest-frame JPEG publisher, shared across viewers; no frames are stored in
the action journal or model context. Frames are bounded to 1600 pixels wide and
1 MiB, with one upload in flight. Losing the connection or local desktop grant
stops publication. The server owns viewer authorization and expiry.

`computer_input` accepts `scroll` with an anchor `x`/`y` in the original
screenshot and bounded `delta_x`/`delta_y`. Positive values move content right or
down. The anchor must be in the fresh observed frame. Backends translate these
units to native scrolling; wheel-only systems approximate them in wheel steps.

The outbound connection announces `X-Kodkod-Executor-Protocol: 2` without changing
strict JSON contracts. Version 2 supports scroll actions; services treating a
missing header as version 1 can reject unsupported input before dispatch. This
version describes wire support and grants no additional OS permissions.

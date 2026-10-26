# Wayland desktop helper

This optional Linux process runs inside the user's graphical session. The
headless executor keeps command execution and its server connection; it reaches
the desktop through a private, same-user Unix socket. Installing or starting the
headless executor never opens a sharing dialog.

Build the helper with the PipeWire development headers, pkg-config and libclang
available:

```sh
cargo build --release -p kodkod-desktop-helper --features portal
```

Start it from the Wayland desktop session:

```sh
kodkod-desktop-helper serve --request-access
```

The desktop's RemoteDesktop/ScreenCast portal asks which screen to share and
whether input is allowed. Approval lasts for this helper session; it is not
silently restored after a restart. Running without `--request-access` exposes
status only. `status` never requests permission:

```sh
kodkod-desktop-helper status
```

The executor discovers `$XDG_RUNTIME_DIR/kodkod-desktop-helper/desktop.sock`
inside a Wayland session, or accepts an explicit `--desktop-helper-socket`.
Include `desktop` in its local `--allow` list. Both the executor's grant and the desktop's portal
approval are required. A headless system service cannot borrow another user's
graphical session.

The helper captures only the portal-granted PipeWire stream and sends input
through EIS. It requires an unambiguous mapping between that stream and an
active absolute-pointer region. It rejects stale observations and stops input
when the granted source or its geometry changes. Restart sharing after such a
change. Text entry prefers EIS Text when available. Otherwise it preflights the
entire string against the compositor-provided XKB keymap and current modifiers,
then sends representable keys through EIS. Unsupported characters or ambiguous
modifier state fail before any text is inserted; there is no clipboard fallback.
Pointer scrolling uses the granted EIS Scroll capability and stops the gesture
explicitly. Check `supported_inputs` before choosing an action.

The socket and its parent directory must belong to the invoking user and be
private. Only one executor connection can control the desktop at a time. Losing
that connection releases its observation lease; uncertain actions are never
automatically replayed.

Read-only `view_frame` requests use a separate capture channel and do not acquire
or replace the input observation lease. JPEG encoding is cached per captured
frame. The outbound executor publishes frames only while the server has an
active owner-authorized viewer; closing the last viewer stops publication.

VM defaults remain X11. Moving guests to Wayland also requires migrating their
persistent input connection and private-entry path; starting this helper does
not silently change that boundary.

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

Connect the executor with `--desktop-helper-socket` pointing to
`$XDG_RUNTIME_DIR/kodkod-desktop-helper/desktop.sock`, and include `desktop` in
its local `--allow` list. Both the executor's grant and the desktop's portal
approval are required. A headless system service cannot borrow another user's
graphical session.

The helper captures only the portal-granted PipeWire stream and sends input
through EIS. It requires an unambiguous mapping between that stream and an
active absolute-pointer region. It rejects stale observations and stops input
when the granted source or its geometry changes. Restart sharing after such a
change. Arbitrary text entry requires the compositor's EIS Text interface;
unsupported input fails explicitly instead of guessing a keyboard layout.

The socket and its parent directory must belong to the invoking user and be
private. Only one executor connection can control the desktop at a time. Losing
that connection releases its observation lease; uncertain actions are never
automatically replayed.

This backend does not change VM desktop defaults or implement a continuous
video stream. Those integrations need separate live verification.

On GNOME 50, live checks verified portal-approved capture, pointer clicks and
keyboard focus/activation against a disposable GTK application. That compositor
did not negotiate EI Text, so arbitrary UTF-8 typing and letter shortcuts remain
unavailable there. Check `supported_inputs` on the returned screenshot before
choosing an action. This is a capability-limited backend, not grounds to switch
existing X11 VMs to Wayland by default.

# Running Listen on your iPhone (free, no App Store)

The backend is already live on Railway and the app is pre-pointed at it, so
the app will work over **any** network (Wi-Fi or cellular) with no LAN setup.

- **Live server:** `https://listen-psl-backend-production.up.railway.app`
- **Health check (open in phone browser):** `.../healthz` → `{"ok":true}`
- **WebSocket the app uses:** `wss://listen-psl-backend-production.up.railway.app/ws/translate`

## One-time iPhone setup

1. **Enable Developer Mode** (iOS 16+): Settings → Privacy & Security →
   Developer Mode → On → restart the phone.
2. **Connect via USB** for the first install (most reliable). Tap **Trust**
   on the phone when prompted.

## Build & run from Xcode

1. Open the **workspace** (not the project):
   ```
   open flutter_app/ios/Runner.xcworkspace
   ```
2. First build only: Xcode will prompt to download the **iOS SDK
   (~7–8 GB)** — this is required to build for any device and is a one-time
   download. Let it finish.
3. Select the **Runner** target → **Signing & Capabilities**:
   - **Team:** click the dropdown → **Add an Account**, sign in with your
     free Apple ID, then select it as the team.
   - Leave signing on **Automatic** — Xcode generates a free provisioning
     profile.
   - If Xcode says the bundle ID `com.listen.psl.flutterApp` is
     unavailable, change it to something unique, e.g.
     `com.<yourname>.listenpsl`.
4. Pick your iPhone in the run-destination dropdown (top toolbar) → press ▶.
5. First launch: on the phone, go Settings → General → **VPN & Device
   Management** → tap your Apple ID → **Trust**. Re-launch the app.

## Command-line alternative (after signing is set once in Xcode)

```bash
cd flutter_app
flutter run --release            # uses the baked-in live server URL
```

## Running wirelessly (after the first wired run)

In Xcode → Window → **Devices and Simulators** → select your iPhone →
check **Connect via network**. Then you can unplug and run over Wi-Fi
(both devices on the same network).

## Free-account limitation

The free provisioning profile **expires after 7 days** — just re-run from
Xcode to refresh it. This is an Apple restriction on free Apple IDs, not a
project issue.

## If something looks wrong on-device

- **Camera preview / recognition looks rotated or mirrored:** the JPEG
  rotation mapping in `ios/Runner/YuvJpegPlugin.swift`
  (`orientationForDegrees`) is the thing to adjust — swap `.right`/`.left`.
  It's the one piece that can only be verified on a real device.
- **"Inference server unreachable":** open the `/healthz` URL above in the
  phone's browser. If that works but the app doesn't, check Profile →
  Server URL is the `wss://` Railway URL.
- **No suggestions appear:** they're a soft feature; the translate pipeline
  still works without them.

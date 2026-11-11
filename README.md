# Clash Bot — NeuroML on the phone

The NeuroML brain from `HASTY-CR`/`game-website`, packaged so the **phone plays
itself**: the app runs the policy, the spiking nucleus, the referee, the digital
twin and the dashboard *on the device*. No PC, no ADB, no Wi-Fi pairing.

```
phone screen ──(MediaProjection)──▶ vision ──▶ brain ──▶ taps (Accessibility)
                                      │
                                      └──▶ dashboard http://127.0.0.1:8765
```

## What runs where

| Piece | Where | Notes |
|---|---|---|
| `bot/` | Python, or inside the APK (Chaquopy) | bot loop, vision, policy, spiking core, referee, twin |
| `android/` | Gradle app | services + WebView dashboard, same page as the desktop localhost |
| `public/` | Vercel | landing page + download links |

The dashboard is literally the same `dashboard.py` page as on the desktop, served
on `127.0.0.1:8765` by the `:brain` process and shown in the app's WebView.

**One source of truth.** `bot/` is the only copy of the brain and the dashboard.
`android/app/build.gradle` copies it in at build time (there is no vendored
Python under `android/`), and the APK workflow diffs the built app against `bot/`
so the phone cannot ship a different page than `python bot/dashboard.py` serves
on the computer:

```
bot/dashboard.py  ==  android/app/src/main/python/dashboard.py  (in the APK)
```

## Start behaviour

Opening the app starts the brain (auto-start defaults ON):

1. one-time screen-capture consent (Android requires this per session — the only
   tap you make),
2. the tap accessibility service, enabled once in Settings,
3. the brain loop + dashboard, then the app opens the live dashboard.

Turn `auto_start` off in the app's preferences to go back to the manual toggle.

## Building the APK

CI builds it; there is no local Android SDK requirement:

```
Actions → build-apk → NeuroML-Android-APK          (artifact)
https://github.com/<owner>/Clash-Bot/releases/download/android/NeuroML-Android.apk
```

Locally, with a JDK 17 + Android SDK 33 + Python 3.11:

```
cd android
gradle assembleRelease -PchaquopyBuildPython=$(which python)
```

The debug-signed keystore is committed on purpose so every build shares one
signature and updates install in place over the previous version.

## Tactical layer (`bot/tactics.py`)

The network proposes a placement; the tactical layer vetoes the two errors the
reward ledger kept flagging, both decidable without learning:

- **wrong-lane defence** — a live push pulls the answer to that lane,
- **empty-lane conversion** — at 7+ elixir with no threat, `center_back` and
  `*_defend` become a push on the weaker enemy tower,
- plus "never play a card you cannot afford".

It is pure functions, torch-free, and returns a log line that shows up in the
dashboard's decision provenance.

```
cd bot && py -3.12 test_tactics.py
```

## Honest limits

- Vision is pixel-scanning, not object detection.
- Automating Clash Royale violates Supercell's ToS — use an account you accept
  losing.
- The APK build is CI-only in this repo; nothing here was compiled on a machine
  without the Android toolchain.

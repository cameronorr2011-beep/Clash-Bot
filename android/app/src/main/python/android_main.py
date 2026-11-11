# android_main.py — on-phone entry point for the FULL NeuroML brain.
#
# The player phone here IS the phone playing Clash Royale: no computer, no ADB,
# no WiFi pairing — the brain sees the screen via MediaProjection
# (CaptureService) and taps via the accessibility service
# (TapAccessibilityService). All learning, the coach (through the maintainer's
# zero-key mentor proxy) and the federated package-exchange loop run exactly as
# on desktop.
import os
import threading
import time
from types import SimpleNamespace

os.environ.setdefault("NEUROML_PHONE_LOCAL", "1")
os.environ.setdefault("NEUROML_DEVICE", "local")

from com.neuroml.app import CaptureService, TapAccessibilityService, BrainService

BRAIN_STATE = {"started": False, "result": ""}
# phase: starting -> waiting_capture -> running ; detail: human-readable failure
_STATE = {"phase": "starting", "detail": ""}


def _log(msg):
    print(f"[android] {msg}", flush=True)


def _fail(msg):
    """Honest OFF + persist the reason (BrainService runs in THIS process, and
    reportError mirrors it into the shared \"nml\" prefs the UI process reads)."""
    BRAIN_STATE["result"] = msg
    _STATE["phase"] = "stopped"
    _STATE["detail"] = msg
    _log(msg)
    if _CTX is not None:
        BrainService.reportError(_CTX, msg)
    return msg


_CTX = None  # Android Context, passed in by BrainService


def run(files_dir, ctx):
    """Called by BrainService on its own thread: starts the dashboard server
    FIRST (so the dashboard page always answers), waits for the capture grant,
    then runs the brain loop."""
    global _CTX
    _CTX = ctx
    if _STATE["phase"] == "running":
        _log("run() called again while already running — reusing the live brain")
        return "running"   # toggling twice must not spawn a second loop/port bind
    os.chdir(files_dir)  # relative paths (bot.log, cr_bot/, cards.json, *.pt)
    BRAIN_STATE["started"] = True
    _STATE["phase"] = "waiting_capture"
    _STATE["detail"] = ""

    # ---- dashboard server in-process: start EARLY --------------------------
    import dashboard

    def _android_screencap(retries=1):
        if _STATE["phase"] != "running":
            raise RuntimeError(
                f"brain is {_STATE['phase']} — {_STATE['detail'] or 'screen capture not granted yet'}")
        if CaptureService.screenshotPng(
                os.path.join(files_dir, "cr_bot", "latest.png"), 5.0):
            from PIL import Image
            return Image.open(
                open(os.path.join(files_dir, "cr_bot", "latest.png"), "rb")).convert("RGB")
        raise RuntimeError("no frame yet (screen capture did not deliver)")

    dashboard._screencap = _android_screencap

    def _android_health():
        if _STATE["phase"] == "running":
            return {"bot": True, "device": "this phone (screen capture)",
                    "mem": "brain data on device"}
        d = {"bot": False, "device": "this phone", "mem": "brain data on device"}
        if _STATE["detail"]:
            d["detail"] = _STATE["detail"]
        return d

    def _lan_ip():
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return ""

    dashboard.LAN_URL = f"http://{_lan_ip()}:8765" if _lan_ip() else ""

    def _android_health():
        if _STATE["phase"] == "running":
            return {"bot": True, "device": "this phone (screen capture)",
                    "mem": "brain data on device",
                    "lan": getattr(dashboard, "LAN_URL", "")}
        d = {"bot": False, "device": "this phone", "mem": "brain data on device",
             "lan": getattr(dashboard, "LAN_URL", "")}
        if _STATE["detail"]:
            d["detail"] = _STATE["detail"]
        return d

    dashboard.health = _android_health

    def _serve():
        try:
            threading.Thread(target=dashboard._auto_refresher, daemon=True).start()
            from http.server import ThreadingHTTPServer
            try:
                # 0.0.0.0 so the dashboard is ALSO reachable from any PC on the
                # same Wi-Fi — the phone screen is tiny; the user asked to view
                # it on the desktop too ("give localhost too").
                httpd = ThreadingHTTPServer(("0.0.0.0", dashboard.PORT), dashboard.H)
            except OSError as e:
                if getattr(e, "errno", None) == 98:  # EADDRINUSE
                    _log("dashboard already serving in this process — reusing it")
                    return
                raise
            httpd.serve_forever()
        except Exception as e:
            _log(f"dashboard problem: {e!r}")
            _fail(f"dashboard server failed: {e!r}")

    threading.Thread(target=_serve, daemon=True, name="neuroml-dashboard").start()
    lan = getattr(dashboard, "LAN_URL", "")
    _log("dashboard up: " + (f"{lan} (from any PC on the Wi-Fi)" if lan else "")
         + " · http://127.0.0.1:8765 on this phone")

    # ---- wait for the screen-capture grant ---------------------------------
    end = time.time() + 90.0
    while time.time() < end:
        if CaptureService.ready():
            break
        if _STATE["phase"] == "stopped":  # dashboard thread failed → don't hang
            return BRAIN_STATE["result"]
        time.sleep(0.5)
    else:
        return _fail("screen capture not granted — toggle BRAIN OFF and ON, "
                     "then pick Start Now (or Entire screen) in the dialog")

    # ---- import the full brain (can take ~30-60 s on first run) ------------
    # v27: point the NEW data modules at filesDir BEFORE importing the brain.
    # card_index / deck_state compute their default data paths from __file__,
    # which in Chaquopy resolves INSIDE the APK (not a readable filesystem
    # location); their data now lives in filesDir like every other json
    # (cards_v2.json etc. extracted by MainActivity.copyAssets).
    import card_index
    import deck_state
    card_index.DB_PATH = os.path.join(files_dir, "cards_v2.json")
    card_index._DB = None                    # force a re-load from filesDir
    deck_state.ACTIVE_PATH = os.path.join(files_dir, "active_deck.json")
    deck_state.RECO_PATH = os.path.join(files_dir, "recommended_deck.json")

    # ---- v30: the phone defaults to the FULL 270M 'advanced' brain -------
    # (owner request: modern phones have the storage for it). But we PROVE the
    # OS will actually grant the native memory instead of trusting headline
    # RAM: allocate the policy-sized ballast first; if the system refuses,
    # step down LOUDLY (advanced -> extended -> standard) — the chosen profile
    # is logged either way, and the dashboard can never show a different brain
    # than the one actually running.
    def _preflight_profile():
        import torch  # already shipped in the APK — import is warm by now
        for prof, mb in (("advanced", 1100), ("extended", 400), ("standard", 200)):
            try:
                _ballast = torch.empty(int(mb) * 1024 * 1024 // 4, dtype=torch.float32)
                del _ballast
                _log(f"RAM pre-flight: {mb} MB granted -> requesting '{prof}' profile")
                return prof
            except Exception as pf_exc:
                _log(f"RAM pre-flight: {mb} MB refused ({pf_exc}) — stepping down")
        return "standard"

    os.environ.setdefault("QNET_PROFILE", _preflight_profile())
    _log(f"selected brain profile: {os.environ.get('QNET_PROFILE')} "
         f"(env override still wins if you set NEUROML_PROFILE yourself)")

    _log("importing full brain (torch DQN, spiking nucleus, referee, coach)")
    import bot_ml

    # ---- bridge phone I/O over Chaquopy ------------------------------------
    cfg = bot_ml.CFG
    cfg_w = float(cfg.get("screen", {}).get("w") or 1080)
    cfg_h = float(cfg.get("screen", {}).get("h") or 2340)
    real_w = float(CaptureService.screenW() or cfg_w)
    real_h = float(CaptureService.screenH() or cfg_h)
    sx = (real_w / cfg_w) if cfg_w else 1.0
    sy = (real_h / cfg_h) if cfg_h else 1.0

    shot_path = os.path.join(files_dir, "cr_bot", "latest.png")

    def _capture_async():
        threading.Thread(
            target=lambda: CaptureService.screenshotPng(shot_path, 5.0),
            daemon=True).start()

    def _bridge_screenshot():
        if CaptureService.screenshotPng(shot_path, 5.0):
            return shot_path
        return None

    def _bridge_tap(x, y):
        ok = TapAccessibilityService.tap(int(round(x * sx)), int(round(y * sy)))
        time.sleep(0.4)
        return ok

    def _bridge_sh(*args, timeout=25, **kw):
        # on the desktop this ran `adb -s DEV ...`; on the phone both directions
        # (screencap + input tap) are native bridge calls. One probe MUST work
        # for real: `shell dumpsys window` — bot_ml.foreground_app() uses it for
        # the "user opened Clash" gate. Apps can't run dumpsys, but the tap
        # accessibility service (in THIS process) sees every
        # TYPE_WINDOW_STATE_CHANGED broadcast, so we synthesize the exact two
        # lines foreground_app() parses from the service's tracked package.
        joined = " ".join(str(a) for a in args)
        if "screencap" in joined:
            # The desktop captures through `adb exec-out screencap -p`; the
            # on-device brain makes the SAME call, so it must go through
            # MediaProjection here and return real PNG bytes. Before v29 this
            # bridge returned EMPTY stdout (returncode 0) for every sh-screencap:
            # consumers like the post-result play-again transition loop saw
            # nothing and silently skipped their logic.
            try:
                if CaptureService.screenshotPng(shot_path, 5.0):
                    with open(shot_path, "rb") as fh:
                        return SimpleNamespace(stdout=fh.read(), stderr=b"", returncode=0)
            except Exception:
                pass
            return SimpleNamespace(stdout=b"", stderr=b"", returncode=1)
        if "dumpsys" in joined and "window" in joined:
            pkg = (TapAccessibilityService.foregroundPackage() or "").strip()
            if pkg:
                fake = (f"  mCurrentFocus=Window{{3a1b2c u0 {pkg}/com.activity}}\n"
                        f"  mFocusedApp=ActivityRecord{{1d2e3f u0 {pkg}/com.activity t42}}\n")
                return SimpleNamespace(stdout=fake.encode(), stderr=b"", returncode=0)
            return SimpleNamespace(stdout=b"", stderr=b"", returncode=1)  # "unknown"
        _log(f"sh(neutralized on device): {args!r}")
        return SimpleNamespace(stdout=b"", stderr=b"", returncode=0)

    bot_ml.sh = _bridge_sh
    bot_ml.screenshot = _bridge_screenshot
    bot_ml.screenshot_async = _capture_async
    bot_ml.tap = _bridge_tap
    _log(f"bridge ready {real_w:.0f}x{real_h:.0f} (scale {sx:.3f},{sy:.3f})")
    dashboard._screencap = _android_screencap  # now safely usable

    # ---- the brain loop ----------------------------------------------------
    def _brain():
        try:
            _log("bot main loop starting")
            bot_ml.main()
        except Exception as e:
            _log(f"brain crashed: {e!r}")
            _fail(f"brain: {e!r}")

    threading.Thread(target=_brain, daemon=True, name="neuroml-runner").start()
    _STATE["phase"] = "running"
    _STATE["detail"] = ""
    BRAIN_STATE["result"] = "running"
    BrainService.pythonCrashed = None
    _log("brain + dashboard running (playing the account signed in on this phone)")

    # ---- WATCHDOG + HEARTBEAT (v27 fixes: toggle desync after leaving screen)
    # The UI process previously trusted a SharedPreferences flag that could go
    # stale when Android reclaimed the :brain process or the MediaProjection
    # died with the screen off — the toggle then showed a lie and OFF felt
    # broken. Now Python writes a live heartbeat the UI can trust, and a
    # genuinely dead capture is reported honestly instead of zombifying.
    def _watchdog():
        alive_path = os.path.join(files_dir, "cr_bot", "brain_alive")
        dead_since = None
        while True:
            if _STATE["phase"] != "running":
                break
            try:
                open(alive_path, "w").write(str(time.time()))
                live = CaptureService.ready()
                if not live and CaptureService.screenW() != 0:
                    # projection existed once — so it DIED (screen off / revoke)
                    dead_since = dead_since or time.time()
                    if time.time() - dead_since > 20.0:
                        _fail("screen capture died (screen off or Android revoked "
                              "it) — the toggle was running a zombie; tap 🧠 OFF, "
                              "then ON to restart cleanly")
                        break
                else:
                    dead_since = None
            except Exception:
                pass
            time.sleep(3.0)
        try:
            open(os.path.join(files_dir, "cr_bot", "brain_alive"), "w").close()
        except Exception:
            pass

    threading.Thread(target=_watchdog, daemon=True, name="neuroml-watchdog").start()
    return "running"

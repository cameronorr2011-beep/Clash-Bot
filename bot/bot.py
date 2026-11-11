"""Clash Royale bot v6 — autonomous session player.

Fixes over v5:
  1. Robust scene classifier: battle = elixir bar strip AND dark card tray
     together; known overlay screens detected explicitly; everything else
     = unknown (BACK key, with hard limits so it neverstacks to quit dialog).
  2. Post-match / reward screens: taps center OK spots, never BACK mashing.
  3. BACK key rate-limited: max 2 presses per stuck screen, then wait.
  4. Anti-quit guard: if the quit dialog ever appears, tap Cancel (red button).
  5. Strategy: elixir read + lane threat + cheap-defend / push-at-9 logic
     (unchanged from v4, which verified well on dry runs).
All taps stay y<1985 (no nav bar) except known button coordinates.
"""
import itertools
import json
import os
import subprocess
import time

CFG = json.load(open("bot_config.json"))
DEV = CFG["device"]
SHOT_DIR = "cr_bot"
os.makedirs(SHOT_DIR, exist_ok=True)
LOG = open("bot.log", "a", buffering=1)
CARDS_DB = json.load(open("cards.json"))


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    LOG.write(line + "\n")
    print(line, flush=True)


def sh(args, timeout=25):
    return subprocess.run(
        ["adb", "-s", DEV] + args, capture_output=True, timeout=timeout
    )


def screenshot():
    out = os.path.join(SHOT_DIR, "latest.png")
    result = sh(["exec-out", "screencap", "-p"])
    with open(out, "wb") as f:
        f.write(result.stdout)
    return out


def tap(x, y):
    sh(["shell", "input", "tap", str(x), str(y)])
    time.sleep(0.4)


def classify(png):
    from PIL import Image

    img = Image.open(png).convert("RGB")
    # probe 1: elixir bar magenta strip — scan rows; require a WIDE magenta run
    purple = 0
    for y in range(2200, 2340, 8):
        right = ELIXIR_X0
        for x in range(ELIXIR_X0, 700, 10):
            r, g, b = img.getpixel((x, y))
            if r > 90 and g < 105 and b > 150:
                right = x
        if right - ELIXIR_X0 > 300:
            purple += 1
    # probe 2: dark navy card tray band
    tray = [img.getpixel((x, y)) for y in range(2050, 2160, 20) for x in range(320, 940, 60)]
    navy = sum(1 for r, g, b in tray if b > 70 and b > r + 30 and r < 80)
    if purple >= 5 and navy >= 8:
        return "battle"
    # quit dialog: red Cancel + blue OK buttons side by side at y~1480
    row = [img.getpixel((x, 1480)) for x in range(250, 880, 30)]
    red = sum(1 for r, g, b in row if r > 170 and r > g + 70 and r > b + 70)
    blue = sum(1 for r, g, b in row if b > 170 and b > r + 40 and g > 110)
    if red >= 3 and blue >= 3:
        return "quit_dialog"
    # result screen: blue ribbon band across the middle
    band = [img.getpixel((x, y)) for y in (1400, 1440) for x in range(240, 880, 60)]
    blue_rib = sum(1 for r, g, b in band if b > 140 and b > r + 40)
    if blue_rib >= 8:
        return "result"
    # lobby: big orange Battle button (y 1850..2000, x 300..800)
    btn = [img.getpixel((x, y)) for y in (1850, 1900, 1950, 2000) for x in range(320, 800, 40)]
    orange = sum(1 for r, g, b in btn if r > 200 and 110 < g < 200 and b < 100)
    if orange >= 20:
        return "lobby"
    return "unknown"


CARDS = [(332, 2031), (543, 2031), (737, 2031), (931, 2031)]
FIELD_SPOTS = {
    "left_defend": (337, 1250),
    "right_defend": (742, 1250),
    "center_back": (540, 1300),
    "bridge_left": (220, 980),
    "bridge_right": (860, 980),
}
BATTLE_BUTTON = (540, 1876)
CANCEL_BUTTON = (365, 1480)   # red Cancel on quit dialog (1080x2340)
OK_CENTER = (540, 2050)       # OK button row on result/reward screens


ELIXIR_X0 = 228
ELIXIR_X1 = 948


def elixir_count(png):
    from PIL import Image

    img = Image.open(png).convert("RGB")
    best_fill = 0.0
    # find the row with the widest magenta run (bar position shifts with UI)
    for y in range(2200, 2340, 6):
        right = ELIXIR_X0
        for x in range(ELIXIR_X0, ELIXIR_X1, 4):
            r, g, b = img.getpixel((x, y))
            if r > 90 and g < 105 and b > 150:
                right = x
        fill = (right - ELIXIR_X0) / float(ELIXIR_X1 - ELIXIR_X0)
        if fill > best_fill:
            best_fill = fill
    if best_fill < 0.05:
        return 0
    return round(best_fill * 10) if best_fill < 0.95 else 10


def threat_lane(png):
    from PIL import Image
    import statistics

    img = Image.open(png).convert("RGB")
    def variance(box):
        region = list(img.crop(box).getdata())
        rs = [p[0] for p in region]
        gs = [p[1] for p in region]
        return statistics.pstdev(rs) + statistics.pstdev(gs)
    left_var = variance((100, 1500, 500, 1900))
    right_var = variance((580, 1500, 980, 1900))
    if left_var > right_var * 1.3 and left_var > 30:
        return "left"
    if right_var > left_var * 1.3 and right_var > 30:
        return "right"
    return None


DECK_ROTATION = ["musketeer", "knight", "archers", "giant"]


def pick_card(elixir, lane):
    affordable = [n for n in DECK_ROTATION if CARDS_DB[n]["cost"] <= elixir]
    if not affordable:
        return None
    if lane:
        affordable.sort(key=lambda n: CARDS_DB[n]["cost"])
    else:
        affordable.sort(key=lambda n: -CARDS_DB[n]["cost"])
    return affordable[0]


def build_deck():
    roles = {}
    for name, info in CARDS_DB.items():
        roles.setdefault(info["role"], []).append(name)
    deck = []
    for role, count in ("win_condition", 1), ("tank", 1), ("support", 3), ("spell", 2), ("building", 1):
        pool = roles.get(role, [])
        deck.extend(pool[:count])
    return deck[:8]


def play_card(slot, spot):
    x, y = CARDS[slot % len(CARDS)]
    sh(["shell", "input", "swipe", str(x), str(y), str(spot[0]), str(spot[1]), "350"])
    time.sleep(0.6)


def main():
    log(f"bot v6 started on {DEV}")
    slot_cycle = itertools.cycle(range(4))
    push_side_cycle = itertools.cycle(["bridge_left", "bridge_right"])
    just_queued = False
    back_streak = 0
    stuck_since = None
    while True:
        try:
            png = screenshot()
            state = classify(png)
            now = time.time()
            if stuck_since and now - stuck_since > 90:
                log("stuck >90s — restarting app")
                sh(["shell", "am", "force-stop", "com.supercell.clashroyale"])
                time.sleep(2)
                sh(["shell", "monkey", "-p", "com.supercell.clashroyale",
                    "-c", "android.intent.category.LAUNCHER", "1"])
                time.sleep(25)
                stuck_since = None
                back_streak = 0
                just_queued = False
                continue
            if state == "quit_dialog":
                log("quit dialog -> Cancel")
                tap(*CANCEL_BUTTON)
                back_streak = 0
                stuck_since = stuck_since or now
                continue
            elif state == "lobby":
                stuck_since = None
                back_streak = 0
                if not just_queued:
                    # trophy-road claim bubble appears above-left of the pass banner
                    log("lobby -> tapping claim bubble + battle button")
                    tap(700, 1420)  # CLAIM REWARD bubble (harmless if absent)
                    time.sleep(1.5)
                    tap(*BATTLE_BUTTON)
                    log("tapped Battle button")
                    just_queued = True
                    time.sleep(6)
                else:
                    time.sleep(6)
            if state == "battle":
                stuck_since = None
                back_streak = 0
                just_queued = False
                elixir = elixir_count(png)
                lane = threat_lane(png)
                if lane and elixir >= 3:
                    spot = FIELD_SPOTS[f"{lane}_defend"]
                    slot = next(slot_cycle)
                    log(f"elixir={elixir} threat={lane} -> defend slot {slot}")
                    play_card(slot, spot)
                    time.sleep(3.5)
                elif elixir >= 9:
                    spot_name = next(push_side_cycle)
                    spot = FIELD_SPOTS[spot_name]
                    slot = next(slot_cycle)
                    log(f"elixir={elixir} -> full push slot {slot} @ {spot_name}")
                    play_card(slot, spot)
                    time.sleep(4.0)
                else:
                    time.sleep(2.0)
            elif state == "result":
                stuck_since = stuck_since or now
                log("result/reward screen -> tapping OK / continue")
                tap(*OK_CENTER)
                time.sleep(2)
            else:  # unknown
                stuck_since = stuck_since or now
                if back_streak < 2:
                    log(f"unknown -> BACK ({back_streak + 1}/2)")
                    sh(["shell", "input", "keyevent", "KEYCODE_BACK"])
                    back_streak += 1
                    time.sleep(3)
                else:
                    # unknown+stuck: try the lobby battle button blindly once,
                    # then wait — never BACK-mashes into the quit dialog
                    log("unknown, trying battle button blind")
                    tap(*BATTLE_BUTTON)
                    time.sleep(8)
                    just_queued = True
        except Exception as e:
            import traceback
            log(f"loop error: {type(e).__name__}: {e!r} {traceback.format_exc(limit=2)}")
            time.sleep(5)


if __name__ == "__main__":
    import sys
    if "--build-deck" in sys.argv:
        print(json.dumps(build_deck(), indent=2))
    else:
        main()

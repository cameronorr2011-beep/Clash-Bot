"""NeuroML brain fetcher.

Downloads current pretrained files from this project's GitHub Releases into the
app folder so a fresh install doesn't have to rebuild a brain from zero:

    ml_policy.pt     DQN cortex checkpoint (policy + target weights)
    spiking_brain.json  LIF/STDP nucleus + affect state (small)

Both files are OPTIONAL. When neither is published (or the machine is offline)
you get a fresh brain that trains itself live — the app is fully playable and
saves its own copies the first time a match ends.

Features on your disk are always kept: if a file already exists it is left
alone, never overwritten, so this can run every install safely.
"""
import os
import sys
import urllib.request

REPO = "cameronorr2011-beep/game-website"
FILES = ["ml_policy.pt", "spiking_brain.json"]
HERE = os.path.dirname(os.path.abspath(__file__))


def fetch(remote_name: str) -> str:
    dest = os.path.join(HERE, remote_name)
    if os.path.exists(dest):
        return f"   [OK]      {remote_name} kept as-is (already present)"
    url = f"https://github.com/{REPO}/releases/latest/download/{remote_name}"
    tmp = dest + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)
        with open(tmp, "rb") as f:
            head = f.read(2)
        if head != b"PK" and remote_name.endswith(".pt"):
            raise ValueError("not a valid checkpoint file")
        os.replace(tmp, dest)
        return f"   [OK]      {remote_name} downloaded ({os.path.getsize(dest)} bytes)"
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        return f"   [skip]    {remote_name} not published yet — fresh-brain mode"


def main() -> int:
    print("[..] Checking for pretrained brain files...")
    results = [fetch(name) for name in FILES]
    for line in results:
        print(line)
    if not any("downloaded" in r for r in results):
        print("[..] Fresh brain: watch the dashboard's score curves move up as it learns.")
    else:
        print("[OK] Pretrained brain assembled — improved starting knowledge loaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

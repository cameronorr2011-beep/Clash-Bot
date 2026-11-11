package com.neuroml.app;

import android.Manifest;
import android.accessibilityservice.AccessibilityServiceInfo;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Typeface;
import android.media.projection.MediaProjectionManager;
import android.os.Bundle;
import android.view.View;
import android.view.accessibility.AccessibilityManager;
import android.widget.CompoundButton;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Button;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * ONE control: the brain switch.
 *  - toggle ON  -> requests the screen-capture grant itself, asks for the tap
 *                 service if missing, then STARTS the brain and opens the
 *                 dashboard. It runs right away — nothing else to press.
 *  - toggle OFF -> stops the brain completely (BotService hard-exits python).
 * The dashboard shows the account signed in on THIS phone (the player's).
 */
public class MainActivity extends Activity {
    private static final int PROJECTION_REQ = 4242;
    private static final String PREFS = "nml";
    private TextView status;
    private Switch brainSwitch;
    private Button updateBtn;
    private boolean autoUpdateStarted;

    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.parseColor("#0b0f14"));
        root.setPadding(48, 72, 48, 48);

        TextView title = new TextView(this);
        title.setText("NeuroML");
        title.setTextColor(Color.parseColor("#58a6ff"));
        title.setTextSize(30);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        root.addView(title);

        TextView sub = new TextView(this);
        sub.setText("The full RL brain runs on THIS phone. It sees the game through the screen, plays matches, learns, and federates anonymized packages. No API key, no computer.");
        sub.setTextColor(Color.parseColor("#8b98a8"));
        sub.setTextSize(14);
        sub.setPadding(0, 16, 0, 24);
        root.addView(sub);

        brainSwitch = new Switch(this);
        brainSwitch.setText("  🧠 BRAIN (run / stop)");
        brainSwitch.setTextSize(18);
        brainSwitch.setTextColor(Color.WHITE);
        root.addView(brainSwitch);

        TextView addr = new TextView(this);
        addr.setText("Dashboard address: http://127.0.0.1:8765\n(reachable only on this phone, only while the brain runs)");
        addr.setTextColor(Color.parseColor("#58a6ff"));
        addr.setTextSize(12);
        addr.setPadding(0, 0, 0, 8);
        root.addView(addr);

        status = new TextView(this);
        status.setTextColor(Color.parseColor("#3fb950"));
        status.setTextSize(13);
        status.setPadding(0, 16, 0, 28);
        root.addView(status);

        root.addView(btn("📱 Open live dashboard", v -> {
            copyAssets();
            startActivity(new Intent(this, DashboardActivity.class));
        }));

        // ---- AUTOUPDATE (v27: every push publishes a new release — keep up) --
        updateBtn = btn("⬇ Update available — tap to install", v -> installUpdate());
        updateBtn.setVisibility(View.GONE);
        updateBtn.setBackgroundColor(Color.parseColor("#123b33"));
        root.addView(updateBtn);
        startUpdateCheck();

        TextView note = new TextView(this);
        note.setText("Automating Clash Royale violates Supercell's ToS — use an account you accept losing. The bot taps only inside the game and never opens account or deck screens.");
        note.setTextColor(Color.parseColor("#6e7b8a"));
        note.setTextSize(11);
        note.setPadding(0, 40, 0, 0);
        root.addView(note);

        brainSwitch.setOnCheckedChangeListener((CompoundButton sb, boolean on) -> {
            if (on) {
                proceedToRun();
            } else {
                stopEverything();
            }
        });

        setContentView(new ScrollView(this) {{ addView(root); }});

        // ---- AUTO-START (v33): opening the app is enough ---------------------
        // The brain + dashboard used to require a manual toggle after every
        // launch. With auto_start (default ON) the app walks straight into the
        // same start chain, so the localhost dashboard comes up by itself.
        // Android still requires the one-time screen-capture consent per
        // session, so that dialog is the only thing the user touches.
        autoStartIfConfigured();
    }

    /** Start the brain chain on launch unless the user switched auto-start off. */
    private void autoStartIfConfigured() {
        SharedPreferences p = getSharedPreferences(PREFS, Context.MODE_PRIVATE);
        if (!p.getBoolean("auto_start", true)) return;
        if (p.getBoolean("brain_running", false)) {
            startActivity(new Intent(this, DashboardActivity.class));
            return;
        }
        new android.os.Handler(android.os.Looper.getMainLooper())
                .postDelayed(this::proceedToRun, 500);
    }

    /** Persist the user's auto-start choice (off = manual toggle every launch). */
    public void setAutoStart(boolean on) {
        getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
                .putBoolean("auto_start", on).commit();
    }

    /** Toggling OFF: stop the brain AND retire the screen-capture session
     *  (its process is going down; a fresh one-time grant is requested on the
     *  next start — Android security calls for that anyway).
     *  v27 hardening: the MAIN process commits its own truth synchronously
     *  too, and stopService() runs beside the ACTION_STOP intent so a frozen
     *  :brain process can never leave a zombie that the toggle can't kill. */
    private void stopEverything() {
        Intent stop = new Intent(this, BrainService.class);
        stop.setAction(BrainService.ACTION_STOP);
        try { startService(stop); } catch (Exception ignored) { }
        try { stopService(new Intent(this, BrainService.class)); } catch (Exception ignored) { }
        try { stopService(new Intent(this, CaptureService.class)); } catch (Exception ignored) { }
        getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
                .putBoolean("capture_ready", false)
                .putBoolean("brain_running", false)
                .commit();   // synchronous: the toggle must reflect this NOW
        status.setText("Brain stopped — nothing is playing. Toggle ON to start again.");
    }

    @Override
    protected void onResume() {
        super.onResume();
        SharedPreferences p = getSharedPreferences(PREFS, Context.MODE_PRIVATE);
        boolean running = p.getBoolean("brain_running", false);
        // v27 truth check: a stale flag used to survive Android reclaiming the
        // :brain process — the toggle then showed a LIE and OFF did nothing.
        // The heartbeat (cr_bot/brain_alive, touched every 3s by the live
        // brain) is the ground truth: stale heartbeat ≠ running.
        if (running) {
            File beat = new File(getFilesDir(), "cr_bot/brain_alive");
            long beatMs = beat.lastModified();
            boolean alive = beat.exists() && beatMs > 0
                    && (System.currentTimeMillis() - beatMs) < 30_000;
            if (!alive) {
                p.edit().putBoolean("brain_running", false).commit();
                p.edit().putBoolean("capture_ready", false).commit();
                p.edit().putString("brain_error",
                        "process was reclaimed by Android — toggle ON to restart").commit();
                running = false;
            }
        }
        brainSwitch.setOnCheckedChangeListener(null);   // reflect state, don't retrigger
        brainSwitch.setChecked(running);
        brainSwitch.setOnCheckedChangeListener((CompoundButton sb, boolean on) -> {
            if (on) proceedToRun();
            else stopEverything();
        });
        String err = p.getString("brain_error", "");
        if (!running && err != null && !err.isEmpty()) {
            status.setText("✖ Brain stopped: " + err + "\nToggle ON to try again.");
        } else if (running) {
            status.setText("✓ Brain running — it plays the account signed in on this phone.\nDashboard: the in-app screen, or any browser at http://127.0.0.1:8765 while it runs.");
        }
    }

    /** Cross-process capture readiness (CaptureService runs in :brain).
     *  SharedPreferences are per-process cached — a write from the :brain
     *  process is NOT visible through the cached object in this process, so
     *  poll the actual file on disk instead. (This stale-cache read is what
     *  made the poll report "capture taking long" even after a grant.) */
    private boolean captureReady() {
        try {
            File f = new File(getApplicationInfo().dataDir, "shared_prefs/nml.xml");
            BufferedReader r = new BufferedReader(new java.io.FileReader(f));
            String line;
            while ((line = r.readLine()) != null) {
                if (line.contains("capture_ready") && line.contains("\"true\"")) {
                    r.close();
                    return true;
                }
            }
            r.close();
        } catch (Exception ignored) { }
        return false;
    }

    /** Toggle ON or returning from a permission grant: run what's missing, in order. */
    private void proceedToRun() {
        // Already running? Just show the dashboard instead of re-running the chain
        // (double-toggle was what poisoned the state on older builds).
        if (getSharedPreferences(PREFS, Context.MODE_PRIVATE).getBoolean("brain_running", false)) {
            startActivity(new Intent(this, DashboardActivity.class));
            return;
        }
        if (!captureReady()) {
            status.setText(" Android needs your one-time OK to see the screen…");
            requestCapture();
            return;
        }
        if (!tapsEnabled()) {
            status.setText(" Android needs the NeuroML tap service ON — enabling it in settings…");
            startActivity(new Intent(android.provider.Settings.ACTION_ACCESSIBILITY_SETTINGS));
            Toast.makeText(this, "Find \u201CNeuroML brain tap service\u201D and switch it ON, then come back — the brain will start by itself", Toast.LENGTH_LONG).show();
            return;
        }
        startBrain();
    }

    /** Hard, reliable check: read the SYSTEM's enabled accessibility list
     *  (static-instance detection lost races when the app process restarted). */
    private boolean tapsEnabled() {
        AccessibilityManager am = (AccessibilityManager) getSystemService(ACCESSIBILITY_SERVICE);
        for (AccessibilityServiceInfo info : am.getEnabledAccessibilityServiceList(AccessibilityServiceInfo.FEEDBACK_ALL_MASK)) {
            String pkg = info.getResolveInfo() == null ? "" :
                    info.getResolveInfo().serviceInfo == null ? "" :
                            info.getResolveInfo().serviceInfo.packageName;
            if (getPackageName().equals(pkg)) return true;
        }
        return false;
    }

    private void startBrain() {
        copyAssets();
        startForegroundService(new Intent(this, BrainService.class));
        startActivity(new Intent(this, DashboardActivity.class));
    }

    private android.widget.Button btn(String label, View.OnClickListener c) {
        android.widget.Button b = new android.widget.Button(this);
        b.setText(label);
        b.setAllCaps(false);
        b.setTextColor(Color.WHITE);
        b.setPadding(16, 16, 16, 16);
        b.setOnClickListener(c);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        lp.setMargins(0, 12, 0, 12);
        b.setLayoutParams(lp);
        return b;
    }

    // ================= AUTO-UPDATE (v27) =====================================
    // Every push publishes NeuroML-version.txt next to the APK in the `apps`
    // release. On launch the app fetches it, compares with its own
    // BuildConfig.VERSION_NAME, and if a NEWER version exists shows a one-tap
    // "Update available" button that downloads the APK and hands it to the
    // Android package installer (REQUEST_INSTALL_PACKAGES permission).
    private void startUpdateCheck() {
        new Thread(() -> {
            try {
                String raw = readUrl("https://github.com/cameronorr2011-beep/Clash-Bot/releases/download/android/NeuroML-version.txt");
                final String v = raw == null ? null : raw.trim();
                final String mine = getPackageManager()
                        .getPackageInfo(getPackageName(), 0).versionName;
                if (v != null && !v.isEmpty() && !v.equals(mine)
                        && versionValue(v) > versionValue(mine)) {
                    runOnUiThread(() -> {
                        updateBtn.setText("⬆ Update available: v" + v + " (installed " + mine + ")");
                        updateBtn.setVisibility(View.VISIBLE);
                        // AUTO-UPDATE: fetch it without waiting for a tap, so a
                        // new build lands without anyone re-installing by hand.
                        // Android still shows ONE install confirmation — a truly
                        // silent install needs root or device-owner privileges,
                        // which this app deliberately does not use.
                        if (!autoUpdateStarted) {
                            autoUpdateStarted = true;
                            installUpdate();
                        }
                    });
                }
            } catch (Exception e) {
                android.util.Log.w("NeuroML-upd", "update check failed: " + e);
            }
        }, "neuroml-update").start();
    }

    private static long versionValue(String v) {
        long acc = 0;
        for (String part : v.replace("v", "").split("\\.")) {
            try { acc = acc * 1000 + Long.parseLong(part); } catch (Exception e) { acc *= 1000; }
        }
        return acc;
    }

    private static String readUrl(String spec) {
        try {
            java.net.HttpURLConnection c = (java.net.HttpURLConnection) new java.net.URL(spec).openConnection();
            c.setConnectTimeout(4000);
            c.setReadTimeout(6000);
            c.setInstanceFollowRedirects(true);
            try (java.io.BufferedReader r = new java.io.BufferedReader(new java.io.InputStreamReader(c.getInputStream()))) {
                return r.readLine();
            }
        } catch (java.io.IOException e) {
            return null;
        }
    }

    private void installUpdate() {
        try {
            File apkFile = new File(getFilesDir(), "update/NeuroML-Android.apk");
            final String target = updateApkVersion();
            String cachedVer = getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                    .getString("update_cached_ver", "");
            // Auto-updates must never install a stale download left over from an
            // older release, so the cache is only reused for the advertised one.
            if (apkFile.exists() && apkFile.length() > 1_000_000
                    && target != null && target.equals(cachedVer)) {
                launchInstaller(apkFile);
                return;
            }
            if (apkFile.exists()) apkFile.delete();
            status.setTextColor(Color.parseColor("#8b98a8"));
            status.setText("Downloading update… (5-10 MB)");
            new Thread(() -> {
                try {
                    File dir = new File(getFilesDir(), "update");
                    dir.mkdirs();
                    java.io.InputStream in = new java.net.URL(
                            "https://github.com/cameronorr2011-beep/Clash-Bot/releases/download/android/NeuroML-Android.apk")
                            .openStream();
                    File tmp = new File(dir, "NeuroML-Android.apk.tmp");
                    try (java.io.OutputStream out = new FileOutputStream(tmp)) {
                        byte[] buf = new byte[16384];
                        int n;
                        while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
                    }
                    tmp.renameTo(apkFile);
                    getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
                            .putString("update_cached_ver", target).commit();
                    runOnUiThread(() -> launchInstaller(apkFile));
                } catch (Exception e) {
                    runOnUiThread(() -> {
                        status.setTextColor(Color.parseColor("#f85c70"));
                        status.setText("✖ Update download failed: " + e + "\nInstall manually from the GitHub releases page instead.");
                    });
                }
            }, "neuroml-download").start();
        } catch (Exception e) {
            status.setText("✖ Update failed: " + e);
        }
    }

    private void launchInstaller(File apkFile) {
        try {
            android.net.Uri uri = androidx.core.content.FileProvider.getUriForFile(
                    this, "com.neuroml.app.update", apkFile);
            Intent i = new Intent(Intent.ACTION_VIEW)
                    .setDataAndType(uri, "application/vnd.android.package-archive")
                    .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_ACTIVITY_NEW_TASK);
            if (android.os.Build.VERSION.SDK_INT >= 26
                    && !getPackageManager().canRequestPackageInstalls()) {
                status.setTextColor(Color.parseColor("#d9a13b"));
                status.setText("Tap  Install unknown apps  and allow NeuroML, then tap ⬆ again — one-time.");
                startActivity(new Intent(android.provider.Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES,
                        android.net.Uri.parse("package:" + getPackageName())));
                return;
            }
            status.setTextColor(Color.parseColor("#3ad0b6"));
            status.setText("★ Confirm the installer that popped up — v" + updateApkVersion()
                    + " installs over your current build (data kept).");
            startActivity(i);
        } catch (Exception e) {
            status.setTextColor(Color.parseColor("#f85c70"));
            status.setText("✖ Installer could not start: " + e + "\nInstall manually from the GitHub releases page.");
        }
    }

    private String updateApkVersion() {
        try { return readUrl("https://github.com/cameronorr2011-beep/Clash-Bot/releases/download/android/NeuroML-version.txt").trim(); }
        catch (Exception e) { return "?"; }
    }

    private void requestCapture() {
        if (android.os.Build.VERSION.SDK_INT >= 33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS}, 3);
        }
        // The brain only needs the GAME, not the user's other apps. The consent
        // dialog lets the user hand over a single app window — guide them to
        // exactly that (the public API can't force app-only without user choice).
        status.setText("In the consent dialog, tap  🎯 App window  →  choose  Clash Royale.\nDo NOT pick the whole screen — the bot only needs the game.");
        MediaProjectionManager mgr =
                (MediaProjectionManager) getSystemService(MEDIA_PROJECTION_SERVICE);
        startActivityForResult(mgr.createScreenCaptureIntent(), PROJECTION_REQ);
    }

    @Override
    protected void onActivityResult(int req, int code, Intent data) {
        super.onActivityResult(req, code, data);
        if (req != PROJECTION_REQ || code != RESULT_OK || data == null) {
            status.setText("✖ Screen capture not granted — the brain cannot see the game");
            return;
        }
        Intent svc = new Intent(this, CaptureService.class);
        svc.putExtra(CaptureService.EXTRA_RESULT, code);
        svc.putExtras(data); // MediaProjection result payload must reach the service
        startForegroundService(svc);
        status.setText("✓ Capture active — checking the tap service…");
        // The capture service activates asynchronously (different process):
        // poll its readiness flag, then continue the chain tap → brain.
        pollCaptureReady(0);
    }

    private void pollCaptureReady(final int attempt) {
        if (captureReady()) {
            proceedToRun();
            return;
        }
        if (attempt > 60) {   // ~36 s — cold :brain process start can be slow
            status.setText("Capture is taking long — toggle 🧠 BRAIN OFF and ON again, and pick 🎯 App window → Clash Royale this time");
            return;
        }
        new android.os.Handler(android.os.Looper.getMainLooper()).postDelayed(
                () -> pollCaptureReady(attempt + 1), 600);
    }

    /**
     * bot data files (json) → filesDir so relative-path logic in bot_ml.py
     * works. Two classes:
     *  - USER data (my_deck.json, bot_config.json): once-only — an existing
     *    file is NEVER overwritten (v26 explicit rule, kept).
     *  - SHIPPED reference data (cards.json, cards_v2.json, meta_decks.json):
     *    refreshed on app update when the bundled copy differs, otherwise the
     *    phone keeps playing against an outdated card DB after every update
     *    (the deck-aware counters hotfix depends on cards_v2.json).
     */
    private void copyAssets() {
        String[] userFiles = {"my_deck.json", "bot_config.json"};
        String[] shippedFiles = {"cards.json", "cards_v2.json", "meta_decks.json"};
        for (String f : userFiles) {
            File out = new File(getFilesDir(), f);
            if (out.exists()) continue; // never overwrite user memory/config
            copyOne(f, out);
        }
        for (String f : shippedFiles) {
            File out = new File(getFilesDir(), f);
            if (!out.exists()) { copyOne(f, out); continue; }
            // refresh only when the bundled copy actually differs (cheap read)
            byte[] onDisk = readAll(out);
            byte[] bundled = readAllQuietly("neuroml/" + f);
            if (bundled != null && !java.util.Arrays.equals(onDisk, bundled)) copyOne(f, out);
        }
    }

    private void copyOne(String asset, File out) {
        try (InputStream in = getAssets().open("neuroml/" + asset);
             OutputStream os = new FileOutputStream(out)) {
            byte[] buf = new byte[8192];
            int n;
            while ((n = in.read(buf)) > 0) os.write(buf, 0, n);
        } catch (Exception ignored) { }
    }

    private byte[] readAll(File f) {
        try (InputStream in = new FileInputStream(f);
             java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream()) {
            byte[] buf = new byte[8192];
            int n;
            while ((n = in.read(buf)) > 0) bos.write(buf, 0, n);
            return bos.toByteArray();
        } catch (Exception e) { return new byte[0]; }
    }

    private byte[] readAllQuietly(String asset) {
        try (InputStream in = getAssets().open(asset);
             java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream()) {
            byte[] buf = new byte[8192];
            int n;
            while ((n = in.read(buf)) > 0) bos.write(buf, 0, n);
            return bos.toByteArray();
        } catch (Exception e) { return null; }
    }
}

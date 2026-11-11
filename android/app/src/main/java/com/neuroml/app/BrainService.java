package com.neuroml.app;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.IBinder;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.io.File;

/**
 * Runs bot_ml.py's brain loop + dashboard.py's server from inside ONE Python
 * process (android_main.py) on this phone — in its OWN ":brain" Android
 * process so a stop can hard-exit Python without touching the UI.
 * The user's Clash account shown on screen is whoever is signed in on this
 * device — the player's own account.
 */
public class BrainService extends Service {
    private static final String CH = "neuroml";
    public static final String ACTION_STOP = "com.neuroml.app.STOP";
    /** Last failure from the Python side ("screen capture not granted…" etc.).
     *  Mirrored into the "nml" prefs so the MAIN process can read it — static
     *  fields do NOT cross process boundaries. */
    public static volatile String pythonCrashed;

    /** Called from android_main on any failure: honest OFF + reason for the UI.
     *  A real failure also retires the capture session — its MediaProjection
     *  token died with whatever crashed, so keeping capture_ready=true would
     *  make the NEXT toggle skip the grant dialog and hang on a dead capture. */
    public static void reportError(Context c, String msg) {
        pythonCrashed = msg;
        // v27: commit() (synchronous) — these flag writes are the ONLY truth
        // the UI toggle gets, and the caller may os._exit() THIS PROCESS
        // milliseconds later; apply() is async and could be lost mid-exit,
        // which left the toggle permanently stuck ("can't be turned off").
        if (msg != null) c.getSharedPreferences("nml", Context.MODE_PRIVATE)
                .edit().putBoolean("capture_ready", false).commit();
        c.getSharedPreferences("nml", Context.MODE_PRIVATE).edit()
                .putString("brain_error", msg == null ? "" : msg)
                .putBoolean("brain_running", msg == null).commit();
    }

    @Override
    public IBinder onBind(Intent i) { return null; }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null && ACTION_STOP.equals(intent.getAction())) {
            setRunning(false);
            reportError(this, null);
            stopForeground(true);
            stopSelf();
            hardExitPython();
            return START_NOT_STICKY;
        }
        // START_STICKY auto-restart after the process was killed: only auto-run
        // again if the previous run was HEALTHY (Android reclaimed memory). A
        // failed run must never loop: fail → exit → restart → fail…
        if (intent == null) {
            boolean wasHealthy = pythonCrashed == null
                    && getSharedPreferences("nml", MODE_PRIVATE)
                            .getBoolean("brain_running", false);
            if (!wasHealthy) {
                setRunning(false);
                stopSelf();
                return START_NOT_STICKY;
            }
        }
        setRunning(true);
        reportError(this, null);
        startForeground(2, buildNotif("running"));

        if (!Python.isStarted()) {
            Python.start(new AndroidPlatform(getApplicationContext()));
        }
        final File filesDir = getFilesDir();
        new Thread(() -> {
            try {
                Python py = Python.getInstance();
                PyObject mod = py.getModule("android_main");
                // pass this Service as the Context so android_main can report
                // failures straight into the shared prefs we agreed on
                Object res = mod.callAttr("run", filesDir.getAbsolutePath(), BrainService.this);
                if (res != null && !"running".equals(res.toString())) {
                    // android_main already called reportError — now tear the
                    // :brain process down so the next toggle starts fresh
                    // (no zombie python threads holding ports and state).
                    stopSelf();
                    hardExitPython();
                }
            } catch (Exception e) {
                android.util.Log.e("NeuroML", "brain crashed: " + e, e);
                reportError(this, String.valueOf(e));
                hardExitPython();
            }
        }, "neuroml-brain").start();

        // Auto-restart support when Android reclaims the process: a null intent
        // (START_STICKY restart) still means "brain should be running".
        return START_STICKY;
    }

    private void hardExitPython() {
        new Thread(() -> {
            try {
                Thread.sleep(150);
                if (Python.isStarted()) {
                    Python.getInstance().getModule("os").callAttr("_exit", 0);
                }
            } catch (Throwable t) {
                // last resort: this is the :brain process, safe to terminate
                android.os.Process.killProcess(android.os.Process.myPid());
            }
        }, "neuroml-stop").start();
    }

    private void setRunning(boolean r) {
        getSharedPreferences("nml", Context.MODE_PRIVATE)
                .edit().putBoolean("brain_running", r).commit();
    }

    private Notification buildNotif(String text) {
        NotificationManager nm = getSystemService(NotificationManager.class);
        Notification.Builder b;
        if (android.os.Build.VERSION.SDK_INT >= 26) {
            nm.createNotificationChannel(new NotificationChannel(CH, "NeuroML",
                    NotificationManager.IMPORTANCE_LOW));
            b = new Notification.Builder(this, CH);
        } else {
            b = new Notification.Builder(this);
        }
        return b.setContentTitle("NeuroML brain " + text)
                .setContentText("It plays the account signed in on this phone")
                .setSmallIcon(android.R.drawable.ic_media_play)
                .build();
    }

    @Override
    public void onDestroy() {
        setRunning(false);
        super.onDestroy();
    }
}

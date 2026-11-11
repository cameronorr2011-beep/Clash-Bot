package com.neuroml.app;

import android.accessibilityservice.AccessibilityService;
import android.accessibilityservice.GestureDescription;
import android.graphics.Path;
import android.os.Bundle;
import android.util.Log;
import android.view.accessibility.AccessibilityEvent;

/**
 * The only way the brain touches the game: tap gestures at the coordinates it
 * decides. Runs on THIS phone, so no computer or ADB is involved.
 */
public class TapAccessibilityService extends AccessibilityService {
    private static final String TAG = "NeuroMLTap";
    private static volatile TapAccessibilityService instance;
    /** Foreground app tracking for the brain's "is Clash open?" gate.
     *  The brain process runs INSIDE this same process, so this static is safely
     *  shared — and it replaces the adb `dumpsys window` probe that desktop
     *  bot_ml.py uses (apps can't run dumpsys; this service simply SEESES the
     *  window-state-change broadcast the game fires whenever it takes focus). */
    private static volatile String foregroundPackage = "";
    private static volatile long foregroundAt = 0;

    @Override
    protected void onServiceConnected() {
        super.onServiceConnected();
        instance = this;
        Log.i(TAG, "tap service connected");
    }

    @Override
    public void onAccessibilityEvent(AccessibilityEvent event) {
        if (event != null && event.getEventType() == AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED) {
            CharSequence p = event.getPackageName();
            if (p != null && p.length() > 0) {
                foregroundPackage = p.toString();
                foregroundAt = System.currentTimeMillis();
            }
        }
    }

    /** Package id of the app that took window focus most recently (best-effort
     *  foreground probe — battles take focus, so seeing com.supercell here IS
     *  "the user has Clash open"). A few seconds of staleness is fine: the
     *  gate only decides whether to start playing, and stale info never taps
     *  blind because every tap still goes through the live service. */
    public static String foregroundPackage() {
        return foregroundPackage;
    }

    public static long foregroundAtMillis() { return foregroundAt; }

    @Override
    public void onInterrupt() { }

    @Override
    public void onDestroy() {
        instance = null;
        // the process usually goes down WITH the service; keep the probe honest
        foregroundPackage = "";
        super.onDestroy();
    }

    public static boolean available() { return instance != null; }

    /** Tap once. Blocking up to ~150 ms. */
    public static boolean tap(int x, int y) {
        TapAccessibilityService s = instance;
        if (s == null) return false;
        try {
            Path p = new Path();
            p.moveTo(x, y);
            GestureDescription.Builder b = new GestureDescription.Builder();
            b.addStroke(new GestureDescription.StrokeDescription(p, 0, 55));
            final boolean[] done = {false};
            s.dispatchGesture(b.build(), new AccessibilityService.GestureResultCallback() {
                @Override public void onCompleted(GestureDescription g) { done[0] = true; }
                @Override public void onCancelled(GestureDescription g) { }
            }, null);
            long deadline = System.currentTimeMillis() + 600;
            while (!done[0] && System.currentTimeMillis() < deadline) {
                try { Thread.sleep(12); } catch (InterruptedException ignored) {}
            }
            return done[0];
        } catch (Exception e) {
            Log.w(TAG, "tap failed: " + e);
            return false;
        }
    }
}

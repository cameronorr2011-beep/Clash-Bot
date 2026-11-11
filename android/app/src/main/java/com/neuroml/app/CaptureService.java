package com.neuroml.app;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.PixelFormat;
import android.hardware.display.DisplayManager;
import android.hardware.display.VirtualDisplay;
import android.media.Image;
import android.media.ImageReader;
import android.media.projection.MediaProjection;
import android.media.projection.MediaProjectionManager;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.util.Log;
import android.view.Display;
import android.view.WindowManager;

import java.io.File;
import java.io.FileOutputStream;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

/**
 * Captures the screen of THIS phone (the phone running Clash Royale) with
 * MediaProjection — no ADB, no computer, works on any network. Python's
 * brain sees the frames through CaptureBridge.
 */
public class CaptureService extends Service {
    public static final int REQUEST_CODE = 4242;
    public static final String EXTRA_RESULT = "result";
    private static final String TAG = "NeuroMLCapture";
    private static final String CH = "neuroml";

    private static MediaProjection projection;
    private static VirtualDisplay vdisplay;
    private static ImageReader reader;
    private static volatile Bitmap latest;
    private static Semaphore frameReady = new Semaphore(0);
    private static HandlerThread readerThread;
    private static Handler readerHandler;
    private static boolean captureRequested;

    @Override
    public IBinder onBind(Intent i) { return null; }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent == null || !intent.hasExtra(EXTRA_RESULT)) return START_NOT_STICKY;
        // Android 14+ requires the FGS to run with the mediaProjection type BEFORE
        // getMediaProjection() is called — otherwise SecurityException.
        if (android.os.Build.VERSION.SDK_INT >= 29) {
            startForeground(1, buildNotif(),
                    android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION);
        } else {
            startForeground(1, buildNotif());
        }
        Context c = getApplicationContext();
        MediaProjectionManager mgr =
                (MediaProjectionManager) c.getSystemService(Context.MEDIA_PROJECTION_SERVICE);
        if (projection != null) { projection.stop(); releaseDisplay(); }
        try {
            android.graphics.Point size = new android.graphics.Point();
            android.hardware.display.DisplayManager dm =
                    (android.hardware.display.DisplayManager) c.getSystemService(Context.DISPLAY_SERVICE);
            Display d = dm.getDisplay(Display.DEFAULT_DISPLAY);
            if (d == null) {
                WindowManager wm = (WindowManager) c.getSystemService(Context.WINDOW_SERVICE);
                d = wm.getDefaultDisplay();
            }
            d.getRealSize(size);
            int w = Math.min(1080, size.x);
            int h = Math.min(2340, size.y);
            projection = mgr.getMediaProjection(intent.getIntExtra(EXTRA_RESULT, 0), intent);
            reader = ImageReader.newInstance(w, h, PixelFormat.RGBA_8888, 2);
            readerThread = new HandlerThread("neuroml-capture");
            readerThread.start();
            readerHandler = new Handler(readerThread.getLooper());
            reader.setOnImageAvailableListener(r -> {
                Image img = null;
                try {
                    img = r.acquireLatestImage();
                    if (img == null) return;
                    latest = toBitmap(img);
                    frameReady.release();
                } catch (Exception e) {
                    Log.w(TAG, "frame: " + e);
                } finally {
                    if (img != null) img.close();
                }
            }, readerHandler);
            vdisplay = projection.createVirtualDisplay("neuroml", w, h, 420,
                    DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR, reader.getSurface(), null, readerHandler);
            projection.registerCallback(new MediaProjection.Callback() {
                @Override public void onStop() { releaseDisplay(); }
            }, readerHandler);
            captureRequested = true;
            // Cross-process readiness flag: MainActivity (main process) polls the
            // XML file on disk (its SharedPreferences cache never sees writes from
            // this process), so commit synchronously and deterministically here.
            getSharedPreferences("nml", Context.MODE_PRIVATE)
                    .edit().putBoolean("capture_ready", true).commit();
            Log.i(TAG, "capture ready");
        } catch (Exception e) {
            Log.e(TAG, "capture setup failed", e);
            captureRequested = false;
        }
        return START_STICKY;
    }

    private static Bitmap toBitmap(Image img) {
        Image.Plane p = img.getPlanes()[0];
        int rowStride = p.getRowStride();
        int pixelStride = p.getPixelStride();
        int width = img.getWidth();
        int rowPad = rowStride - pixelStride * width;
        Bitmap bmp = Bitmap.createBitmap(width + rowPad / pixelStride, img.getHeight(),
                Bitmap.Config.ARGB_8888);
        bmp.copyPixelsFromBuffer(p.getBuffer());
        if (rowPad != 0) {
            bmp = Bitmap.createBitmap(bmp, 0, 0, width, img.getHeight());
        }
        return bmp;
    }

    /** Python calls this: block until a fresh frame, compress to PNG at path. */
    public static boolean screenshotPng(String path, double timeoutS) {
        if (!captureRequested || projection == null) return false;
        long gave = frameReady.drainPermits();
        try {
            if (!frameReady.tryAcquire((long) (timeoutS * 1000), TimeUnit.MILLISECONDS)) return false;
        } catch (InterruptedException e) {
            return false;
        }
        Bitmap b = latest;
        if (b == null) return false;
        try {
            File f = new File(path);
            f.getParentFile().mkdirs();
            FileOutputStream out = new FileOutputStream(f);
            b.compress(Bitmap.CompressFormat.PNG, 100, out);
            out.close();
            return f.length() > 1000;
        } catch (Exception e) {
            return false;
        }
    }

    public static int screenW() {
        if (reader != null) return reader.getWidth();
        return 0;
    }

    public static int screenH() {
        if (reader != null) return reader.getHeight();
        return 0;
    }

    public static boolean ready() { return captureRequested && projection != null; }

    private void releaseDisplay() {
        captureRequested = false;
        getSharedPreferences("nml", Context.MODE_PRIVATE)
                .edit().putBoolean("capture_ready", false).apply();
        try { if (reader != null) reader.close(); } catch (Exception ignored) {}
        try { if (vdisplay != null) vdisplay.release(); } catch (Exception ignored) {}
        reader = null; vdisplay = null; latest = null;
    }

    private Notification buildNotif() {
        NotificationManager nm = getSystemService(NotificationManager.class);
        if (android.os.Build.VERSION.SDK_INT >= 26) {
            nm.createNotificationChannel(new NotificationChannel(CH, "NeuroML",
                    NotificationManager.IMPORTANCE_LOW));
            return new Notification.Builder(this, CH)
                    .setContentTitle("NeuroML seeing the game")
                    .setContentText("Screen capture active — the brain plays what you see")
                    .setSmallIcon(android.R.drawable.ic_menu_compass)
                    .build();
        }
        return new Notification.Builder(this).build();
    }

    @Override
    public void onDestroy() {
        try { if (projection != null) projection.stop(); } catch (Exception ignored) {}
        releaseDisplay();
        super.onDestroy();
    }
}

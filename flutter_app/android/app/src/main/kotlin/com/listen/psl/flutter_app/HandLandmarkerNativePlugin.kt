package com.listen.psl.flutter_app

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import io.flutter.embedding.engine.loader.FlutterLoader
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.MethodChannel
import java.io.FileOutputStream

class HandLandmarkerNativePlugin(
    private val context: Context,
    messenger: BinaryMessenger,
) : MethodChannel.MethodCallHandler {

    private val channel = MethodChannel(messenger, CHANNEL).apply {
        setMethodCallHandler(this@HandLandmarkerNativePlugin)
    }

    private var landmarker: HandLandmarker? = null

    // Reused per-frame scratch — sized lazily on first frame.
    // Bitmap itself can't be reused: MPImage.close() recycles the underlying bitmap,
    // so we allocate a fresh one each frame and let MediaPipe own its lifecycle.
    private var nv21: ByteArray = ByteArray(0)
    private var argb: IntArray = IntArray(0)

    override fun onMethodCall(call: io.flutter.plugin.common.MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "init" -> {
                val numHands = call.argument<Int>("numHands") ?: 2
                val minConf = (call.argument<Double>("minConf") ?: 0.5).toFloat()
                val useGpu = call.argument<Boolean>("useGpu") ?: true
                val assetPath = call.argument<String>("assetPath")
                if (assetPath == null) {
                    result.error("ARG", "assetPath required", null); return
                }
                try {
                    initLandmarker(assetPath, numHands, minConf, useGpu)
                    result.success(null)
                } catch (e: Throwable) {
                    Log.e(TAG, "init failed", e)
                    result.error("INIT", e.message, null)
                }
            }
            "detectYuv" -> {
                try {
                    val res = detectYuv(call)
                    result.success(res)
                } catch (e: Throwable) {
                    Log.e(TAG, "detect failed", e)
                    result.error("DETECT", e.message, null)
                }
            }
            "dispose" -> {
                landmarker?.close()
                landmarker = null
                result.success(null)
            }
            else -> result.notImplemented()
        }
    }

    private fun initLandmarker(assetPath: String, numHands: Int, minConf: Float, useGpu: Boolean) {
        landmarker?.close()

        // setModelAssetPath needs a real file path, not an asset lookup key —
        // copy from APK assets to internal storage on first init.
        val loader = FlutterLoader()
        if (!loader.initialized()) loader.startInitialization(context)
        loader.ensureInitializationComplete(context, null)
        val lookupKey = loader.getLookupKeyForAsset(assetPath)

        val outFile = java.io.File(context.filesDir, "hand_landmarker.task")
        if (!outFile.exists() || outFile.length() == 0L) {
            context.assets.open(lookupKey).use { input ->
                FileOutputStream(outFile).use { output -> input.copyTo(output) }
            }
        }

        val baseOptions = BaseOptions.builder()
            .setModelAssetPath(outFile.absolutePath)
            .setDelegate(if (useGpu) Delegate.GPU else Delegate.CPU)
            .build()
        val options = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setNumHands(numHands)
            // VIDEO mode: per-frame detect with temporal coherence between frames,
            // significantly reduces landmark jitter vs IMAGE (independent per call).
            .setRunningMode(RunningMode.VIDEO)
            .setMinHandDetectionConfidence(minConf)
            .setMinHandPresenceConfidence(minConf)
            .setMinTrackingConfidence(minConf)
            .build()
        landmarker = HandLandmarker.createFromOptions(context, options)
        Log.i(TAG, "HandLandmarker ready (gpu=$useGpu, numHands=$numHands)")
    }

    private fun detectYuv(call: io.flutter.plugin.common.MethodCall): DoubleArray {
        val landmarker = this.landmarker ?: throw IllegalStateException("not initialized")
        val y = call.argument<ByteArray>("y") ?: throw IllegalArgumentException("y required")
        val u = call.argument<ByteArray>("u") ?: throw IllegalArgumentException("u required")
        val v = call.argument<ByteArray>("v") ?: throw IllegalArgumentException("v required")
        val width = call.argument<Int>("width") ?: throw IllegalArgumentException("width required")
        val height = call.argument<Int>("height") ?: throw IllegalArgumentException("height required")
        val yRowStride = call.argument<Int>("yRowStride") ?: width
        val uvRowStride = call.argument<Int>("uvRowStride") ?: (width / 2)
        val uvPixelStride = call.argument<Int>("uvPixelStride") ?: 1
        val rotation = call.argument<Int>("rotation") ?: 0

        // Pack into NV21, then NV21 → ARGB → Bitmap. MediaPipe Tasks Vision only accepts
        // Bitmap-backed MPImage (ByteBuffer with NV21 throws "Unsupported image format: 4").
        val nvSize = width * height * 3 / 2
        if (nv21.size < nvSize) nv21 = ByteArray(nvSize)
        yuvToNv21(y, u, v, width, height, yRowStride, uvRowStride, uvPixelStride, nv21)

        val px = width * height
        if (argb.size < px) argb = IntArray(px)
        nv21ToArgb(nv21, width, height, argb)

        val bm = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bm.setPixels(argb, 0, width, 0, 0, width, height)

        val mpImage: MPImage = BitmapImageBuilder(bm).build()

        val ipo = ImageProcessingOptions.builder()
            .setRotationDegrees(rotation)
            .build()

        // VIDEO mode requires monotonically increasing timestamps in ms.
        val tsMs = System.currentTimeMillis()
        val result = landmarker.detectForVideo(mpImage, ipo, tsMs)
        mpImage.close()

        if (result == null || result.landmarks().isEmpty()) {
            return DoubleArray(1).also { it[0] = 0.0 }
        }

        // Pack: [numHands, (handedness, 21*(x,y,z))×numHands]
        // handedness: 0.0 = Left, 1.0 = Right (raw MediaPipe label — Dart inverts).
        val n = result.landmarks().size
        val out = DoubleArray(1 + n * (1 + 63))
        out[0] = n.toDouble()
        var w = 1
        for (i in 0 until n) {
            val hand = result.landmarks()[i]
            val cat = result.handedness()[i].first()
            out[w++] = if (cat.categoryName() == "Right") 1.0 else 0.0
            for (j in 0 until 21) {
                val lm = hand[j]
                out[w++] = lm.x().toDouble()
                out[w++] = lm.y().toDouble()
                out[w++] = lm.z().toDouble()
            }
        }
        return out
    }

    /**
     * Pack Flutter CameraImage YUV_420_888 planes into NV21 layout (Y plane, then interleaved
     * V/U pairs).
     */
    private fun yuvToNv21(
        y: ByteArray, u: ByteArray, v: ByteArray,
        width: Int, height: Int,
        yRowStride: Int, uvRowStride: Int, uvPixelStride: Int,
        out: ByteArray,
    ) {
        // Y plane: copy row-by-row, accounting for stride padding.
        if (yRowStride == width) {
            System.arraycopy(y, 0, out, 0, width * height)
        } else {
            var dst = 0
            for (row in 0 until height) {
                System.arraycopy(y, row * yRowStride, out, dst, width)
                dst += width
            }
        }

        // VU interleaved: NV21 expects V first, then U, per chroma sample.
        val uvWidth = width / 2
        val uvHeight = height / 2
        var dst = width * height
        for (row in 0 until uvHeight) {
            val rowBase = row * uvRowStride
            for (col in 0 until uvWidth) {
                val idx = rowBase + col * uvPixelStride
                out[dst++] = v[idx]
                out[dst++] = u[idx]
            }
        }
    }

    /**
     * NV21 → ARGB_8888 using BT.601 fixed-point coefficients. Tight loop, no allocation.
     * Pixel 3 XL handles 320×240 in ~2-3ms — well under the per-frame budget.
     */
    private fun nv21ToArgb(nv21: ByteArray, width: Int, height: Int, out: IntArray) {
        val frameSize = width * height
        var yp = 0
        for (j in 0 until height) {
            var uvp = frameSize + (j shr 1) * width
            var u = 0
            var v = 0
            for (i in 0 until width) {
                var yVal = (0xff and nv21[yp].toInt()) - 16
                if (yVal < 0) yVal = 0
                if (i and 1 == 0) {
                    v = (0xff and nv21[uvp++].toInt()) - 128
                    u = (0xff and nv21[uvp++].toInt()) - 128
                }
                val y1192 = 1192 * yVal
                var r = y1192 + 1634 * v
                var g = y1192 - 833 * v - 400 * u
                var b = y1192 + 2066 * u
                if (r < 0) r = 0 else if (r > 262143) r = 262143
                if (g < 0) g = 0 else if (g > 262143) g = 262143
                if (b < 0) b = 0 else if (b > 262143) b = 262143
                out[yp] = (-0x1000000) or
                        ((r shl 6) and 0x00ff0000) or
                        ((g shr 2) and 0x0000ff00) or
                        ((b shr 10) and 0x000000ff)
                yp++
            }
        }
    }

    fun detach() {
        landmarker?.close()
        landmarker = null
        channel.setMethodCallHandler(null)
    }

    companion object {
        private const val TAG = "PSL/HandLM"
        private const val CHANNEL = "psl/handlandmarker"
    }
}

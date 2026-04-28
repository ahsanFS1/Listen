package com.listen.psl.flutter_app

import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Log
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.MethodChannel
import java.io.ByteArrayOutputStream

/**
 * Encodes a Flutter CameraImage YUV_420_888 frame into a JPEG byte array,
 * applying rotation. Sent over the method channel `psl/yuvjpeg`.
 *
 * We do this natively instead of in Dart to avoid the per-frame cost of
 * shipping raw YUV over the platform channel and the Dart-side allocations
 * that the `image` package incurs.
 */
class YuvJpegPlugin(messenger: BinaryMessenger) : MethodChannel.MethodCallHandler {

    private val channel = MethodChannel(messenger, CHANNEL).apply {
        setMethodCallHandler(this@YuvJpegPlugin)
    }

    private var nv21: ByteArray = ByteArray(0)

    override fun onMethodCall(
        call: io.flutter.plugin.common.MethodCall,
        result: MethodChannel.Result,
    ) {
        when (call.method) {
            "encode" -> {
                try {
                    val bytes = encode(call)
                    result.success(bytes)
                } catch (e: Throwable) {
                    Log.e(TAG, "encode failed", e)
                    result.error("ENCODE", e.message, null)
                }
            }
            else -> result.notImplemented()
        }
    }

    private fun encode(call: io.flutter.plugin.common.MethodCall): ByteArray {
        val y = call.argument<ByteArray>("y") ?: throw IllegalArgumentException("y required")
        val u = call.argument<ByteArray>("u") ?: throw IllegalArgumentException("u required")
        val v = call.argument<ByteArray>("v") ?: throw IllegalArgumentException("v required")
        val width = call.argument<Int>("width") ?: throw IllegalArgumentException("width required")
        val height = call.argument<Int>("height") ?: throw IllegalArgumentException("height required")
        val yRowStride = call.argument<Int>("yRowStride") ?: width
        val uvRowStride = call.argument<Int>("uvRowStride") ?: (width / 2)
        val uvPixelStride = call.argument<Int>("uvPixelStride") ?: 1
        val rotation = call.argument<Int>("rotation") ?: 0
        val quality = call.argument<Int>("quality") ?: 70

        val nvSize = width * height * 3 / 2
        if (nv21.size < nvSize) nv21 = ByteArray(nvSize)
        yuvToNv21(y, u, v, width, height, yRowStride, uvRowStride, uvPixelStride, nv21)

        val out = ByteArrayOutputStream(64 * 1024)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        yuvImage.compressToJpeg(Rect(0, 0, width, height), quality, out)
        val rawJpeg = out.toByteArray()

        if (rotation % 360 == 0) return rawJpeg

        // Rotate via Bitmap. JPEG decode + re-encode is cheap relative to the
        // network round-trip and keeps the wire format simple (JPEG only).
        val src = BitmapFactory.decodeByteArray(rawJpeg, 0, rawJpeg.size)
            ?: return rawJpeg
        val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
        val rotated = Bitmap.createBitmap(src, 0, 0, src.width, src.height, matrix, true)
        if (rotated !== src) src.recycle()
        val out2 = ByteArrayOutputStream(64 * 1024)
        rotated.compress(Bitmap.CompressFormat.JPEG, quality, out2)
        rotated.recycle()
        return out2.toByteArray()
    }

    /**
     * Pack Flutter CameraImage YUV_420_888 planes into NV21 layout (Y plane,
     * then interleaved V/U pairs).
     */
    private fun yuvToNv21(
        y: ByteArray, u: ByteArray, v: ByteArray,
        width: Int, height: Int,
        yRowStride: Int, uvRowStride: Int, uvPixelStride: Int,
        out: ByteArray,
    ) {
        if (yRowStride == width) {
            System.arraycopy(y, 0, out, 0, width * height)
        } else {
            var dst = 0
            for (row in 0 until height) {
                System.arraycopy(y, row * yRowStride, out, dst, width)
                dst += width
            }
        }
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

    fun detach() {
        channel.setMethodCallHandler(null)
    }

    companion object {
        private const val TAG = "PSL/YuvJpeg"
        private const val CHANNEL = "psl/yuvjpeg"
    }
}

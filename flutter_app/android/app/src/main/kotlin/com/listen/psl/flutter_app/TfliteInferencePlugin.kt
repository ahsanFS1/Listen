package com.listen.psl.flutter_app

import android.content.Context
import android.util.Log
import io.flutter.embedding.engine.loader.FlutterLoader
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.MethodChannel
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class TfliteInferencePlugin(
    private val context: Context,
    messenger: BinaryMessenger,
) : MethodChannel.MethodCallHandler {

    private val channel = MethodChannel(messenger, CHANNEL).apply {
        setMethodCallHandler(this@TfliteInferencePlugin)
    }

    private var interpreter: Interpreter? = null
    private var modelBuffer: MappedByteBuffer? = null
    private var numClasses: Int = 0

    private val inputBuf: ByteBuffer = ByteBuffer
        .allocateDirect(BUF_FRAMES * FRAME_DIM * 4)
        .order(ByteOrder.nativeOrder())

    override fun onMethodCall(call: io.flutter.plugin.common.MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "load" -> {
                val assetPath = call.argument<String>("assetPath")
                if (assetPath == null) {
                    result.error("ARG", "assetPath required", null); return
                }
                try {
                    load(assetPath)
                    result.success(numClasses)
                } catch (e: Throwable) {
                    Log.e(TAG, "load failed", e)
                    result.error("LOAD", e.message, null)
                }
            }
            "runInference" -> {
                val features = call.argument<ByteArray>("features")
                if (features == null) {
                    result.error("ARG", "features required", null); return
                }
                try {
                    val probs = runInference(features)
                    result.success(probs)
                } catch (e: Throwable) {
                    Log.e(TAG, "inference failed", e)
                    result.error("INFER", e.message, null)
                }
            }
            "dispose" -> {
                interpreter?.close()
                interpreter = null
                modelBuffer = null
                result.success(null)
            }
            else -> result.notImplemented()
        }
    }

    private fun load(assetPath: String) {
        interpreter?.close()

        val loader = FlutterLoader()
        if (!loader.initialized()) loader.startInitialization(context)
        loader.ensureInitializationComplete(context, null)
        val lookupKey = loader.getLookupKeyForAsset(assetPath)

        val afd = context.assets.openFd(lookupKey)
        val mapped = FileInputStream(afd.fileDescriptor).channel.map(
            FileChannel.MapMode.READ_ONLY,
            afd.startOffset,
            afd.declaredLength,
        )
        modelBuffer = mapped

        val opts = Interpreter.Options().apply {
            setNumThreads(4)
            // Select TF ops are auto-registered when tensorflow-lite-select-tf-ops is on the
            // classpath (libtensorflowlite_flex_jni.so loads via JNI on first interpreter use).
        }
        val interp = Interpreter(mapped, opts)
        interp.resizeInput(0, intArrayOf(1, BUF_FRAMES, FRAME_DIM))
        interp.allocateTensors()

        val outShape = interp.getOutputTensor(0).shape() // [1, numClasses]
        numClasses = outShape[outShape.size - 1]
        interpreter = interp
        Log.i(TAG, "TFLite loaded: input=[1,$BUF_FRAMES,$FRAME_DIM] output=[1,$numClasses]")
    }

    private fun runInference(features: ByteArray): DoubleArray {
        val interp = interpreter ?: throw IllegalStateException("interpreter not loaded")
        val expected = BUF_FRAMES * FRAME_DIM * 4
        if (features.size != expected) {
            throw IllegalArgumentException("features size ${features.size} != $expected")
        }
        inputBuf.clear()
        inputBuf.put(features)
        inputBuf.rewind()

        val output = Array(1) { FloatArray(numClasses) }
        interp.run(inputBuf, output)
        // Return as DoubleArray so Flutter's StandardMessageCodec maps it directly
        // to Float64List on the Dart side (most reliable mapping).
        val out = DoubleArray(numClasses)
        for (i in 0 until numClasses) out[i] = output[0][i].toDouble()
        return out
    }

    fun detach() {
        interpreter?.close()
        interpreter = null
        modelBuffer = null
        channel.setMethodCallHandler(null)
    }

    companion object {
        private const val TAG = "PSL/TFLite"
        private const val CHANNEL = "psl/tflite"
        private const val BUF_FRAMES = 60
        private const val FRAME_DIM = 126
    }
}

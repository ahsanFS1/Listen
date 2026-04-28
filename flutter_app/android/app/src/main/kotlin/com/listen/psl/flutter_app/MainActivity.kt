package com.listen.psl.flutter_app

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine

class MainActivity : FlutterActivity() {
    private var tflitePlugin: TfliteInferencePlugin? = null
    private var handPlugin: HandLandmarkerNativePlugin? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        val messenger = flutterEngine.dartExecutor.binaryMessenger
        tflitePlugin = TfliteInferencePlugin(applicationContext, messenger)
        handPlugin = HandLandmarkerNativePlugin(applicationContext, messenger)
    }

    override fun onDestroy() {
        tflitePlugin?.detach()
        handPlugin?.detach()
        tflitePlugin = null
        handPlugin = null
        super.onDestroy()
    }
}

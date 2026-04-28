package com.listen.psl.flutter_app

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine

class MainActivity : FlutterActivity() {
    private var tflitePlugin: TfliteInferencePlugin? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        tflitePlugin = TfliteInferencePlugin(
            applicationContext,
            flutterEngine.dartExecutor.binaryMessenger,
        )
    }

    override fun onDestroy() {
        tflitePlugin?.detach()
        tflitePlugin = null
        super.onDestroy()
    }
}

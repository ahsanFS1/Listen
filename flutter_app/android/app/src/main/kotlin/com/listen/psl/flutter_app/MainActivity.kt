package com.listen.psl.flutter_app

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine

class MainActivity : FlutterActivity() {
    private var yuvJpegPlugin: YuvJpegPlugin? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        val messenger = flutterEngine.dartExecutor.binaryMessenger
        yuvJpegPlugin = YuvJpegPlugin(messenger)
    }

    override fun onDestroy() {
        yuvJpegPlugin?.detach()
        yuvJpegPlugin = null
        super.onDestroy()
    }
}

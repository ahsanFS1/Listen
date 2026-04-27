import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:hand_landmarker/hand_landmarker.dart';
import 'sign_pipeline.dart';

// ── Messages passed between isolates ─────────────────────────────────────────

class FrameMessage {
  final Uint8List yPlane;
  final Uint8List uPlane;
  final Uint8List vPlane;
  final int width;
  final int height;
  final int uvRowStride;
  final int uvPixelStride;
  final int rotation;
  const FrameMessage({
    required this.yPlane,
    required this.uPlane,
    required this.vPlane,
    required this.width,
    required this.height,
    required this.uvRowStride,
    required this.uvPixelStride,
    required this.rotation,
  });
}

class PredictionMessage {
  final String label;
  final String english;
  final String urdu;
  final double confidence;
  final String state;
  final bool committed;
  final bool hasHands;
  const PredictionMessage({
    required this.label,
    required this.english,
    required this.urdu,
    required this.confidence,
    required this.state,
    required this.committed,
    required this.hasHands,
  });
}

// ── Background isolate entry point ────────────────────────────────────────────

void detectorIsolateEntry(SendPort mainSendPort) {
  final receivePort = ReceivePort();
  mainSendPort.send(receivePort.sendPort);

  HandLandmarkerPlugin? landmarker;
  final pipeline = SignPipeline();
  bool pipelineReady = false;

  pipeline.init().then((_) => pipelineReady = true).catchError((Object _) => false);

  receivePort.listen((message) {
    if (message == 'init') {
      landmarker = HandLandmarkerPlugin.create(
        numHands: 2,
        minHandDetectionConfidence: 0.6,
        delegate: HandLandmarkerDelegate.cpu,
      );
      mainSendPort.send('ready');
      return;
    }

    if (message == 'reset') {
      pipeline.reset();
      return;
    }

    if (message is! FrameMessage || !pipelineReady) return;

    try {
      final cameraImage = _buildCameraImage(message);
      final hands = landmarker?.detect(cameraImage, message.rotation) ?? [];
      final pred = pipeline.process(hands);

      mainSendPort.send(PredictionMessage(
        label:      pred.label,
        english:    pred.english,
        urdu:       pred.urdu,
        confidence: pred.confidence,
        state:      pred.state.name,
        committed:  pred.committed,
        hasHands:   pred.hasHands,
      ));
    } catch (_) {
      // Silently drop failed frames
    }
  });
}

// Reconstruct a CameraImage from raw YUV plane data.
// Uses the deprecated fromPlatformData constructor which accepts plain Maps —
// avoids having to implement the CameraImage interface with its private ctors.
// ignore: deprecated_member_use
CameraImage _buildCameraImage(FrameMessage m) => CameraImage.fromPlatformData({
  'format': 35, // android.graphics.ImageFormat.YUV_420_888
  'height': m.height,
  'width':  m.width,
  'lensAperture':       null,
  'sensorExposureTime': null,
  'sensorSensitivity':  null,
  'planes': [
    {
      'bytes':         m.yPlane,
      'bytesPerRow':   m.width,
      'bytesPerPixel': 1,
    },
    {
      'bytes':         m.uPlane,
      'bytesPerRow':   m.uvRowStride,
      'bytesPerPixel': m.uvPixelStride,
    },
    {
      'bytes':         m.vPlane,
      'bytesPerRow':   m.uvRowStride,
      'bytesPerPixel': m.uvPixelStride,
    },
  ],
});

// ── Manager that lives on the main isolate ────────────────────────────────────

class DetectorIsolate {
  Isolate? _isolate;
  SendPort? _sendPort;
  ReceivePort? _receivePort;

  bool _ready = false;
  bool get isReady => _ready;

  // Prevent flooding the isolate — drop incoming frames while one is in-flight
  bool _busy = false;

  void Function(PredictionMessage)? onPrediction;

  Future<void> start() async {
    _receivePort = ReceivePort();
    _isolate = await Isolate.spawn(
      detectorIsolateEntry,
      _receivePort!.sendPort,
    );

    final completer = Completer<void>();

    _receivePort!.listen((msg) {
      if (msg is SendPort) {
        _sendPort = msg;
        _sendPort!.send('init');
      } else if (msg == 'ready') {
        _ready = true;
        completer.complete();
      } else if (msg is PredictionMessage) {
        _busy = false; // isolate finished — ready for next frame
        onPrediction?.call(msg);
      }
    });

    await completer.future;
  }

  void sendFrame(CameraImage image, int rotation) {
    if (!_ready || _sendPort == null || _busy) return;
    _busy = true;

    final planes = image.planes;
    _sendPort!.send(FrameMessage(
      yPlane:        Uint8List.fromList(planes[0].bytes),
      uPlane:        Uint8List.fromList(planes[1].bytes),
      vPlane:        Uint8List.fromList(planes[2].bytes),
      width:         image.width,
      height:        image.height,
      uvRowStride:   planes[1].bytesPerRow,
      uvPixelStride: planes[1].bytesPerPixel ?? 2,
      rotation:      rotation,
    ));
  }

  void reset() => _sendPort?.send('reset');

  void stop() {
    _receivePort?.close();
    _isolate?.kill(priority: Isolate.immediate);
    _isolate     = null;
    _sendPort    = null;
    _receivePort = null;
    _ready       = false;
    _busy        = false;
  }
}

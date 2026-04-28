import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/services.dart';

/// One detected hand with its 21 landmarks and MediaPipe's raw handedness label.
///
/// Note: [isRightLabel] is the **raw** label from MediaPipe's classifier, which is
/// trained on selfie-flipped images. We feed unflipped frames, so the anatomical
/// truth is inverted (label "Right" ⇒ the signer's anatomical LEFT hand). The
/// inversion happens in the feature extractor, not here.
class NativeHand {
  final List<double> coords; // length 63 — [x0,y0,z0, x1,y1,z1, ...]
  final bool isRightLabel;
  NativeHand(this.coords, this.isRightLabel);
}

class HandLandmarkerNative {
  static const _channel = MethodChannel('psl/handlandmarker');

  bool _ready = false;
  bool get isReady => _ready;

  Future<void> init({
    int numHands = 2,
    double minConf = 0.5,
    bool useGpu = true,
  }) async {
    await _channel.invokeMethod('init', {
      'numHands': numHands,
      'minConf': minConf,
      'useGpu': useGpu,
      'assetPath': 'assets/models/hand_landmarker.task',
    });
    _ready = true;
  }

  Future<void> dispose() async {
    _ready = false;
    try {
      await _channel.invokeMethod('dispose');
    } catch (_) {}
  }

  /// Returns the hands detected in the given camera frame.
  /// Sends raw YUV planes — no JPEG roundtrip, no per-pixel work in Dart.
  Future<List<NativeHand>> detect(CameraImage image, int rotation) async {
    if (!_ready) return const [];
    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];

    final probs = await _channel.invokeMethod<Float64List>('detectYuv', {
      'y': yPlane.bytes,
      'u': uPlane.bytes,
      'v': vPlane.bytes,
      'width': image.width,
      'height': image.height,
      'yRowStride': yPlane.bytesPerRow,
      'uvRowStride': uPlane.bytesPerRow,
      'uvPixelStride': uPlane.bytesPerPixel ?? 1,
      'rotation': rotation,
    });

    if (probs == null || probs.isEmpty) return const [];
    final n = probs[0].toInt();
    if (n == 0) return const [];

    final hands = <NativeHand>[];
    int r = 1;
    for (int i = 0; i < n; i++) {
      final isRight = probs[r++] > 0.5;
      final coords = List<double>.filled(63, 0);
      for (int j = 0; j < 63; j++) {
        coords[j] = probs[r++];
      }
      hands.add(NativeHand(coords, isRight));
    }
    return hands;
  }
}

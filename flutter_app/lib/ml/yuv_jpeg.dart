import 'package:camera/camera.dart';
import 'package:flutter/services.dart';

/// Bridges to the native YuvJpegPlugin for fast JPEG encoding of camera
/// frames without copying YUV planes through Dart.
class YuvJpeg {
  static const _channel = MethodChannel('psl/yuvjpeg');

  /// Encode the given camera frame to a JPEG byte buffer, applying
  /// [rotation] degrees so the image is upright. [quality] 1-100.
  static Future<Uint8List?> encode(
    CameraImage image, {
    int rotation = 0,
    int quality = 70,
  }) async {
    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];

    final result = await _channel.invokeMethod<Uint8List>('encode', {
      'y': yPlane.bytes,
      'u': uPlane.bytes,
      'v': vPlane.bytes,
      'width': image.width,
      'height': image.height,
      'yRowStride': yPlane.bytesPerRow,
      'uvRowStride': uPlane.bytesPerRow,
      'uvPixelStride': uPlane.bytesPerPixel ?? 1,
      'rotation': rotation,
      'quality': quality,
    });
    return result;
  }
}

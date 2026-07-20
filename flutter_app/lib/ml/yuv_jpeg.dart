import 'dart:io' show Platform;

import 'package:camera/camera.dart';
import 'package:flutter/services.dart';

/// Bridges to the native YuvJpegPlugin for fast JPEG encoding of camera
/// frames without copying YUV planes through Dart.
///
/// Android's ImageFormatGroup.yuv420 delivers 3 separate Y/U/V planes
/// (YUV_420_888). iOS delivers the same format group as a *biplanar* buffer
/// (kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange) -- one Y plane plus one
/// interleaved CbCr plane, so `image.planes` only has 2 entries there. The
/// two platforms' native plugins expect different argument shapes below.
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

    if (Platform.isIOS) {
      final uvPlane = image.planes[1];
      final result = await _channel.invokeMethod<Uint8List>('encode', {
        'y': yPlane.bytes,
        'uv': uvPlane.bytes,
        'width': image.width,
        'height': image.height,
        'yRowStride': yPlane.bytesPerRow,
        'uvRowStride': uvPlane.bytesPerRow,
        'uvPixelStride': uvPlane.bytesPerPixel ?? 2,
        'rotation': rotation,
        'quality': quality,
      });
      return result;
    }

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

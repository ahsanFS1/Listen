import Flutter
import CoreVideo
import CoreImage
import CoreGraphics
import ImageIO

/// Encodes a Flutter CameraImage biplanar YUV 4:2:0 frame (iOS delivers
/// kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange for ImageFormatGroup.yuv420
/// -- one Y plane plus one interleaved CbCr plane, unlike Android's 3
/// separate Y/U/V planes) into a JPEG byte array, applying rotation.
/// Sent over the same method channel `psl/yuvjpeg` the Android plugin uses,
/// so the Dart-side call site (yuv_jpeg.dart) doesn't need per-platform UI.
public class YuvJpegPlugin: NSObject, FlutterPlugin {
  private let ciContext = CIContext(options: nil)

  public static func register(with registrar: FlutterPluginRegistrar) {
    let channel = FlutterMethodChannel(name: "psl/yuvjpeg", binaryMessenger: registrar.messenger())
    registrar.addMethodCallDelegate(YuvJpegPlugin(), channel: channel)
  }

  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard call.method == "encode" else {
      result(FlutterMethodNotImplemented)
      return
    }
    guard let args = call.arguments as? [String: Any],
          let yData = (args["y"] as? FlutterStandardTypedData)?.data,
          let uvData = (args["uv"] as? FlutterStandardTypedData)?.data,
          let width = args["width"] as? Int,
          let height = args["height"] as? Int
    else {
      result(FlutterError(code: "ENCODE", message: "missing required args (y/uv/width/height)", details: nil))
      return
    }
    let yRowStride = (args["yRowStride"] as? Int) ?? width
    let uvRowStride = (args["uvRowStride"] as? Int) ?? width
    let uvPixelStride = (args["uvPixelStride"] as? Int) ?? 2
    let rotation = (args["rotation"] as? Int) ?? 0
    let quality = (args["quality"] as? Int) ?? 70

    do {
      let jpeg = try encode(
        y: yData, uv: uvData, width: width, height: height,
        yRowStride: yRowStride, uvRowStride: uvRowStride, uvPixelStride: uvPixelStride,
        rotation: rotation, quality: quality)
      result(FlutterStandardTypedData(bytes: jpeg))
    } catch {
      result(FlutterError(code: "ENCODE", message: "\(error)", details: nil))
    }
  }

  private enum EncodeError: Error {
    case pixelBufferCreationFailed
    case cgImageCreationFailed
    case jpegEncodingFailed
  }

  private func encode(
    y: Data, uv: Data, width: Int, height: Int,
    yRowStride: Int, uvRowStride: Int, uvPixelStride: Int,
    rotation: Int, quality: Int
  ) throws -> Data {
    var pixelBuffer: CVPixelBuffer?
    let attrs: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary]
    let status = CVPixelBufferCreate(
      kCFAllocatorDefault, width, height,
      kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
      attrs as CFDictionary, &pixelBuffer)
    guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
      throw EncodeError.pixelBufferCreationFailed
    }

    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

    // Plane 0: Y, one byte per pixel.
    if let dst = CVPixelBufferGetBaseAddressOfPlane(buffer, 0) {
      let dstStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
      y.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
        guard let base = src.baseAddress else { return }
        for row in 0..<height {
          memcpy(dst + row * dstStride, base + row * yRowStride, width)
        }
      }
    }

    // Plane 1: interleaved Cb/Cr, half resolution. Source is already the
    // exact biplanar layout the destination wants -- no channel reordering
    // needed (unlike Android's Y+U+V -> NV21 repack).
    if let dst = CVPixelBufferGetBaseAddressOfPlane(buffer, 1) {
      let dstStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1)
      let uvWidthBytes = (width / 2) * 2
      let uvHeight = height / 2
      uv.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
        guard let base = src.baseAddress else { return }
        for row in 0..<uvHeight {
          if uvPixelStride == 2 {
            memcpy(dst + row * dstStride, base + row * uvRowStride, uvWidthBytes)
          } else {
            // Defensive fallback for an unexpected (non-tightly-packed) pixel
            // stride -- normal biplanar 4:2:0 frames always use stride 2.
            let dstRow = dst + row * dstStride
            let srcRow = base + row * uvRowStride
            for col in 0..<(width / 2) {
              dstRow.storeBytes(of: srcRow.load(fromByteOffset: col * uvPixelStride, as: UInt8.self), toByteOffset: col * 2, as: UInt8.self)
              dstRow.storeBytes(of: srcRow.load(fromByteOffset: col * uvPixelStride + 1, as: UInt8.self), toByteOffset: col * 2 + 1, as: UInt8.self)
            }
          }
        }
      }
    }

    // iOS delivers the camera buffer in landscape sensor orientation, and
    // camera_avfoundation reports sensorOrientation as 0 (the Dart-passed
    // `rotation` is therefore unreliable here). Empirically verified on
    // device against the captured frame: a 90° CCW rotation (.left) makes
    // the front- and back-camera buffers upright portrait, matching the
    // orientation the classifier was trained on. Fix it here rather than
    // trusting the passed value.
    _ = rotation
    let ciImage = CIImage(cvPixelBuffer: buffer).oriented(.left)
    guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
      throw EncodeError.cgImageCreationFailed
    }

    let outData = NSMutableData()
    guard let dest = CGImageDestinationCreateWithData(outData, "public.jpeg" as CFString, 1, nil) else {
      throw EncodeError.jpegEncodingFailed
    }
    let options: [CFString: Any] = [kCGImageDestinationLossyCompressionQuality: Double(quality) / 100.0]
    CGImageDestinationAddImage(dest, cgImage, options as CFDictionary)
    guard CGImageDestinationFinalize(dest) else {
      throw EncodeError.jpegEncodingFailed
    }
    return outData as Data
  }

}

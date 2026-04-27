import 'dart:typed_data';
import 'package:hand_landmarker/hand_landmarker.dart';

/// Converts raw MediaPipe hand detections into the 126-D feature vector
/// that matches the PSL model's training format exactly.
///
/// Training format (from psl_words_v2.py):
///   [left_hand_xyz(63), right_hand_xyz(63)]
///   where "left" = anatomical left (signer's left hand)
///
/// Since hand_landmarker package doesn't expose handedness, we infer it
/// from the wrist x-position (landmark[0].x):
///   - Back camera, signer facing lens:
///     signer's LEFT hand appears on the RIGHT of the image → higher x
///     signer's RIGHT hand appears on the LEFT of the image → lower x
class FeatureExtractor {
  static const int kHandCount = 2;
  static const int kLandmarksPerHand = 21;
  static const int kCoordsPerLandmark = 3;
  static const int kFrameDim = kHandCount * kLandmarksPerHand * kCoordsPerLandmark; // 126

  /// Extract and normalize a 126-D vector from detected hands.
  /// Returns Float32List(126) — zeros for absent hands.
  static Float32List extractAndNormalize(List<Hand> hands) {
    final raw = _extractRaw(hands);
    return _normalize(raw);
  }

  static Float32List _extractRaw(List<Hand> hands) {
    final out = Float32List(kFrameDim); // zeros = absent hand

    if (hands.isEmpty) return out;

    Hand? leftHand;   // anatomical left (higher wrist x in back-camera frame)
    Hand? rightHand;  // anatomical right (lower wrist x)

    if (hands.length == 1) {
      final wristX = hands[0].landmarks[0].x;
      if (wristX >= 0.5) {
        leftHand = hands[0];
      } else {
        rightHand = hands[0];
      }
    } else {
      // Sort by wrist x descending; first = higher x = anatomical left
      final sorted = List<Hand>.from(hands)
        ..sort((a, b) => b.landmarks[0].x.compareTo(a.landmarks[0].x));
      leftHand  = sorted[0];
      rightHand = sorted[1];
    }

    _fillHand(out, 0,  leftHand);   // indices 0..62
    _fillHand(out, 63, rightHand);  // indices 63..125

    return out;
  }

  static void _fillHand(Float32List buf, int offset, Hand? hand) {
    if (hand == null) return;
    final lm = hand.landmarks;
    for (int i = 0; i < kLandmarksPerHand && i < lm.length; i++) {
      final base = offset + i * kCoordsPerLandmark;
      buf[base]     = lm[i].x.toDouble();
      buf[base + 1] = lm[i].y.toDouble();
      buf[base + 2] = lm[i].z.toDouble();
    }
  }

  /// Per-hand wrist-centred + max-abs normalization.
  /// Identical to normalize_frame() in psl_words_v2.py.
  static Float32List _normalize(Float32List frame) {
    final f = Float32List.fromList(frame);
    for (int h = 0; h < kHandCount; h++) {
      final start = h * kLandmarksPerHand * kCoordsPerLandmark;
      // Check if this hand is all zeros (absent)
      bool allZero = true;
      for (int i = start; i < start + kLandmarksPerHand * kCoordsPerLandmark; i++) {
        if (f[i] != 0.0) { allZero = false; break; }
      }
      if (allZero) continue;

      // Subtract wrist (landmark 0)
      final wx = f[start];
      final wy = f[start + 1];
      final wz = f[start + 2];
      for (int i = 0; i < kLandmarksPerHand; i++) {
        final base = start + i * kCoordsPerLandmark;
        f[base]     -= wx;
        f[base + 1] -= wy;
        f[base + 2] -= wz;
      }

      // Divide by max(abs)
      double maxAbs = 0.0;
      for (int i = start; i < start + kLandmarksPerHand * kCoordsPerLandmark; i++) {
        final v = f[i].abs();
        if (v > maxAbs) maxAbs = v;
      }
      if (maxAbs > 0) {
        for (int i = start; i < start + kLandmarksPerHand * kCoordsPerLandmark; i++) {
          f[i] /= maxAbs;
        }
      }
    }
    return f;
  }

  /// Mean-squared diff of two consecutive normalized frames (motion gate).
  static double computeMotion(Float32List? prev, Float32List curr) {
    if (prev == null) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < curr.length; i++) {
      final d = curr[i] - prev[i];
      sum += d * d;
    }
    return sum / curr.length;
  }
}

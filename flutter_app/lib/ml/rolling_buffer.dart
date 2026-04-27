import 'dart:typed_data';
import 'feature_extractor.dart';

/// Circular rolling buffer of fixed capacity [kCapacity] frames,
/// each of length [FeatureExtractor.kFrameDim] (126 floats).
///
/// Mirrors the RollingBuffer logic in the Python backend:
///   - Slides on every push; always holds the most recent frames.
///   - snapshot() returns a flat Float32List of shape [capacity * frameDim].
class RollingBuffer {
  static const int kCapacity = 60;

  final _data = Float32List(kCapacity * FeatureExtractor.kFrameDim);
  int _fill = 0;

  bool get isFull => _fill >= kCapacity;
  int  get fillCount => _fill;

  void push(Float32List frame) {
    assert(frame.length == FeatureExtractor.kFrameDim);
    // Shift existing data left by one slot
    if (_fill == kCapacity) {
      _data.setRange(
        0,
        (kCapacity - 1) * FeatureExtractor.kFrameDim,
        _data,
        FeatureExtractor.kFrameDim,
      );
    } else {
      _fill++;
    }
    // Write the new frame into the last slot
    final offset = (_fill - 1) * FeatureExtractor.kFrameDim;
    _data.setRange(offset, offset + FeatureExtractor.kFrameDim, frame);
  }

  /// Returns a copy of the current window shaped [fill * frameDim].
  /// Callers use this as the model input (reshaped to [1, 60, 126]).
  Float32List snapshot() {
    if (!isFull) {
      // Pad with zeros at the front so we always return kCapacity frames
      final padded = Float32List(kCapacity * FeatureExtractor.kFrameDim);
      final usedStart = (kCapacity - _fill) * FeatureExtractor.kFrameDim;
      padded.setRange(usedStart, kCapacity * FeatureExtractor.kFrameDim, _data);
      return padded;
    }
    return Float32List.fromList(_data);
  }

  void reset() {
    _data.fillRange(0, _data.length, 0.0);
    _fill = 0;
  }
}

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import '../data/signs.dart';
import 'hand_landmarker_native.dart';
import 'prediction.dart';

// Constants mirror src/inference/psl_words_v2.py exactly so mobile predictions
// match the desktop reference. Drift here = drift in accuracy.
class SignPipeline {
  static const int _kFrameDim = 126;
  static const int _kBufCap = 60;
  static const int _kStride = 3;                 // STRIDE_FRAMES
  static const double _kCommitConf = 0.70;       // COMMIT_CONF_DEFAULT
  static const double _kEmaAlpha = 0.60;         // EMA_INFER_ALPHA
  static const double _kCooldownSec = 0.8;       // COOLDOWN_SECONDS
  static const double _kMotionVarMin = 1e-4;     // MOTION_VAR_MIN
  static const double _kSigningQuietSec = 1.2;   // SIGNING→IDLE timeout

  static const MethodChannel _channel = MethodChannel('psl/tflite');

  bool _ready = false;
  bool get isReady => _ready;
  int _numClasses = 0;

  // Rolling buffer (circular)
  final Float32List _buf = Float32List(_kBufCap * _kFrameDim);
  int _wIdx = 0;
  int _fill = 0;

  // FSM
  SignState _state = SignState.idle;
  DateTime _stateSince = DateTime.now();
  int _framesSinceInfer = 0;
  Float32List? _emaProbs;
  DateTime? _cooldownUntil;

  // Reusable buffers
  final Float32List _raw = Float32List(_kFrameDim);
  final Float32List _norm = Float32List(_kFrameDim);
  final Float32List _prevNorm = Float32List(_kFrameDim);
  bool _hasPrevNorm = false;
  double _lastMotion = 0.0;
  final Float32List _snap = Float32List(_kBufCap * _kFrameDim);

  // Inference is async (MethodChannel) — guard against re-entry.
  bool _inferInFlight = false;

  Prediction? _last;

  int get bufferFill => _fill;
  int get bufferCapacity => _kBufCap;

  Future<void> init() async {
    final classes = await _channel.invokeMethod<int>('load', {
      'assetPath': 'assets/models/psl_word_classifier.tflite',
    });
    _numClasses = classes ?? 0;
    _ready = true;
    debugPrint('PSL: native TFLite ready, classes=$_numClasses, buffer=$_kBufCap frames');
  }

  Future<void> dispose() async {
    _ready = false;
    try {
      await _channel.invokeMethod('dispose');
    } catch (_) {}
  }

  void reset() {
    _buf.fillRange(0, _buf.length, 0);
    _wIdx = 0;
    _fill = 0;
    _enterState(SignState.idle, DateTime.now());
    _framesSinceInfer = 0;
    _emaProbs = null;
    _cooldownUntil = null;
    _hasPrevNorm = false;
    _lastMotion = 0.0;
    _last = null;
  }

  void _enterState(SignState s, DateTime now) {
    if (_state == s) return;
    _state = s;
    _stateSince = now;
  }

  /// Process a frame. Returns the latest prediction synchronously; inference
  /// itself runs asynchronously in the background and updates [_last] when done.
  Prediction process(List<NativeHand> hands) {
    if (!_ready) return Prediction.idle;
    final hasHands = hands.isNotEmpty;
    final now = DateTime.now();

    // Match psl_words_v2.py:1493 — always extract + push, even when no hands
    // (frame_from_mediapipe returns zeros). Skipping pushes makes the 60-frame
    // window span a longer wall-clock time than the model was trained on.
    if (hasHands) {
      _extractFeatures(hands);
    } else {
      _raw.fillRange(0, _raw.length, 0);
    }
    _normalizeFeatures();
    _lastMotion = _computeMotion();

    final off = _wIdx * _kFrameDim;
    for (int i = 0; i < _kFrameDim; i++) {
      _buf[off + i] = _norm[i];
    }
    _wIdx = (_wIdx + 1) % _kBufCap;
    if (_fill < _kBufCap) _fill++;
    for (int i = 0; i < _kFrameDim; i++) {
      _prevNorm[i] = _norm[i];
    }
    _hasPrevNorm = true;
    _framesSinceInfer++;

    _updateFsm(hasHands, now);

    // Only SIGNING triggers fresh inference (matches psl_words_v2.py:1510).
    if (_state == SignState.signing &&
        _fill >= _kBufCap &&
        _framesSinceInfer >= _kStride &&
        _lastMotion >= _kMotionVarMin &&
        !_inferInFlight) {
      _framesSinceInfer = 0;
      _enterState(SignState.predicting, now);
      _kickInference();
    }

    return _last ?? Prediction(
      label: '', english: '\u2014', urdu: '\u2014',
      confidence: 0, state: _state, committed: false, hasHands: hasHands,
    );
  }

  // ── Feature extraction ──────────────────────────────────────────────────

  void _extractFeatures(List<NativeHand> hands) {
    _raw.fillRange(0, _raw.length, 0);

    // psl_words_v2.py:323 sets HANDS_INVERT_HANDEDNESS=True: MediaPipe is trained
    // on selfie-flipped frames, but we feed raw frames, so its label is reversed
    // relative to anatomy. Label "Right" → signer's anatomical LEFT → first 63
    // dims; "Left" → second 63 dims.
    for (final h in hands) {
      final offset = h.isRightLabel ? 0 : 63;
      bool slotFilled = false;
      for (int i = 0; i < 63; i++) {
        if (_raw[offset + i] != 0) { slotFilled = true; break; }
      }
      if (slotFilled) continue;
      for (int i = 0; i < 63; i++) {
        _raw[offset + i] = h.coords[i];
      }
    }
  }

  void _normalizeFeatures() {
    for (int i = 0; i < _kFrameDim; i++) {
      _norm[i] = _raw[i];
    }

    for (int h = 0; h < 2; h++) {
      final s = h * 21 * 3;
      final e = s + 21 * 3;

      bool allZero = true;
      for (int i = s; i < e; i++) {
        if (_norm[i] != 0) { allZero = false; break; }
      }
      if (allZero) continue;

      final wx = _norm[s], wy = _norm[s + 1], wz = _norm[s + 2];
      for (int i = 0; i < 21; i++) {
        final b = s + i * 3;
        _norm[b] -= wx;
        _norm[b + 1] -= wy;
        _norm[b + 2] -= wz;
      }

      double mx = 0;
      for (int i = s; i < e; i++) {
        final v = _norm[i].abs();
        if (v > mx) mx = v;
      }
      if (mx > 0) {
        for (int i = s; i < e; i++) {
          _norm[i] /= mx;
        }
      }
    }
  }

  double _computeMotion() {
    if (!_hasPrevNorm) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < _kFrameDim; i++) {
      final d = _norm[i] - _prevNorm[i];
      sum += d * d;
    }
    return sum / _kFrameDim;
  }

  // ── FSM (matches psl_words_v2.py:1499-1529) ─────────────────────────────

  void _updateFsm(bool hasHands, DateTime now) {
    switch (_state) {
      case SignState.idle:
        if (hasHands && _lastMotion > _kMotionVarMin) {
          _enterState(SignState.signing, now);
        }
      case SignState.signing:
        // Only return to IDLE if quiet (no hands, no motion) for >1.2s.
        if (!hasHands &&
            _lastMotion < _kMotionVarMin &&
            now.difference(_stateSince).inMilliseconds >
                (_kSigningQuietSec * 1000).round()) {
          _enterState(SignState.idle, now);
        }
      case SignState.predicting:
        // Stays here until inference callback resolves it.
        break;
      case SignState.committed:
        _cooldownUntil = now.add(
          Duration(milliseconds: (_kCooldownSec * 1000).round()),
        );
        _enterState(SignState.cooldown, now);
      case SignState.cooldown:
        if (hasHands) {
          _cooldownUntil = now.add(
            Duration(milliseconds: (_kCooldownSec * 1000).round()),
          );
        }
        if (_cooldownUntil != null &&
            now.isAfter(_cooldownUntil!) &&
            !hasHands) {
          _enterState(SignState.idle, now);
          _buf.fillRange(0, _buf.length, 0);
          _wIdx = 0;
          _fill = 0;
          _emaProbs = null;
        }
    }
  }

  // ── TFLite inference (native) ───────────────────────────────────────────

  void _kickInference() {
    _inferInFlight = true;
    if (_fill < _kBufCap) {
      final pad = (_kBufCap - _fill) * _kFrameDim;
      _snap.fillRange(0, pad, 0);
      for (int i = 0; i < _fill * _kFrameDim; i++) {
        _snap[pad + i] = _buf[i];
      }
    } else {
      final sOff = _wIdx * _kFrameDim;
      final tail = _kBufCap * _kFrameDim - sOff;
      for (int i = 0; i < tail; i++) {
        _snap[i] = _buf[sOff + i];
      }
      for (int i = 0; i < sOff; i++) {
        _snap[tail + i] = _buf[i];
      }
    }

    final bytes = _snap.buffer.asUint8List(_snap.offsetInBytes, _snap.lengthInBytes);

    _channel.invokeMethod<Float64List>('runInference', {
      'features': bytes,
    }).then((probs) {
      _inferInFlight = false;
      if (probs == null) return;
      _onProbs(probs);
    }).catchError((e) {
      _inferInFlight = false;
      debugPrint('PSL: inference error: $e');
    });
  }

  void _onProbs(Float64List probs) {
    if (_emaProbs == null) {
      _emaProbs = Float32List(probs.length);
      for (int i = 0; i < probs.length; i++) {
        _emaProbs![i] = probs[i];
      }
    } else {
      for (int i = 0; i < probs.length; i++) {
        _emaProbs![i] = _kEmaAlpha * probs[i] + (1 - _kEmaAlpha) * _emaProbs![i];
      }
    }

    int topIdx = 0;
    double topConf = _emaProbs![0];
    for (int i = 1; i < _emaProbs!.length; i++) {
      if (_emaProbs![i] > topConf) {
        topConf = _emaProbs![i];
        topIdx = i;
      }
    }

    final label = topIdx < kClassLabels.length ? kClassLabels[topIdx] : '';
    debugPrint('PSL: $label ${(topConf * 100).toStringAsFixed(1)}% fill=$_fill motion=${_lastMotion.toStringAsExponential(2)}');
    final now = DateTime.now();
    _last = _buildPrediction(label, topConf, now);
    // Drop back to SIGNING so subsequent windows can fire (matches py:1522).
    if (!(_last?.committed ?? false) && _state == SignState.predicting) {
      _enterState(SignState.signing, now);
    }
  }

  Prediction _buildPrediction(String label, double confidence, DateTime now) {
    final sign = findSign(label);
    final eng = sign?.english ?? label;
    final urdu = sign?.urdu ?? label;

    bool committed = false;
    if (!kIdleClasses.contains(label) &&
        confidence >= _kCommitConf &&
        _state != SignState.cooldown) {
      committed = true;
      _enterState(SignState.committed, now);
    }

    return Prediction(
      label: label,
      english: eng,
      urdu: urdu,
      confidence: confidence,
      state: _state,
      committed: committed,
      hasHands: true,
    );
  }
}

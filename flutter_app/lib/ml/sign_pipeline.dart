import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:hand_landmarker/hand_landmarker.dart';

import '../data/signs.dart';
import 'prediction.dart';

class SignPipeline {
  static const int _kFrameDim = 126;
  static const int _kBufCap = 60;      // model requires exactly 60 frames
  static const int _kStride = 2;       // infer every 2 frames
  static const double _kCommitConf = 0.65;
  static const double _kEmaAlpha = 0.55;
  static const double _kCooldownSec = 0.8;
  static const int _kMissGrace = 5;    // tolerate 5 missed frames before reset

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
  int _framesSinceInfer = 0;
  Float32List? _lastFrame;
  Float32List? _emaProbs;
  DateTime? _cooldownUntil;
  int _missCount = 0;

  // Reusable buffers
  final Float32List _raw = Float32List(_kFrameDim);
  final Float32List _norm = Float32List(_kFrameDim);
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
    _state = SignState.idle;
    _framesSinceInfer = 0;
    _lastFrame = null;
    _emaProbs = null;
    _cooldownUntil = null;
    _missCount = 0;
    _last = null;
  }

  /// Process a frame. Returns the latest prediction synchronously; inference
  /// itself runs asynchronously in the background and updates [_last] when done.
  Prediction process(List<Hand> hands) {
    if (!_ready) return Prediction.idle;

    if (hands.isEmpty) {
      _missCount++;
      if (_missCount >= _kMissGrace) {
        reset();
        return Prediction.idle;
      }
      return _last ?? Prediction.idle;
    }
    _missCount = 0;

    _extractFeatures(hands);
    _normalizeFeatures();
    _lastFrame = Float32List.fromList(_norm);

    final off = _wIdx * _kFrameDim;
    for (int i = 0; i < _kFrameDim; i++) {
      _buf[off + i] = _norm[i];
    }
    _wIdx = (_wIdx + 1) % _kBufCap;
    if (_fill < _kBufCap) _fill++;
    _framesSinceInfer++;

    final now = DateTime.now();
    _updateFsm(true, now);

    if ((_state == SignState.signing || _state == SignState.predicting) &&
        _fill >= _kBufCap &&
        _framesSinceInfer >= _kStride &&
        !_inferInFlight) {
      _framesSinceInfer = 0;
      _kickInference();
    }

    return _last ?? Prediction(
      label: '', english: '\u2014', urdu: '\u2014',
      confidence: 0, state: _state, committed: false, hasHands: true,
    );
  }

  // ── Feature extraction ──────────────────────────────────────────────────

  void _extractFeatures(List<Hand> hands) {
    _raw.fillRange(0, _raw.length, 0);

    if (hands.length == 1) {
      final wristX = hands[0].landmarks[0].x;
      if (wristX >= 0.5) {
        _fillHand(0, hands[0]);
      } else {
        _fillHand(63, hands[0]);
      }
    } else {
      final sorted = List<Hand>.from(hands)
        ..sort((a, b) => b.landmarks[0].x.compareTo(a.landmarks[0].x));
      _fillHand(0, sorted[0]);
      _fillHand(63, sorted[1]);
    }
  }

  void _fillHand(int offset, Hand hand) {
    final lms = hand.landmarks;
    for (int i = 0; i < 21 && i < lms.length; i++) {
      final b = offset + i * 3;
      _raw[b] = lms[i].x;
      _raw[b + 1] = lms[i].y;
      _raw[b + 2] = lms[i].z;
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

  // ── FSM ─────────────────────────────────────────────────────────────────

  void _updateFsm(bool hasHands, DateTime now) {
    switch (_state) {
      case SignState.idle:
        if (hasHands) _state = SignState.signing;
      case SignState.signing:
        if (!hasHands) {
          _state = SignState.idle;
        } else {
          _state = SignState.predicting;
        }
      case SignState.predicting:
        break;
      case SignState.committed:
        _cooldownUntil = now.add(
          Duration(milliseconds: (_kCooldownSec * 1000).round()),
        );
        _state = SignState.cooldown;
      case SignState.cooldown:
        if (_cooldownUntil != null && now.isAfter(_cooldownUntil!)) {
          _state = SignState.idle;
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
    // Snapshot circular buffer into linear order [oldest..newest].
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

    // Pass features as raw bytes (native-order float32) — fastest MethodChannel
    // path: typed-data lists are zero-copy on the platform side.
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
    debugPrint('PSL: $label ${(topConf * 100).toStringAsFixed(1)}% fill=$_fill');
    _last = _buildPrediction(label, topConf, DateTime.now());
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
      _state = SignState.committed;
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

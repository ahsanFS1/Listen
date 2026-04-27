import 'dart:typed_data';
import 'package:hand_landmarker/hand_landmarker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import '../data/signs.dart';
import 'feature_extractor.dart';
import 'rolling_buffer.dart';

enum SignState { idle, signing, predicting, committed, cooldown }

class Prediction {
  final String label;
  final String english;
  final String urdu;
  final double confidence;
  final SignState state;
  final bool committed;
  final bool hasHands;

  const Prediction({
    required this.label,
    required this.english,
    required this.urdu,
    required this.confidence,
    required this.state,
    required this.committed,
    required this.hasHands,
  });

  static const Prediction idle = Prediction(
    label: '', english: '—', urdu: '—',
    confidence: 0, state: SignState.idle,
    committed: false, hasHands: false,
  );
}

/// Full on-device PSL sign recognition pipeline.
///
/// Mirrors the Python pipeline exactly:
///   MediaPipe hands → 126-D vector → normalize → rolling buffer (60 frames)
///   → motion gate → TFLite inference (every 3 frames) → EMA smoothing
///   → FSM (IDLE→SIGNING→PREDICTING→COMMITTED→COOLDOWN)
class SignPipeline {
  static const int    _kStride        = 3;
  static const double _kCommitConf    = 0.70;
  static const double _kMotionMin     = 1e-4;
  static const double _kEmaAlpha      = 0.60;
  static const double _kCooldownSec   = 0.8;

  Interpreter? _interpreter;
  bool _ready = false;

  final _buffer = RollingBuffer();
  SignState _state = SignState.idle;
  int _framesSinceInfer = 0;
  Float32List? _lastFrame;
  Float32List? _emaProbs;
  DateTime? _cooldownUntil;

  Prediction? _last;

  bool get isReady => _ready;

  // ── init ─────────────────────────────────────────────────────────────────

  Future<void> init() async {
    try {
      final options = InterpreterOptions();
      _interpreter = await Interpreter.fromAsset(
        'assets/models/psl_word_classifier.tflite',
        options: options,
      );
      _interpreter!.resizeInputTensor(0, [1, RollingBuffer.kCapacity, FeatureExtractor.kFrameDim]);
      _interpreter!.allocateTensors();
      _ready = true;
    } catch (e) {
      _ready = false;
      rethrow;
    }
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _ready = false;
  }

  // ── main entry point ─────────────────────────────────────────────────────

  Prediction process(List<Hand> hands) {
    if (!_ready) return Prediction.idle;

    final hasHands = hands.isNotEmpty;

    if (!hasHands) {
      _buffer.reset();
      _lastFrame = null;
      _emaProbs  = null;
      _state     = SignState.idle;
      _last      = Prediction.idle;
      return Prediction.idle;
    }

    final frame  = FeatureExtractor.extractAndNormalize(hands);
    final motion = FeatureExtractor.computeMotion(_lastFrame, frame);
    _lastFrame    = frame;
    _buffer.push(frame);
    _framesSinceInfer++;

    // FSM transitions
    final now = DateTime.now();
    _updateFsm(hasHands, motion, now);

    // Run inference when appropriate
    if (_state == SignState.signing || _state == SignState.predicting) {
      if (_buffer.isFull &&
          _framesSinceInfer >= _kStride &&
          motion > _kMotionMin) {
        final result = _runInference();
        _framesSinceInfer = 0;
        if (result != null) {
          final pred = _buildPrediction(result, now);
          _last = pred;
          return pred;
        }
      }
    }

    return _last ?? Prediction(
      label: '', english: '—', urdu: '—',
      confidence: 0, state: _state,
      committed: false, hasHands: true,
    );
  }

  void reset() {
    _buffer.reset();
    _lastFrame = null;
    _emaProbs  = null;
    _state     = SignState.idle;
    _last      = null;
    _framesSinceInfer = 0;
    _cooldownUntil    = null;
  }

  // ── FSM ──────────────────────────────────────────────────────────────────

  void _updateFsm(bool hasHands, double motion, DateTime now) {
    switch (_state) {
      case SignState.idle:
        if (hasHands && motion > _kMotionMin) {
          _state = SignState.signing;
        }
      case SignState.signing:
        if (!hasHands && motion < _kMotionMin) {
          _state = SignState.idle;
        } else {
          _state = SignState.predicting;
        }
      case SignState.predicting:
        // stays until inference commits or hands disappear
        break;
      case SignState.committed:
        _cooldownUntil = now.add(
          Duration(milliseconds: (_kCooldownSec * 1000).round()),
        );
        _state = SignState.cooldown;
      case SignState.cooldown:
        if (_cooldownUntil != null && now.isAfter(_cooldownUntil!)) {
          _state = SignState.idle;
          _buffer.reset();
          _emaProbs = null;
        }
    }
  }

  // ── inference ─────────────────────────────────────────────────────────────

  ({String label, double confidence})? _runInference() {
    final interp = _interpreter;
    if (interp == null) return null;

    final window = _buffer.snapshot();
    // Shape [1, 60, 126]
    final input = [
      List.generate(RollingBuffer.kCapacity, (t) {
        return List.generate(FeatureExtractor.kFrameDim, (f) {
          return window[t * FeatureExtractor.kFrameDim + f];
        });
      }),
    ];

    final output = [List.filled(kClassLabels.length, 0.0)];
    interp.run(input, output);

    final probs = Float32List.fromList(output[0].cast<double>());

    // EMA smoothing
    if (_emaProbs == null) {
      _emaProbs = probs;
    } else {
      for (int i = 0; i < probs.length; i++) {
        _emaProbs![i] = _kEmaAlpha * probs[i] + (1 - _kEmaAlpha) * _emaProbs![i];
      }
    }

    // Argmax
    int topIdx = 0;
    double topConf = _emaProbs![0];
    for (int i = 1; i < _emaProbs!.length; i++) {
      if (_emaProbs![i] > topConf) {
        topConf = _emaProbs![i];
        topIdx  = i;
      }
    }

    return (label: kClassLabels[topIdx], confidence: topConf);
  }

  Prediction _buildPrediction(
    ({String label, double confidence}) result,
    DateTime now,
  ) {
    final label  = result.label;
    final conf   = result.confidence;
    final sign   = findSign(label);
    final eng    = sign?.english ?? _prettify(label);
    final urdu   = sign?.urdu    ?? label;
    final isIdle = kIdleClasses.contains(label);

    bool committed = false;
    if (!isIdle && conf >= _kCommitConf && _state != SignState.cooldown) {
      committed = true;
      _state    = SignState.committed;
    }

    return Prediction(
      label:     label,
      english:   eng,
      urdu:      urdu,
      confidence: conf,
      state:     _state,
      committed: committed,
      hasHands:  true,
    );
  }

  static String _prettify(String id) =>
      id.replaceAll('_', ' ')
        .split(' ')
        .map((w) => w.isEmpty ? w : '${w[0].toUpperCase()}${w.substring(1)}')
        .join(' ');
}

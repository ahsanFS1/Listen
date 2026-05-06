import 'dart:async';
import 'dart:math';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../data/signs.dart';
import '../ml/prediction.dart';
import '../ml/sign_client.dart';
import '../ml/yuv_jpeg.dart';
import '../services/progress_service.dart';
import '../services/settings_service.dart';
import '../theme/app_colors.dart';
import '../widgets/state_pill.dart';

/// 5-question gesture quiz: the app picks 5 signs at random, the user
/// performs each one in front of the camera, the inference server
/// replies with the recognised label. A question passes when the
/// committed label matches the target. Reuses the same WebSocket
/// pipeline as the Translate tab — no separate model needed.
const int _kQuizCount = 5;

class QuizScreen extends StatefulWidget {
  const QuizScreen({super.key});

  @override
  State<QuizScreen> createState() => _QuizScreenState();
}

class _QuizScreenState extends State<QuizScreen> {
  late List<SignInfo> _questions;
  int _idx = 0;
  int _score = 0;
  String? _lastFeedback; // 'correct' | 'wrong' | null
  bool _finished = false;

  CameraController? _camera;
  bool _encoding = false;
  bool _cameraReady = false;

  late SignClient _client;
  bool _serverReady = false;
  String? _connectError;
  StreamSubscription<Prediction>? _commitSub;
  StreamSubscription<String?>? _errSub;
  Prediction _liveTop = Prediction.idle;
  StreamSubscription<Prediction>? _predSub;

  // Match progress: while the live prediction matches the current
  // target with sufficient confidence, this fills towards 1.0; on
  // reaching 1.0 we count the answer as correct without waiting for
  // the server's strict commit (which can take longer than expected).
  double _matchProgress = 0.0;
  DateTime? _lastPredAt;
  static const int _matchTargetMs = 1500;
  static const double _matchMinConf = 0.55;

  @override
  void initState() {
    super.initState();
    final rng = Random();
    final pool = List<SignInfo>.from(kSigns)..shuffle(rng);
    _questions = pool.take(_kQuizCount).toList();
    _client = SignClient(
      url: SettingsService.instance.serverUrl,
      mode: SignMode.words,
    );
    WidgetsBinding.instance.addPostFrameCallback((_) => _start());
  }

  Future<void> _start() async {
    try {
      await _client.connect();
      _serverReady = true;
    } catch (e) {
      _connectError = '$e';
    }
    _commitSub = _client.commits.listen(_onCommit);
    _predSub = _client.predictions.listen((p) {
      if (!mounted) return;
      _updateMatchProgress(p);
      setState(() => _liveTop = p);
    });
    _errSub = _client.errors.listen((err) {
      if (!mounted) return;
      setState(() {
        _serverReady = err == null ? _serverReady : false;
        _connectError = err;
      });
    });
    await _startCamera();
    if (mounted) setState(() {});
  }

  Future<void> _startCamera() async {
    final cams = await availableCameras();
    if (cams.isEmpty) return;
    final desc = cams.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.front,
      orElse: () => cams.first,
    );
    final ctrl = CameraController(
      desc,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );
    await ctrl.initialize();
    await ctrl.startImageStream(_onCameraFrame);
    if (!mounted) return;
    setState(() {
      _camera = ctrl;
      _cameraReady = true;
    });
  }

  void _onCameraFrame(CameraImage image) async {
    if (_encoding || !_client.isReady || _finished) return;
    _encoding = true;
    try {
      final rotation = _camera?.description.sensorOrientation ?? 0;
      final jpeg =
          await YuvJpeg.encode(image, rotation: rotation, quality: 70);
      if (!mounted || jpeg == null) return;
      _client.sendFrame(jpeg);
    } catch (e) {
      debugPrint('quiz: encode error: $e');
    } finally {
      _encoding = false;
    }
  }

  void _updateMatchProgress(Prediction p) {
    if (_finished || _lastFeedback != null || _idx >= _questions.length) {
      _lastPredAt = DateTime.now();
      return;
    }
    final now = DateTime.now();
    final dtMs =
        _lastPredAt == null ? 0 : now.difference(_lastPredAt!).inMilliseconds;
    _lastPredAt = now;
    final target = _questions[_idx];
    final matches = p.label == target.id && p.confidence >= _matchMinConf;
    if (matches) {
      _matchProgress =
          (_matchProgress + dtMs / _matchTargetMs).clamp(0.0, 1.0);
      if (_matchProgress >= 1.0) {
        _markCorrect();
      }
    } else if (p.label.isNotEmpty && p.confidence > 0.5) {
      // Different sign with conviction → decay quickly
      _matchProgress =
          (_matchProgress - dtMs / (_matchTargetMs * 1.5)).clamp(0.0, 1.0);
    } else {
      // Idle / no hands → mild decay
      _matchProgress =
          (_matchProgress - dtMs / (_matchTargetMs * 4)).clamp(0.0, 1.0);
    }
  }

  void _markCorrect() {
    if (_lastFeedback != null) return;
    final target = _questions[_idx];
    setState(() {
      _lastFeedback = 'correct';
      _score += 1;
      _matchProgress = 1.0;
    });
    ProgressService.instance.markLearned(target.id);
    Future.delayed(const Duration(milliseconds: 900), _advance);
  }

  void _onCommit(Prediction p) {
    if (!mounted || _finished || _idx >= _questions.length) return;
    final target = _questions[_idx];
    final correct = p.label == target.id;
    if (correct) {
      _markCorrect();
      return;
    }
    // Wrong commit — show "Try Again" briefly without consuming the question.
    setState(() => _lastFeedback = 'wrong');
    Future.delayed(const Duration(milliseconds: 1300), () {
      if (mounted) setState(() => _lastFeedback = null);
    });
  }

  void _tryAgain() {
    if (_finished) return;
    setState(() {
      _lastFeedback = null;
      _matchProgress = 0.0;
    });
  }

  void _advance() {
    if (!mounted) return;
    setState(() {
      _lastFeedback = null;
      _matchProgress = 0.0;
      if (_idx + 1 >= _questions.length) {
        _finished = true;
      } else {
        _idx += 1;
      }
    });
  }

  void _skip() {
    if (_finished) return;
    setState(() {
      _lastFeedback = null;
      _matchProgress = 0.0;
      if (_idx + 1 >= _questions.length) {
        _finished = true;
      } else {
        _idx += 1;
      }
    });
  }

  void _restart() {
    final rng = Random();
    final pool = List<SignInfo>.from(kSigns)..shuffle(rng);
    setState(() {
      _questions = pool.take(_kQuizCount).toList();
      _idx = 0;
      _score = 0;
      _finished = false;
      _lastFeedback = null;
      _matchProgress = 0.0;
    });
  }

  @override
  void dispose() {
    try { _camera?.stopImageStream(); } catch (_) {}
    _camera?.dispose();
    _commitSub?.cancel();
    _predSub?.cancel();
    _errSub?.cancel();
    _client.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      appBar: AppBar(
        title: const Text('5-Sign Quiz'),
        backgroundColor: AppColors.bgCard,
      ),
      body: SafeArea(
        child: _finished ? _buildResult() : _buildQuestion(),
      ),
    );
  }

  Widget _buildQuestion() {
    final target = _questions[_idx];
    return Column(
      children: [
        // Top progress
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
          child: Row(children: [
            Text('Question ${_idx + 1} / $_kQuizCount',
                style: const TextStyle(
                    color: AppColors.textDim, fontWeight: FontWeight.w600)),
            const Spacer(),
            Text('Score: $_score',
                style: const TextStyle(
                    color: AppColors.accent, fontWeight: FontWeight.w700)),
          ]),
        ),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(999),
            child: LinearProgressIndicator(
              value: (_idx + 1) / _kQuizCount,
              minHeight: 6,
              backgroundColor: AppColors.bgSoft,
              valueColor: const AlwaysStoppedAnimation(AppColors.accent),
            ),
          ),
        ),
        const SizedBox(height: 12),
        // Prompt card
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: AppColors.bgCard,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: AppColors.border),
            ),
            child: Row(children: [
              const Icon(Icons.front_hand, color: AppColors.accent),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('Sign this:',
                        style: TextStyle(
                            color: AppColors.textDim, fontSize: 12)),
                    const SizedBox(height: 2),
                    Text(target.english,
                        style: const TextStyle(
                            color: AppColors.text,
                            fontSize: 22,
                            fontWeight: FontWeight.w800)),
                  ],
                ),
              ),
              Text(target.urdu,
                  textDirection: TextDirection.rtl,
                  style: const TextStyle(
                      color: AppColors.accent,
                      fontSize: 22,
                      fontWeight: FontWeight.w700)),
            ]),
          ),
        ),
        const SizedBox(height: 10),
        // Match progress — fills as the user holds the correct sign
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: Row(children: [
            Icon(Icons.check_circle,
                color: _matchProgress >= 1.0
                    ? AppColors.ok
                    : (_matchProgress > 0
                        ? AppColors.accent
                        : AppColors.textDim),
                size: 16),
            const SizedBox(width: 6),
            Expanded(
              child: ClipRRect(
                borderRadius: BorderRadius.circular(999),
                child: LinearProgressIndicator(
                  value: _matchProgress,
                  minHeight: 8,
                  backgroundColor: AppColors.bgSoft,
                  valueColor: AlwaysStoppedAnimation(
                    _matchProgress >= 1.0
                        ? AppColors.ok
                        : AppColors.accent,
                  ),
                ),
              ),
            ),
            const SizedBox(width: 8),
            SizedBox(
              width: 70,
              child: Text(
                _matchProgress >= 1.0
                    ? 'MATCHED!'
                    : (_matchProgress > 0
                        ? 'matching ${(_matchProgress * 100).toStringAsFixed(0)}%'
                        : 'sign to match'),
                textAlign: TextAlign.right,
                style: TextStyle(
                  color: _matchProgress > 0
                      ? AppColors.accent
                      : AppColors.textDim,
                  fontSize: 10,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
          ]),
        ),
        const SizedBox(height: 8),
        // Status row: StatePill + hands hint + buffer progress
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: Row(children: [
            StatePill(
              state: _liveTop.state,
              hasHands: _liveTop.hasHands,
              cameraOn: _cameraReady && _serverReady,
            ),
            const SizedBox(width: 10),
            Expanded(
              child: ClipRRect(
                borderRadius: BorderRadius.circular(999),
                child: LinearProgressIndicator(
                  value: _client.bufferCapacity == 0
                      ? 0
                      : (_client.bufferFill / _client.bufferCapacity)
                          .clamp(0.0, 1.0),
                  minHeight: 6,
                  backgroundColor: AppColors.bgSoft,
                  valueColor:
                      const AlwaysStoppedAnimation(AppColors.accent),
                ),
              ),
            ),
          ]),
        ),
        const SizedBox(height: 6),
        // Confidence bar
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: Row(children: [
            const Icon(Icons.bolt, color: AppColors.textDim, size: 14),
            const SizedBox(width: 6),
            Expanded(
              child: ClipRRect(
                borderRadius: BorderRadius.circular(999),
                child: LinearProgressIndicator(
                  value: _liveTop.confidence.clamp(0.0, 1.0),
                  minHeight: 6,
                  backgroundColor: AppColors.bgSoft,
                  valueColor: AlwaysStoppedAnimation(
                    _liveTop.confidence >= 0.7
                        ? AppColors.ok
                        : (_liveTop.confidence >= 0.4
                            ? AppColors.warn
                            : AppColors.textDim),
                  ),
                ),
              ),
            ),
            const SizedBox(width: 8),
            SizedBox(
              width: 36,
              child: Text(
                '${(_liveTop.confidence * 100).toStringAsFixed(0)}%',
                textAlign: TextAlign.right,
                style: const TextStyle(
                    color: AppColors.textDim,
                    fontSize: 11,
                    fontWeight: FontWeight.w700),
              ),
            ),
          ]),
        ),
        const SizedBox(height: 10),
        // Camera
        Expanded(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: Stack(
                fit: StackFit.expand,
                children: [
                  Container(color: Colors.black),
                  if (_camera != null && _cameraReady)
                    FittedBox(
                      fit: BoxFit.cover,
                      child: SizedBox(
                        width: _camera!.value.previewSize?.height ?? 1,
                        height: _camera!.value.previewSize?.width ?? 1,
                        child: CameraPreview(_camera!),
                      ),
                    )
                  else
                    const Center(
                        child: CircularProgressIndicator(
                            color: AppColors.accent)),
                  if (_lastFeedback != null) _buildFeedbackOverlay(),
                  if (_lastFeedback == null &&
                      _cameraReady &&
                      _serverReady &&
                      !_liveTop.hasHands)
                    Positioned(
                      top: 12, left: 12, right: 12,
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 12, vertical: 8),
                        decoration: BoxDecoration(
                          color: Colors.black.withValues(alpha: 0.6),
                          borderRadius: BorderRadius.circular(10),
                        ),
                        child: const Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(Icons.front_hand,
                                color: Colors.white, size: 16),
                            SizedBox(width: 6),
                            Expanded(
                              child: Text(
                                'Show your hands to the camera',
                                style: TextStyle(
                                    color: Colors.white, fontSize: 12),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  Positioned(
                    left: 12, right: 12, bottom: 12,
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 12, vertical: 8),
                      decoration: BoxDecoration(
                        color: Colors.black.withValues(alpha: 0.55),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Row(children: [
                        Icon(_serverReady ? Icons.cloud_done : Icons.cloud_off,
                            color: _serverReady
                                ? AppColors.accent
                                : Colors.redAccent,
                            size: 16),
                        const SizedBox(width: 6),
                        Expanded(
                          child: Text(
                            _serverReady
                                ? (_liveTop.label.isEmpty
                                    ? 'Listening…'
                                    : 'Live: ${_liveTop.english} · ${(_liveTop.confidence * 100).toStringAsFixed(0)}%')
                                : (_connectError ?? 'Connecting…'),
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: const TextStyle(
                                color: Colors.white, fontSize: 12),
                          ),
                        ),
                      ]),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
        const SizedBox(height: 12),
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
          child: Row(children: [
            Expanded(
              child: OutlinedButton.icon(
                onPressed: _tryAgain,
                style: OutlinedButton.styleFrom(
                  foregroundColor: AppColors.accent,
                  side: const BorderSide(color: AppColors.accent),
                  padding: const EdgeInsets.symmetric(vertical: 12),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                icon: const Icon(Icons.refresh),
                label: const Text('Try Again'),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: OutlinedButton.icon(
                onPressed: _skip,
                style: OutlinedButton.styleFrom(
                  foregroundColor: AppColors.textDim,
                  side: const BorderSide(color: AppColors.border),
                  padding: const EdgeInsets.symmetric(vertical: 12),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                icon: const Icon(Icons.skip_next),
                label: const Text('Skip'),
              ),
            ),
          ]),
        ),
      ],
    );
  }

  Widget _buildFeedbackOverlay() {
    final correct = _lastFeedback == 'correct';
    return Container(
      color: (correct ? Colors.green : Colors.red).withValues(alpha: 0.45),
      alignment: Alignment.center,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(correct ? Icons.check_circle : Icons.cancel,
              color: Colors.white, size: 92),
          const SizedBox(height: 10),
          Text(correct ? 'Correct!' : 'Wrong sign',
              style: const TextStyle(
                  color: Colors.white,
                  fontSize: 26,
                  fontWeight: FontWeight.w800)),
          const SizedBox(height: 4),
          Text(
              correct
                  ? 'Next question…'
                  : 'Try again — keep your hand steady',
              style: const TextStyle(
                  color: Colors.white70, fontSize: 13)),
        ],
      ),
    );
  }

  Widget _buildResult() {
    final passed = _score >= 3;
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(passed ? Icons.emoji_events : Icons.replay,
                color: passed ? AppColors.accent : AppColors.textDim, size: 80),
            const SizedBox(height: 16),
            Text(passed ? 'You Passed!' : 'Keep Practicing',
                style: const TextStyle(
                    color: AppColors.text,
                    fontSize: 26,
                    fontWeight: FontWeight.w800)),
            const SizedBox(height: 8),
            Text('$_score / $_kQuizCount correct',
                style: const TextStyle(
                    color: AppColors.textDim, fontSize: 16)),
            const SizedBox(height: 24),
            Row(mainAxisAlignment: MainAxisAlignment.center, children: [
              OutlinedButton.icon(
                onPressed: () => Navigator.of(context).pop(),
                icon: const Icon(Icons.close),
                label: const Text('Done'),
                style: OutlinedButton.styleFrom(
                  foregroundColor: AppColors.text,
                  side: const BorderSide(color: AppColors.border),
                ),
              ),
              const SizedBox(width: 12),
              ElevatedButton.icon(
                onPressed: _restart,
                icon: const Icon(Icons.refresh),
                label: const Text('Retry'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppColors.accent,
                  foregroundColor: const Color(0xFF0B1020),
                ),
              ),
            ]),
          ],
        ),
      ),
    );
  }
}

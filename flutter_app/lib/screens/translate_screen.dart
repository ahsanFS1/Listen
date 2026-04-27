import 'dart:async';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';

import '../ml/detector_isolate.dart';
import '../ml/sign_pipeline.dart';
import '../theme/app_colors.dart';
import '../widgets/confidence_bar_widget.dart';
import '../widgets/state_pill.dart';

class TranslateScreen extends StatefulWidget {
  const TranslateScreen({super.key});

  @override
  State<TranslateScreen> createState() => _TranslateScreenState();
}

class _TranslateScreenState extends State<TranslateScreen> {
  // ── camera ───────────────────────────────────────────────────────────────
  CameraController? _camera;
  bool _cameraOn       = false;
  bool _cameraReady    = false;
  bool _useFrontCamera = false;

  // ── background isolate (owns HandLandmarker + SignPipeline) ───────────────
  final _detector = DetectorIsolate();
  bool _detectorReady = false;
  String? _detectorError;

  // ── prediction state ─────────────────────────────────────────────────────
  Prediction _prediction = Prediction.idle;

  // ── session history ───────────────────────────────────────────────────────
  final List<({String english, String urdu})> _history = [];
  String? _lastCommitted;

  // ── TTS ──────────────────────────────────────────────────────────────────
  final FlutterTts _tts = FlutterTts();
  bool _ttsReady = false;

  // ── show history panel ────────────────────────────────────────────────────
  bool _showHistory = false;

  @override
  void initState() {
    super.initState();
    // Defer heavy init to after first paint
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _initDetector();
      _initTts();
    });
  }

  @override
  void dispose() {
    _stopCamera();
    _detector.stop();
    _tts.stop();
    super.dispose();
  }

  // ── init ─────────────────────────────────────────────────────────────────

  Future<void> _initDetector() async {
    try {
      await _detector.start();
      _detector.onPrediction = _onPredictionReceived;
      if (mounted) setState(() => _detectorReady = true);
    } catch (e) {
      if (mounted) setState(() => _detectorError = e.toString());
    }
  }

  Future<void> _initTts() async {
    await _tts.setLanguage('ur-PK');
    await _tts.setSpeechRate(0.5);
    if (mounted) setState(() => _ttsReady = true);
  }

  // ── prediction callback (called from ReceivePort listener — main isolate) ─

  void _onPredictionReceived(PredictionMessage msg) {
    final pred = _predictionFromMessage(msg);
    if (!mounted) return;

    if (_predictionChanged(pred)) {
      setState(() => _prediction = pred);
    }

    if (pred.committed && pred.label.isNotEmpty &&
        pred.label != _lastCommitted) {
      _lastCommitted = pred.label;
      if (mounted) setState(() => _history.add((english: pred.english, urdu: pred.urdu)));
      if (_ttsReady) _tts.speak(pred.urdu);
    }
  }

  /// Convert PredictionMessage (isolate-safe plain data) → Prediction (typed)
  Prediction _predictionFromMessage(PredictionMessage m) {
    final state = SignState.values.firstWhere(
      (s) => s.name == m.state,
      orElse: () => SignState.idle,
    );
    return Prediction(
      label:      m.label,
      english:    m.english,
      urdu:       m.urdu,
      confidence: m.confidence,
      state:      state,
      committed:  m.committed,
      hasHands:   m.hasHands,
    );
  }

  bool _predictionChanged(Prediction p) =>
      p.label      != _prediction.label      ||
      p.state      != _prediction.state      ||
      p.hasHands   != _prediction.hasHands   ||
      p.committed  != _prediction.committed  ||
      (p.confidence - _prediction.confidence).abs() > 0.02;

  // ── camera control ────────────────────────────────────────────────────────

  Future<void> _startCamera() async {
    final cameras = await availableCameras();
    if (cameras.isEmpty) return;

    final wanted = _useFrontCamera
        ? CameraLensDirection.front
        : CameraLensDirection.back;

    final desc = cameras.firstWhere(
      (c) => c.lensDirection == wanted,
      orElse: () => cameras.first,
    );

    final controller = CameraController(
      desc,
      ResolutionPreset.low, // 320×240 — faster for emulator
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    await controller.initialize();

    final sensorRotation = controller.description.sensorOrientation;

    // Stream frames to the background isolate — no processing on main thread
    await controller.startImageStream((image) {
      _detector.sendFrame(image, sensorRotation);
    });

    if (!mounted) return;
    setState(() {
      _camera      = controller;
      _cameraOn    = true;
      _cameraReady = true;
    });
  }

  void _stopCamera() {
    _camera?.stopImageStream().catchError((_) {});
    _camera?.dispose();
    _camera      = null;
    _cameraOn    = false;
    _cameraReady = false;
    _detector.reset();
    _lastCommitted = null;
  }

  // ── user actions ──────────────────────────────────────────────────────────

  void _onStart() async {
    if (_detectorError != null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Detector error: $_detectorError')),
      );
      return;
    }
    if (!_detectorReady) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Loading model, please wait…')),
      );
      return;
    }
    await _startCamera();
  }

  void _onStop() {
    setState(() {
      _stopCamera();
      _prediction = Prediction.idle;
    });
  }

  Future<void> _onFlipCamera() async {
    final wasOn = _cameraOn;
    _stopCamera();
    setState(() {
      _useFrontCamera = !_useFrontCamera;
      _prediction = Prediction.idle;
    });
    if (wasOn) await _startCamera();
  }

  void _onClear() {
    setState(() {
      _history.clear();
      _lastCommitted = null;
    });
  }

  void _onSpeak() {
    if (_prediction.hasHands && _prediction.urdu != '—') {
      _tts.speak(_prediction.urdu);
    }
  }

  void _onSpeakSentence() {
    if (_history.isEmpty) return;
    _tts.speak(_history.map((h) => h.urdu).join(' '));
  }

  // ── build ─────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      body: SafeArea(
        child: Column(
          children: [
            _buildHeader(),
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.only(bottom: 24),
                child: Column(
                  children: [
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 8),
                      child: Center(
                        child: StatePill(
                          state:    _prediction.state,
                          hasHands: _prediction.hasHands,
                          cameraOn: _cameraOn,
                        ),
                      ),
                    ),
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 18),
                      child: _buildCameraCard(),
                    ),
                    const SizedBox(height: 18),
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 18),
                      child: _buildActionRow(),
                    ),
                    if (_history.isNotEmpty) ...[
                      const SizedBox(height: 18),
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 18),
                        child: _buildSentenceStrip(),
                      ),
                    ],
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
      child: Row(
        children: [
          const Icon(Icons.menu, color: AppColors.textDim, size: 22),
          const SizedBox(width: 12),
          const Text(
            'LISTEN',
            style: TextStyle(
              color: AppColors.accent,
              fontWeight: FontWeight.w900,
              fontSize: 20,
              letterSpacing: 3,
            ),
          ),
          const Spacer(),
          Container(
            width: 32, height: 32,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: AppColors.bgSoft,
              border: Border.all(color: AppColors.border),
            ),
            child: const Icon(Icons.person, color: AppColors.textDim, size: 18),
          ),
        ],
      ),
    );
  }

  Widget _buildCameraCard() {
    return Container(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(24),
        border: Border.all(
          color: _cameraOn ? AppColors.accent : AppColors.border,
          width: 1.5,
        ),
        boxShadow: _cameraOn
            ? [BoxShadow(color: AppColors.accent.withAlpha(80), blurRadius: 22)]
            : null,
      ),
      clipBehavior: Clip.hardEdge,
      child: Column(
        children: [
          AspectRatio(
            aspectRatio: 3 / 4,
            child: Stack(
              fit: StackFit.expand,
              children: [
                _buildCameraPreview(),
                _buildCornerBrackets(),
                if (_cameraOn) _buildLiveBadge(),
                if (_cameraOn) _buildStopButton(),
                _buildFlipButton(),
                if (_cameraOn && !_prediction.hasHands) _buildHandsHint(),
              ],
            ),
          ),
          _buildPredictionPanel(),
        ],
      ),
    );
  }

  Widget _buildCameraPreview() {
    if (!_cameraOn || _camera == null || !_cameraReady) {
      return _buildIdleOverlay();
    }
    return CameraPreview(_camera!);
  }

  Widget _buildIdleOverlay() {
    return Container(
      color: const Color(0xFF05070F),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            width: 84, height: 84,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: AppColors.bgSoft,
              border: Border.all(color: AppColors.accent, width: 2),
            ),
            child: const Icon(Icons.videocam, color: AppColors.accent, size: 36),
          ),
          const SizedBox(height: 18),
          const Text(
            'Tap start to begin live translation',
            style: TextStyle(
              color: AppColors.text,
              fontSize: 16,
              fontWeight: FontWeight.w700,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 8),
          const Padding(
            padding: EdgeInsets.symmetric(horizontal: 32),
            child: Text(
              'Sign one word at a time. Hold each\ngesture for about one second.',
              style: TextStyle(color: AppColors.textDim, fontSize: 13, height: 1.5),
              textAlign: TextAlign.center,
            ),
          ),
          const SizedBox(height: 20),
          GestureDetector(
            onTap: _onStart,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 12),
              decoration: BoxDecoration(
                color: _detectorReady ? AppColors.accent : AppColors.bgSoft,
                borderRadius: BorderRadius.circular(999),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (!_detectorReady && _detectorError == null)
                    const SizedBox(
                      width: 14, height: 14,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: AppColors.textDim,
                      ),
                    )
                  else
                    Icon(
                      _detectorError != null ? Icons.error_outline : Icons.play_arrow,
                      color: _detectorReady ? const Color(0xFF0B1020) : AppColors.textDim,
                      size: 18,
                    ),
                  const SizedBox(width: 6),
                  Text(
                    _detectorError != null
                        ? 'Model Error'
                        : _detectorReady
                            ? 'Start Camera'
                            : 'Loading…',
                    style: TextStyle(
                      color: _detectorReady
                          ? const Color(0xFF0B1020)
                          : AppColors.textDim,
                      fontWeight: FontWeight.w800,
                      letterSpacing: 0.5,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLiveBadge() {
    return Positioned(
      top: 14, left: 14,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: Colors.black54,
          borderRadius: BorderRadius.circular(999),
          border: Border.all(color: AppColors.accent.withAlpha(100)),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 6, height: 6,
              decoration: const BoxDecoration(
                color: AppColors.accent,
                shape: BoxShape.circle,
              ),
            ),
            const SizedBox(width: 5),
            const Text(
              'LIVE',
              style: TextStyle(
                color: AppColors.accent,
                fontWeight: FontWeight.w700,
                fontSize: 11,
                letterSpacing: 1,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStopButton() {
    return Positioned(
      top: 14, right: 14,
      child: GestureDetector(
        onTap: _onStop,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
          decoration: BoxDecoration(
            color: Colors.black54,
            borderRadius: BorderRadius.circular(999),
            border: Border.all(color: AppColors.border),
          ),
          child: const Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.stop, color: AppColors.text, size: 12),
              SizedBox(width: 4),
              Text(
                'STOP',
                style: TextStyle(
                  color: AppColors.text,
                  fontWeight: FontWeight.w700,
                  fontSize: 11,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildFlipButton() {
    return Positioned(
      bottom: 14, right: 14,
      child: GestureDetector(
        onTap: _onFlipCamera,
        child: Container(
          width: 40, height: 40,
          decoration: BoxDecoration(
            color: Colors.black54,
            shape: BoxShape.circle,
            border: Border.all(color: AppColors.border),
          ),
          child: Icon(
            _useFrontCamera
                ? Icons.camera_rear_outlined
                : Icons.camera_front_outlined,
            color: AppColors.text,
            size: 20,
          ),
        ),
      ),
    );
  }

  Widget _buildHandsHint() {
    return Positioned(
      bottom: 18,
      left: 0, right: 0,
      child: Center(
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
          decoration: BoxDecoration(
            color: Colors.black54,
            borderRadius: BorderRadius.circular(999),
            border: Border.all(color: AppColors.border),
          ),
          child: const Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.back_hand_outlined, color: AppColors.warn, size: 14),
              SizedBox(width: 6),
              Text(
                'Show both hands to begin signing',
                style: TextStyle(color: AppColors.text, fontSize: 12),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildCornerBrackets() {
    const size  = 22.0;
    const thick = 2.0;
    const color = AppColors.accent;
    const pad   = 10.0;
    return Stack(
      children: [
        Positioned(top: pad, left: pad,
          child: Container(width: size, height: size,
            decoration: const BoxDecoration(border: Border(top: BorderSide(color: color, width: thick), left: BorderSide(color: color, width: thick))))),
        Positioned(top: pad, right: pad,
          child: Container(width: size, height: size,
            decoration: const BoxDecoration(border: Border(top: BorderSide(color: color, width: thick), right: BorderSide(color: color, width: thick))))),
        Positioned(bottom: pad, left: pad,
          child: Container(width: size, height: size,
            decoration: const BoxDecoration(border: Border(bottom: BorderSide(color: color, width: thick), left: BorderSide(color: color, width: thick))))),
        Positioned(bottom: pad, right: pad,
          child: Container(width: size, height: size,
            decoration: const BoxDecoration(border: Border(bottom: BorderSide(color: color, width: thick), right: BorderSide(color: color, width: thick))))),
      ],
    );
  }

  Widget _buildPredictionPanel() {
    final p = _prediction;
    return Container(
      padding: const EdgeInsets.all(18),
      color: AppColors.bgCard,
      child: Column(
        children: [
          Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _label('ENGLISH'),
                    const SizedBox(height: 4),
                    Text(
                      p.hasHands ? p.english : '—',
                      style: TextStyle(
                        color: p.hasHands ? AppColors.text : AppColors.textDim,
                        fontSize: 22,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ],
                ),
              ),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    _label('URDU'),
                    const SizedBox(height: 4),
                    Text(
                      p.hasHands ? p.urdu : '—',
                      textDirection: TextDirection.rtl,
                      style: TextStyle(
                        color: p.hasHands ? AppColors.text : AppColors.textDim,
                        fontSize: 24,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              _label('CONFIDENCE'),
              Text(
                '${(p.confidence * 100).toStringAsFixed(1)}%',
                style: const TextStyle(
                  color: AppColors.accent,
                  fontWeight: FontWeight.w700,
                  fontSize: 13,
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          ConfidenceBar(value: p.confidence),
        ],
      ),
    );
  }

  Widget _buildActionRow() {
    return Row(
      children: [
        Expanded(
          child: GestureDetector(
            onTap: _onSpeak,
            child: Container(
              height: 52,
              decoration: BoxDecoration(
                color: !_prediction.hasHands
                    ? AppColors.bgSoft
                    : AppColors.accent,
                borderRadius: BorderRadius.circular(16),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.volume_up,
                      color: _prediction.hasHands
                          ? const Color(0xFF0B1020)
                          : AppColors.textDim,
                      size: 18),
                  const SizedBox(width: 6),
                  Text(
                    'Speak',
                    style: TextStyle(
                      color: _prediction.hasHands
                          ? const Color(0xFF0B1020)
                          : AppColors.textDim,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
        const SizedBox(width: 10),
        _iconBtn(Icons.close, _onClear),
        const SizedBox(width: 10),
        _iconBtn(
          _showHistory ? Icons.keyboard_arrow_up : Icons.history,
          () => setState(() => _showHistory = !_showHistory),
        ),
      ],
    );
  }

  Widget _iconBtn(IconData icon, VoidCallback onTap) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 52, height: 52,
        decoration: BoxDecoration(
          color: AppColors.bgSoft,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: AppColors.border),
        ),
        child: Icon(icon, color: AppColors.text, size: 22),
      ),
    );
  }

  Widget _buildSentenceStrip() {
    final sentence = _history.map((h) => h.urdu).join(' ');
    final english  = _history.map((h) => h.english).join(' ');
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.bgCard,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _label('SENTENCE'),
          const SizedBox(height: 8),
          Align(
            alignment: Alignment.centerRight,
            child: Text(
              sentence,
              textDirection: TextDirection.rtl,
              textAlign: TextAlign.right,
              style: const TextStyle(
                color: AppColors.text,
                fontSize: 20,
                height: 1.6,
              ),
            ),
          ),
          const SizedBox(height: 4),
          Text(
            english,
            style: const TextStyle(color: AppColors.textDim, fontSize: 13),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              _textBtn(Icons.undo, 'Undo', () {
                if (_history.isNotEmpty) setState(() => _history.removeLast());
              }),
              const SizedBox(width: 8),
              _textBtn(Icons.volume_up, 'Speak sentence', _onSpeakSentence,
                  accent: true),
            ],
          ),
          if (_showHistory && _history.isNotEmpty) ...[
            const SizedBox(height: 12),
            const Divider(color: AppColors.border),
            const SizedBox(height: 8),
            ..._history.reversed.map((h) => Padding(
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(h.english, style: const TextStyle(color: AppColors.text)),
                      Text(h.urdu,
                          textDirection: TextDirection.rtl,
                          style: const TextStyle(color: AppColors.accent)),
                    ],
                  ),
                )),
          ],
        ],
      ),
    );
  }

  Widget _textBtn(IconData icon, String label, VoidCallback onTap,
      {bool accent = false}) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: accent ? AppColors.accent.withAlpha(30) : AppColors.bgSoft,
          borderRadius: BorderRadius.circular(10),
          border: Border.all(
            color: accent
                ? AppColors.accent.withAlpha(100)
                : AppColors.border,
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon,
                color: accent ? AppColors.accent : AppColors.text, size: 14),
            const SizedBox(width: 4),
            Text(
              label,
              style: TextStyle(
                color: accent ? AppColors.accent : AppColors.text,
                fontSize: 12,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _label(String text) => Text(
        text,
        style: const TextStyle(
          color: AppColors.textDim,
          fontSize: 10,
          fontWeight: FontWeight.w700,
          letterSpacing: 1.6,
        ),
      );
}

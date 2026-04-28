import 'dart:async';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:hand_landmarker/hand_landmarker.dart';

import '../ml/prediction.dart';
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
  bool _cameraOn = false;
  bool _cameraReady = false;
  bool _useFrontCamera = true;

  // ── hand landmarker (MediaPipe) ─────────────────────────────────────────
  HandLandmarkerPlugin? _landmarker;
  bool _landmarkerReady = false;

  // ── pipeline (feature extraction + TFLite + FSM) ────────────────────────
  final _pipeline = SignPipeline();

  // ── throttle: skip frames while previous detection is running ───────────
  bool _detecting = false;

  // ── prediction ──────────────────────────────────────────────────────────
  final _pred = ValueNotifier<Prediction>(Prediction.idle);

  // ── session history ───────────────────────────────────────────────────
  final List<({String english, String urdu})> _history = [];
  String? _lastCommitted;

  // ── TTS ──────────────────────────────────────────────────────────────
  final FlutterTts _tts = FlutterTts();
  bool _ttsReady = false;

  // ── UI ───────────────────────────────────────────────────────────────
  bool _showHistory = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _initAll();
    });
  }

  @override
  void dispose() {
    _stopCamera();
    _landmarker?.dispose();
    _pipeline.dispose();
    _pred.dispose();
    _tts.stop();
    super.dispose();
  }

  Future<void> _initAll() async {
    // Init TTS
    _tts.setLanguage('ur-PK');
    _tts.setSpeechRate(0.5);
    _ttsReady = true;

    // Init hand landmarker (synchronous — JNI call)
    try {
      _landmarker = HandLandmarkerPlugin.create(
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        delegate: HandLandmarkerDelegate.cpu,
      );
      _landmarkerReady = true;
      debugPrint('PSL: HandLandmarker initialized');
    } catch (e) {
      debugPrint('PSL: HandLandmarker init FAILED: $e');
    }

    // Init TFLite model
    try {
      await _pipeline.init();
      debugPrint('PSL: TFLite model loaded');
    } catch (e) {
      debugPrint('PSL: TFLite init FAILED: $e');
    }

    if (mounted) setState(() {});
  }

  // ── camera ────────────────────────────────────────────────────────────

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

    final ctrl = CameraController(
      desc,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );
    await ctrl.initialize();

    // Stream frames — throttled by _detecting flag
    await ctrl.startImageStream(_onCameraFrame);

    if (!mounted) return;
    setState(() {
      _camera = ctrl;
      _cameraOn = true;
      _cameraReady = true;
    });
  }

  void _onCameraFrame(CameraImage image) {
    // Throttle: skip if previous detection still running
    if (_detecting || !_landmarkerReady || !_pipeline.isReady) return;
    _detecting = true;

    try {
      final rotation = _camera?.description.sensorOrientation ?? 0;

      // Call MediaPipe hand landmarker (synchronous JNI — blocks main thread)
      final hands = _landmarker!.detect(image, rotation);

      if (!mounted) { _detecting = false; return; }

      // Run through pipeline (feature extraction + TFLite + FSM)
      final prediction = _pipeline.process(hands);
      _pred.value = prediction;

      // Handle commit
      if (prediction.committed &&
          prediction.label.isNotEmpty &&
          prediction.label != _lastCommitted) {
        _lastCommitted = prediction.label;
        setState(() => _history.add((english: prediction.english, urdu: prediction.urdu)));
        if (_ttsReady) _tts.speak(prediction.urdu);
      }
    } catch (e) {
      debugPrint('PSL: Detection error: $e');
    } finally {
      _detecting = false;
    }
  }

  void _stopCamera() {
    try { _camera?.stopImageStream(); } catch (_) {}
    _camera?.dispose();
    _camera = null;
    _cameraOn = false;
    _cameraReady = false;
    _detecting = false;
    _pipeline.reset();
    _lastCommitted = null;
  }

  // ── actions ───────────────────────────────────────────────────────────

  void _onStart() async {
    if (!_pipeline.isReady || !_landmarkerReady) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Loading models, please wait…')),
      );
      return;
    }
    await _startCamera();
  }

  void _onStop() {
    _stopCamera();
    _pred.value = Prediction.idle;
    setState(() {});
  }

  Future<void> _onFlipCamera() async {
    final wasOn = _cameraOn;
    _stopCamera();
    _pred.value = Prediction.idle;
    setState(() => _useFrontCamera = !_useFrontCamera);
    if (wasOn) await _startCamera();
  }

  void _onClear() => setState(() { _history.clear(); _lastCommitted = null; });

  void _onSpeak() {
    final p = _pred.value;
    if (p.hasHands && p.urdu != '\u2014') _tts.speak(p.urdu);
  }

  void _onSpeakSentence() {
    if (_history.isEmpty) return;
    _tts.speak(_history.map((h) => h.urdu).join(' '));
  }

  // ── build ─────────────────────────────────────────────────────────────

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
                        child: ValueListenableBuilder<Prediction>(
                          valueListenable: _pred,
                          builder: (_, p, __) => StatePill(
                            state: p.state, hasHands: p.hasHands, cameraOn: _cameraOn,
                          ),
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

  Widget _buildHeader() => Container(
    padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
    child: Row(children: [
      const Icon(Icons.menu, color: AppColors.textDim, size: 22),
      const SizedBox(width: 12),
      const Text('LISTEN', style: TextStyle(
        color: AppColors.accent, fontWeight: FontWeight.w900,
        fontSize: 20, letterSpacing: 3)),
      const Spacer(),
      Container(
        width: 32, height: 32,
        decoration: BoxDecoration(
          shape: BoxShape.circle, color: AppColors.bgSoft,
          border: Border.all(color: AppColors.border)),
        child: const Icon(Icons.person, color: AppColors.textDim, size: 18)),
    ]),
  );

  Widget _buildCameraCard() {
    return Container(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(24),
        border: Border.all(
          color: _cameraOn ? AppColors.accent : AppColors.border, width: 1.5),
        boxShadow: _cameraOn
            ? [BoxShadow(color: AppColors.accent.withAlpha(80), blurRadius: 22)]
            : null,
      ),
      clipBehavior: Clip.hardEdge,
      child: Column(children: [
        AspectRatio(
          aspectRatio: 3 / 4,
          child: Stack(fit: StackFit.expand, children: [
            _cameraOn && _camera != null && _cameraReady
                ? CameraPreview(_camera!)
                : _buildIdleOverlay(),
            _buildCornerBrackets(),
            if (_cameraOn) _buildLiveBadge(),
            if (_cameraOn) _buildStopButton(),
            _buildFlipButton(),
            if (_cameraOn)
              ValueListenableBuilder<Prediction>(
                valueListenable: _pred,
                builder: (_, p, __) =>
                    p.hasHands ? const SizedBox.shrink() : _buildHandsHint(),
              ),
          ]),
        ),
        ValueListenableBuilder<Prediction>(
          valueListenable: _pred,
          builder: (_, p, __) => _buildPredictionPanel(p),
        ),
      ]),
    );
  }

  Widget _buildIdleOverlay() => Container(
    color: const Color(0xFF05070F),
    child: Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Container(
          width: 84, height: 84,
          decoration: BoxDecoration(
            shape: BoxShape.circle, color: AppColors.bgSoft,
            border: Border.all(color: AppColors.accent, width: 2)),
          child: const Icon(Icons.videocam, color: AppColors.accent, size: 36)),
        const SizedBox(height: 18),
        const Text('Tap start to begin live translation',
            style: TextStyle(color: AppColors.text, fontSize: 16, fontWeight: FontWeight.w700),
            textAlign: TextAlign.center),
        const SizedBox(height: 8),
        const Padding(
          padding: EdgeInsets.symmetric(horizontal: 32),
          child: Text('Sign one word at a time. The model reads\nthe last ~2 seconds of motion.',
              style: TextStyle(color: AppColors.textDim, fontSize: 13, height: 1.5),
              textAlign: TextAlign.center)),
        const SizedBox(height: 20),
        GestureDetector(
          onTap: _onStart,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 12),
            decoration: BoxDecoration(
              color: AppColors.accent, borderRadius: BorderRadius.circular(999)),
            child: const Row(mainAxisSize: MainAxisSize.min, children: [
              Icon(Icons.play_arrow, color: Color(0xFF0B1020), size: 18),
              SizedBox(width: 6),
              Text('Start Camera', style: TextStyle(
                  color: Color(0xFF0B1020), fontWeight: FontWeight.w800, letterSpacing: 0.5)),
            ]),
          ),
        ),
      ],
    ),
  );

  Widget _buildLiveBadge() => Positioned(
    top: 14, left: 14,
    child: Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: Colors.black54, borderRadius: BorderRadius.circular(999),
        border: Border.all(color: AppColors.accent.withAlpha(100))),
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        Container(width: 6, height: 6,
            decoration: const BoxDecoration(color: AppColors.accent, shape: BoxShape.circle)),
        const SizedBox(width: 5),
        const Text('LIVE', style: TextStyle(
            color: AppColors.accent, fontWeight: FontWeight.w700, fontSize: 11, letterSpacing: 1)),
      ]),
    ),
  );

  Widget _buildStopButton() => Positioned(
    top: 14, right: 14,
    child: GestureDetector(
      onTap: _onStop,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: Colors.black54, borderRadius: BorderRadius.circular(999),
          border: Border.all(color: AppColors.border)),
        child: const Row(mainAxisSize: MainAxisSize.min, children: [
          Icon(Icons.stop, color: AppColors.text, size: 12),
          SizedBox(width: 4),
          Text('STOP', style: TextStyle(color: AppColors.text, fontWeight: FontWeight.w700, fontSize: 11)),
        ]),
      ),
    ),
  );

  Widget _buildFlipButton() => Positioned(
    bottom: 14, right: 14,
    child: GestureDetector(
      onTap: _onFlipCamera,
      child: Container(
        width: 40, height: 40,
        decoration: BoxDecoration(
          color: Colors.black54, shape: BoxShape.circle,
          border: Border.all(color: AppColors.border)),
        child: Icon(
          _useFrontCamera ? Icons.camera_rear_outlined : Icons.camera_front_outlined,
          color: AppColors.text, size: 20),
      ),
    ),
  );

  Widget _buildHandsHint() => Positioned(
    bottom: 18, left: 0, right: 0,
    child: Center(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
        decoration: BoxDecoration(
          color: Colors.black54, borderRadius: BorderRadius.circular(999),
          border: Border.all(color: AppColors.border)),
        child: const Row(mainAxisSize: MainAxisSize.min, children: [
          Icon(Icons.back_hand_outlined, color: AppColors.warn, size: 14),
          SizedBox(width: 6),
          Text('Show your hands to begin signing',
              style: TextStyle(color: AppColors.text, fontSize: 12)),
        ]),
      ),
    ),
  );

  Widget _buildCornerBrackets() {
    const s = 22.0, t = 2.0, c = AppColors.accent, p = 10.0;
    return Stack(children: [
      Positioned(top: p, left: p, child: Container(width: s, height: s,
          decoration: const BoxDecoration(border: Border(top: BorderSide(color: c, width: t), left: BorderSide(color: c, width: t))))),
      Positioned(top: p, right: p, child: Container(width: s, height: s,
          decoration: const BoxDecoration(border: Border(top: BorderSide(color: c, width: t), right: BorderSide(color: c, width: t))))),
      Positioned(bottom: p, left: p, child: Container(width: s, height: s,
          decoration: const BoxDecoration(border: Border(bottom: BorderSide(color: c, width: t), left: BorderSide(color: c, width: t))))),
      Positioned(bottom: p, right: p, child: Container(width: s, height: s,
          decoration: const BoxDecoration(border: Border(bottom: BorderSide(color: c, width: t), right: BorderSide(color: c, width: t))))),
    ]);
  }

  Widget _buildPredictionPanel(Prediction p) {
    final bufPct = _pipeline.bufferCapacity > 0
        ? _pipeline.bufferFill / _pipeline.bufferCapacity
        : 0.0;
    final bufReady = _pipeline.bufferFill >= _pipeline.bufferCapacity;

    return Container(
      padding: const EdgeInsets.all(18),
      color: AppColors.bgCard,
      child: Column(children: [
        Row(children: [
          Expanded(child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _label('ENGLISH'), const SizedBox(height: 4),
              Text(p.hasHands ? p.english : '\u2014', style: TextStyle(
                  color: p.hasHands ? AppColors.text : AppColors.textDim,
                  fontSize: 22, fontWeight: FontWeight.w700)),
            ],
          )),
          Expanded(child: Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              _label('URDU'), const SizedBox(height: 4),
              Text(p.hasHands ? p.urdu : '\u2014', textDirection: TextDirection.rtl,
                  style: TextStyle(
                      color: p.hasHands ? AppColors.text : AppColors.textDim,
                      fontSize: 24, fontWeight: FontWeight.w700)),
            ],
          )),
        ]),
        if (p.hasHands && !bufReady) ...[
          const SizedBox(height: 10),
          Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
            _label('CAPTURING MOTION'),
            Text('${_pipeline.bufferFill}/${_pipeline.bufferCapacity}',
                style: const TextStyle(color: AppColors.warn, fontWeight: FontWeight.w700, fontSize: 12)),
          ]),
          const SizedBox(height: 4),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: bufPct,
              backgroundColor: AppColors.bgSoft,
              color: AppColors.warn,
              minHeight: 4,
            ),
          ),
          const SizedBox(height: 4),
          Text('Sign naturally — the model watches the last 2 seconds.',
              style: TextStyle(color: AppColors.warn.withAlpha(180), fontSize: 11)),
        ],
        const SizedBox(height: 14),
        Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
          _label('CONFIDENCE'),
          Text('${(p.confidence * 100).toStringAsFixed(1)}%',
              style: const TextStyle(color: AppColors.accent, fontWeight: FontWeight.w700, fontSize: 13)),
        ]),
        const SizedBox(height: 6),
        ConfidenceBar(value: p.confidence),
      ]),
    );
  }

  Widget _buildActionRow() => Row(children: [
    Expanded(
      child: ValueListenableBuilder<Prediction>(
        valueListenable: _pred,
        builder: (_, p, __) => GestureDetector(
          onTap: _onSpeak,
          child: Container(
            height: 52,
            decoration: BoxDecoration(
              color: p.hasHands ? AppColors.accent : AppColors.bgSoft,
              borderRadius: BorderRadius.circular(16)),
            child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
              Icon(Icons.volume_up,
                  color: p.hasHands ? const Color(0xFF0B1020) : AppColors.textDim, size: 18),
              const SizedBox(width: 6),
              Text('Speak', style: TextStyle(
                  color: p.hasHands ? const Color(0xFF0B1020) : AppColors.textDim,
                  fontWeight: FontWeight.w700)),
            ]),
          ),
        ),
      ),
    ),
    const SizedBox(width: 10),
    _iconBtn(Icons.close, _onClear),
    const SizedBox(width: 10),
    _iconBtn(
      _showHistory ? Icons.keyboard_arrow_up : Icons.history,
      () => setState(() => _showHistory = !_showHistory)),
  ]);

  Widget _iconBtn(IconData icon, VoidCallback onTap) => GestureDetector(
    onTap: onTap,
    child: Container(
      width: 52, height: 52,
      decoration: BoxDecoration(
        color: AppColors.bgSoft, borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border)),
      child: Icon(icon, color: AppColors.text, size: 22)),
  );

  Widget _buildSentenceStrip() {
    final sentence = _history.map((h) => h.urdu).join(' ');
    final english = _history.map((h) => h.english).join(' ');
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.bgCard, borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border)),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        _label('SENTENCE'),
        const SizedBox(height: 8),
        Align(
          alignment: Alignment.centerRight,
          child: Text(sentence, textDirection: TextDirection.rtl,
              textAlign: TextAlign.right,
              style: const TextStyle(color: AppColors.text, fontSize: 20, height: 1.6))),
        const SizedBox(height: 4),
        Text(english, style: const TextStyle(color: AppColors.textDim, fontSize: 13)),
        const SizedBox(height: 12),
        Row(children: [
          _textBtn(Icons.undo, 'Undo', () {
            if (_history.isNotEmpty) setState(() => _history.removeLast());
          }),
          const SizedBox(width: 8),
          _textBtn(Icons.volume_up, 'Speak sentence', _onSpeakSentence, accent: true),
        ]),
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
                Text(h.urdu, textDirection: TextDirection.rtl,
                    style: const TextStyle(color: AppColors.accent)),
              ]),
          )),
        ],
      ]),
    );
  }

  Widget _textBtn(IconData icon, String label, VoidCallback onTap, {bool accent = false}) =>
    GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: accent ? AppColors.accent.withAlpha(30) : AppColors.bgSoft,
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: accent ? AppColors.accent.withAlpha(100) : AppColors.border)),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          Icon(icon, color: accent ? AppColors.accent : AppColors.text, size: 14),
          const SizedBox(width: 4),
          Text(label, style: TextStyle(
              color: accent ? AppColors.accent : AppColors.text,
              fontSize: 12, fontWeight: FontWeight.w600)),
        ]),
      ),
    );

  Widget _label(String text) => Text(text, style: const TextStyle(
      color: AppColors.textDim, fontSize: 10, fontWeight: FontWeight.w700, letterSpacing: 1.6));
}

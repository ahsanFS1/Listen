import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

import '../data/signs.dart';
import 'prediction.dart';

enum SignMode { words, alphabets }

/// Streams JPEG frames to the inference server and surfaces the latest
/// prediction. API mirrors the old on-device SignPipeline so the UI
/// layer doesn't need to change.
class SignClient {
  SignClient({required this.url, this.mode = SignMode.words});

  /// Base URL e.g. `ws://192.168.1.5:8000/ws/translate`. The `?mode=`
  /// query is appended automatically.
  final String url;
  final SignMode mode;

  String get _connectUrl {
    final m = mode == SignMode.alphabets ? 'alphabets' : 'words';
    final sep = url.contains('?') ? '&' : '?';
    return '$url${sep}mode=$m';
  }

  WebSocketChannel? _channel;
  StreamSubscription? _sub;

  bool _ready = false;
  bool get isReady => _ready;

  // Drop frames if the server is busy — keeping a backlog only adds
  // latency without improving accuracy. _inFlight is incremented when we
  // send and decremented when we get a reply (or after a timeout).
  int _inFlight = 0;
  static const int _maxInFlight = 2;

  int _bufferFill = 0;
  int get bufferFill => _bufferFill;

  int _bufferCapacity = 60;
  int get bufferCapacity => _bufferCapacity;

  Prediction _last = Prediction.idle;
  Prediction get last => _last;

  // Notify listeners (UI) when a new snapshot arrives.
  final _controller = StreamController<Prediction>.broadcast();
  Stream<Prediction> get predictions => _controller.stream;

  // Fired exactly once per recognized word.
  final _commits = StreamController<Prediction>.broadcast();
  Stream<Prediction> get commits => _commits.stream;

  // Surfaces "the WS just dropped / errored" so the UI can show it.
  // Strings are user-readable error messages; null means a clean close.
  final _errors = StreamController<String?>.broadcast();
  Stream<String?> get errors => _errors.stream;
  String? _lastError;
  String? get lastError => _lastError;

  Future<void> connect() async {
    debugPrint('PSL: connecting → $_connectUrl');
    try {
      final ch = WebSocketChannel.connect(Uri.parse(_connectUrl));
      _channel = ch;
      // ready() throws synchronously inside the listener if the handshake
      // fails — listen routes both the failure and any later disconnects
      // into _errors so the UI can react.
      _sub = ch.stream.listen(
        _onMessage,
        onError: (e, st) {
          final msg = 'WS error: $e';
          debugPrint('PSL: $msg');
          _lastError = msg;
          _ready = false;
          if (!_errors.isClosed) _errors.add(msg);
        },
        onDone: () {
          debugPrint('PSL: ws closed');
          _ready = false;
          if (!_errors.isClosed) _errors.add(null);
        },
        cancelOnError: true,
      );
      _ready = true;
      _lastError = null;
    } catch (e) {
      _ready = false;
      _lastError = 'WS connect failed: $e';
      debugPrint('PSL: ${_lastError!}');
      if (!_errors.isClosed) _errors.add(_lastError);
      rethrow;
    }
  }

  Future<void> dispose() async {
    _ready = false;
    await _sub?.cancel();
    await _channel?.sink.close();
    _sub = null;
    _channel = null;
    await _controller.close();
    await _commits.close();
    await _errors.close();
  }

  /// Send a JPEG frame to the server. Drops the frame if the server
  /// hasn't replied to recent ones — backpressure to avoid latency.
  void sendFrame(Uint8List jpegBytes) {
    final ch = _channel;
    if (ch == null || !_ready) return;
    if (_inFlight >= _maxInFlight) return;
    _inFlight++;
    ch.sink.add(jpegBytes);
  }

  void _onMessage(dynamic msg) {
    if (_inFlight > 0) _inFlight--;
    if (msg is! String) return;
    final Map<String, dynamic> j;
    try {
      j = jsonDecode(msg) as Map<String, dynamic>;
    } catch (e) {
      debugPrint('PSL: bad ws json: $e');
      return;
    }

    if (j.containsKey('pong')) return;

    final state = _parseState(j['state'] as String? ?? 'IDLE');
    final label = (j['label'] as String?) ?? '';
    final english = (j['english'] as String?) ?? label;
    final urduFromServer = (j['urdu'] as String?) ?? '';
    final sign = findSign(label);
    final urdu = urduFromServer.isNotEmpty
        ? urduFromServer
        : (sign?.urdu ?? label);
    final conf = (j['confidence'] as num?)?.toDouble() ?? 0.0;
    final committed = (j['committed'] as bool?) ?? false;
    final hasHands = (j['hasHands'] as bool?) ?? false;
    _bufferFill = (j['bufferFill'] as num?)?.toInt() ?? _bufferFill;
    _bufferCapacity = (j['bufferCapacity'] as num?)?.toInt() ?? _bufferCapacity;

    final p = Prediction(
      label: label,
      english: english.isNotEmpty ? english : (sign?.english ?? label),
      urdu: urdu,
      confidence: conf,
      state: state,
      committed: committed,
      hasHands: hasHands,
    );
    _last = p;
    _controller.add(p);
    if (committed && !kIdleClasses.contains(label)) {
      _commits.add(p);
    }
  }

  static SignState _parseState(String s) {
    switch (s) {
      case 'SIGNING':
        return SignState.signing;
      case 'PREDICTING':
        return SignState.predicting;
      case 'COMMITTED':
        return SignState.committed;
      case 'COOLDOWN':
        return SignState.cooldown;
      case 'IDLE':
      default:
        return SignState.idle;
    }
  }
}

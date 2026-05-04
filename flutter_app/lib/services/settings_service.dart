import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// User-tunable runtime settings: TTS speed, prediction confidence
/// threshold, and preferred Urdu/English language for spoken output.
/// Persisted via SharedPreferences and exposed as a singleton
/// [ChangeNotifier] so screens can `AnimatedBuilder` against it.
class SettingsService extends ChangeNotifier {
  SettingsService._();
  static final SettingsService instance = SettingsService._();

  static const _kTtsSpeed = 'settings.tts_speed_v1';
  static const _kConfidence = 'settings.confidence_v1';
  static const _kLang = 'settings.lang_v1'; // 'urdu' | 'english'
  static const _kServerUrl = 'settings.server_url_v1';

  // Build-time default. Overridden at runtime by user-entered URL in profile.
  static const String defaultServerUrl = String.fromEnvironment(
    'PSL_WS_URL',
    defaultValue: 'ws://192.168.1.16:8000/ws/translate',
  );

  // Sensible defaults — match the previously hardcoded values in
  // translate_screen.dart (0.45 rate, Urdu).
  double _ttsSpeed = 0.45;
  double _confidence = 0.70;
  String _lang = 'urdu';
  String _serverUrl = defaultServerUrl;
  bool _loaded = false;

  double get ttsSpeed => _ttsSpeed;
  double get confidence => _confidence;
  String get lang => _lang;
  String get serverUrl => _serverUrl;
  bool get isLoaded => _loaded;
  bool get prefersUrdu => _lang == 'urdu';

  String get ttsSpeedLabel {
    if (_ttsSpeed < 0.35) return 'Slow';
    if (_ttsSpeed < 0.55) return 'Normal';
    if (_ttsSpeed < 0.75) return 'Fast';
    return 'Very Fast';
  }

  String get langLabel => _lang == 'urdu' ? 'اردو (Urdu)' : 'English';

  Future<void> load() async {
    if (_loaded) return;
    final sp = await SharedPreferences.getInstance();
    _ttsSpeed = sp.getDouble(_kTtsSpeed) ?? 0.45;
    _confidence = sp.getDouble(_kConfidence) ?? 0.70;
    _lang = sp.getString(_kLang) ?? 'urdu';
    _serverUrl = sp.getString(_kServerUrl) ?? defaultServerUrl;
    _loaded = true;
    notifyListeners();
  }

  Future<void> setServerUrl(String v) async {
    final t = v.trim();
    if (!t.startsWith('ws://') && !t.startsWith('wss://')) return;
    _serverUrl = t;
    final sp = await SharedPreferences.getInstance();
    await sp.setString(_kServerUrl, _serverUrl);
    notifyListeners();
  }

  Future<void> resetServerUrl() async {
    _serverUrl = defaultServerUrl;
    final sp = await SharedPreferences.getInstance();
    await sp.remove(_kServerUrl);
    notifyListeners();
  }

  Future<void> setTtsSpeed(double v) async {
    _ttsSpeed = v.clamp(0.20, 0.90);
    final sp = await SharedPreferences.getInstance();
    await sp.setDouble(_kTtsSpeed, _ttsSpeed);
    notifyListeners();
  }

  Future<void> setConfidence(double v) async {
    _confidence = v.clamp(0.30, 0.95);
    final sp = await SharedPreferences.getInstance();
    await sp.setDouble(_kConfidence, _confidence);
    notifyListeners();
  }

  Future<void> setLang(String l) async {
    if (l != 'urdu' && l != 'english') return;
    _lang = l;
    final sp = await SharedPreferences.getInstance();
    await sp.setString(_kLang, _lang);
    notifyListeners();
  }
}

import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

import 'settings_service.dart';

/// Urdu word + sentence completion via the inference server's REST
/// endpoints. Mirrors the desktop psl-v1.py predict_words /
/// suggest_phrases queries — the actual SQL lives in server/suggestions.py.
///
/// All failures are swallowed and surfaced as an empty list — suggestions
/// are a UX nicety, not a hard dependency, so a DB outage must never break
/// the live translation flow.
class SuggestionService {
  SuggestionService._();
  static final SuggestionService instance = SuggestionService._();

  final _client = http.Client();

  Uri? _endpoint(String path) {
    final ws = SettingsService.instance.serverUrl;
    if (ws.isEmpty) return null;
    final u = Uri.tryParse(ws);
    if (u == null) return null;
    final scheme = (u.scheme == 'wss') ? 'https' : 'http';
    return Uri(
      scheme: scheme,
      host: u.host,
      port: u.hasPort ? u.port : 8000,
      path: path,
    );
  }

  Future<List<String>> _fetch(String path, String prefix) async {
    final base = _endpoint(path);
    if (base == null) return const [];
    final url = base.replace(queryParameters: {'prefix': prefix});
    try {
      final r = await _client
          .get(url)
          .timeout(const Duration(milliseconds: 1500));
      if (r.statusCode != 200) return const [];
      final j = jsonDecode(r.body) as Map<String, dynamic>;
      final list = (j['suggestions'] as List?) ?? const [];
      return list.map((e) => e.toString()).toList(growable: false);
    } catch (_) {
      return const [];
    }
  }

  Future<List<String>> words(String prefix) =>
      _fetch('/suggest/words', prefix);

  Future<List<String>> sentences(String prefix) =>
      _fetch('/suggest/sentences', prefix);
}

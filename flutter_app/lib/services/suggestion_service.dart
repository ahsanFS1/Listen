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
    // Only LAN dev URLs carry an explicit port (e.g. :8000); hosted
    // deployments (Railway et al.) serve on the scheme's standard port with
    // no port in the URL, so falling back to 8000 there would silently break
    // suggestions against any host that isn't a bare LAN IP.
    final port = u.hasPort ? u.port : (scheme == 'https' ? 443 : 80);
    return Uri(
      scheme: scheme,
      host: u.host,
      port: port,
      path: path,
    );
  }

  Future<({List<String> items, String? error})> _fetch(
      String path, String prefix) async {
    final base = _endpoint(path);
    if (base == null) {
      return (items: const <String>[], error: 'no server URL configured');
    }
    final url = base.replace(queryParameters: {'prefix': prefix});
    try {
      final r = await _client
          .get(url)
          .timeout(const Duration(milliseconds: 2500));
      if (r.statusCode != 200) {
        return (items: const <String>[], error: 'HTTP ${r.statusCode}');
      }
      final j = jsonDecode(r.body) as Map<String, dynamic>;
      final raw = (j['suggestions'] as List?) ?? const [];
      final list = raw.map((e) => e.toString()).toList(growable: false);
      return (items: list, error: null);
    } catch (e) {
      return (items: const <String>[], error: e.toString());
    }
  }

  Future<({List<String> items, String? error})> words(String prefix) =>
      _fetch('/suggest/words', prefix);

  Future<({List<String> items, String? error})> sentences(String prefix) =>
      _fetch('/suggest/sentences', prefix);
}

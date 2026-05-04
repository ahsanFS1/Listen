import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../data/signs.dart';

/// Tracks which of the PSL words the user has marked as learned.
/// Persisted via SharedPreferences; surfaced as a [ChangeNotifier] +
/// process-wide singleton so the Learn screen's progress card and any
/// other widget that listens stays in sync without prop-drilling.
class ProgressService extends ChangeNotifier {
  ProgressService._();
  static final ProgressService instance = ProgressService._();

  static const _kKey = 'learned_signs_v1';

  Set<String> _learned = <String>{};
  bool _loaded = false;

  Set<String> get learned => Set.unmodifiable(_learned);
  int get learnedCount => _learned.length;
  int get total => kSigns.length;
  double get fraction => total == 0 ? 0 : learnedCount / total;
  bool get isLoaded => _loaded;

  bool isLearned(String id) => _learned.contains(id);

  Future<void> load() async {
    if (_loaded) return;
    final sp = await SharedPreferences.getInstance();
    _learned = sp.getStringList(_kKey)?.toSet() ?? <String>{};
    _loaded = true;
    notifyListeners();
  }

  Future<void> markLearned(String id) async {
    if (_learned.add(id)) {
      await _persist();
      notifyListeners();
    }
  }

  Future<void> unmark(String id) async {
    if (_learned.remove(id)) {
      await _persist();
      notifyListeners();
    }
  }

  Future<void> resetAll() async {
    _learned.clear();
    await _persist();
    notifyListeners();
  }

  Future<void> _persist() async {
    final sp = await SharedPreferences.getInstance();
    await sp.setStringList(_kKey, _learned.toList());
  }
}

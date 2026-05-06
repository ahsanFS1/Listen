import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Tracks the user's consecutive-days-of-use streak. Call [touchToday]
/// once per app launch — if the last touch was yesterday the streak
/// extends, if it was earlier the streak resets to 1.
class StreakService extends ChangeNotifier {
  StreakService._();
  static final StreakService instance = StreakService._();

  static const _kStreak = 'streak.count_v1';
  static const _kLastDay = 'streak.last_day_v1'; // yyyy-mm-dd

  int _streak = 0;
  String? _lastDay;
  bool _loaded = false;

  int get streak => _streak;
  bool get isLoaded => _loaded;

  Future<void> load() async {
    if (_loaded) return;
    final sp = await SharedPreferences.getInstance();
    _streak = sp.getInt(_kStreak) ?? 0;
    _lastDay = sp.getString(_kLastDay);
    _loaded = true;
    notifyListeners();
  }

  Future<void> touchToday() async {
    await load();
    final now = DateTime.now();
    final today = _fmt(now);
    if (_lastDay == today) return;

    final yesterday = _fmt(now.subtract(const Duration(days: 1)));
    if (_lastDay == yesterday) {
      _streak += 1;
    } else {
      _streak = 1;
    }
    _lastDay = today;
    final sp = await SharedPreferences.getInstance();
    await sp.setInt(_kStreak, _streak);
    await sp.setString(_kLastDay, _lastDay!);
    notifyListeners();
  }

  Future<void> reset() async {
    _streak = 0;
    _lastDay = null;
    final sp = await SharedPreferences.getInstance();
    await sp.remove(_kStreak);
    await sp.remove(_kLastDay);
    notifyListeners();
  }

  String _fmt(DateTime d) =>
      '${d.year.toString().padLeft(4, '0')}-${d.month.toString().padLeft(2, '0')}-${d.day.toString().padLeft(2, '0')}';
}

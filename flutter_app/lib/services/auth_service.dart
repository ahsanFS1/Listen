import 'package:flutter/foundation.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import '../config/supabase_config.dart';

/// Thin wrapper over supabase_flutter's auth API so the rest of the app
/// doesn't import supabase_flutter directly. Surfaces the current user
/// as a [ChangeNotifier] for trivial widget rebuilds.
class AuthService extends ChangeNotifier {
  AuthService._();
  static final AuthService instance = AuthService._();

  SupabaseClient get _sb => Supabase.instance.client;
  User? _user;
  bool _guest = false;
  bool _initialized = false;

  User? get currentUser => _user;
  bool get isSignedIn => _user != null;
  bool get isGuest => _guest;
  bool get isInitialized => _initialized;

  /// Enter the app without an account. Used because the demo backend has no
  /// live auth server — lets anyone who installs the app reach the main UI
  /// with one tap. Real sign-in/up still works if a Supabase project is set.
  void continueAsGuest() {
    _guest = true;
    notifyListeners();
  }

  String get displayName {
    final u = _user;
    if (u == null) return 'Guest';
    final meta = u.userMetadata;
    final name = meta?['name'] as String?;
    if (name != null && name.trim().isNotEmpty) return name;
    final email = u.email ?? '';
    if (email.isEmpty) return 'PSL Learner';
    final at = email.indexOf('@');
    return at > 0 ? email.substring(0, at) : email;
  }

  String get email => _user?.email ?? '';

  /// Call after Supabase.initialize(). Subscribes to auth state changes
  /// so the rest of the UI rebuilds when the user signs in / out.
  void start() {
    if (_initialized) return;
    _user = _sb.auth.currentUser;
    _sb.auth.onAuthStateChange.listen((event) {
      _user = event.session?.user;
      notifyListeners();
    });
    _initialized = true;
    notifyListeners();
  }

  Future<void> signIn(String email, String password) async {
    final res = await _sb.auth.signInWithPassword(
      email: email.trim(),
      password: password,
    );
    _user = res.user;
    notifyListeners();
  }

  Future<void> signUp(String email, String password, {String? name}) async {
    final res = await _sb.auth.signUp(
      email: email.trim(),
      password: password,
      data: name == null || name.trim().isEmpty ? null : {'name': name.trim()},
    );
    _user = res.user;
    notifyListeners();
  }

  Future<void> signOut() async {
    // Swallow errors — with no live auth backend the network call throws,
    // but the user still expects to be returned to the auth screen.
    try {
      await _sb.auth.signOut();
    } catch (_) {}
    _user = null;
    _guest = false;
    notifyListeners();
  }

  Future<void> updateName(String name) async {
    final res = await _sb.auth.updateUser(
      UserAttributes(data: {'name': name.trim()}),
    );
    _user = res.user;
    notifyListeners();
  }

  Future<void> resetPassword(String email) async {
    await _sb.auth.resetPasswordForEmail(email.trim());
  }

  // Re-export for main.dart
  static Future<void> initSupabase() async {
    await Supabase.initialize(
      url: SupabaseConfig.url,
      anonKey: SupabaseConfig.anonKey,
      debug: kDebugMode,
    );
  }
}

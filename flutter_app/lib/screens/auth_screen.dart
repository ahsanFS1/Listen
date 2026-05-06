import 'package:flutter/material.dart';

import '../services/auth_service.dart';
import '../theme/app_colors.dart';

/// Combined sign-in / sign-up screen. Toggles via tab at top.
/// Routes to the main shell automatically once Supabase auth state
/// changes (handled by [AuthGate] in app.dart).
class AuthScreen extends StatefulWidget {
  const AuthScreen({super.key});

  @override
  State<AuthScreen> createState() => _AuthScreenState();
}

class _AuthScreenState extends State<AuthScreen> {
  bool _isSignUp = false;
  bool _busy = false;
  bool _obscure = true;
  String? _error;
  String? _info;

  final _emailCtrl = TextEditingController();
  final _passCtrl = TextEditingController();
  final _nameCtrl = TextEditingController();
  final _formKey = GlobalKey<FormState>();

  @override
  void dispose() {
    _emailCtrl.dispose();
    _passCtrl.dispose();
    _nameCtrl.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() {
      _busy = true;
      _error = null;
      _info = null;
    });
    try {
      if (_isSignUp) {
        await AuthService.instance.signUp(
          _emailCtrl.text,
          _passCtrl.text,
          name: _nameCtrl.text,
        );
        if (!mounted) return;
        if (AuthService.instance.isSignedIn) {
          // session is live — AuthGate will swap us out automatically
        } else {
          setState(() => _info =
              'Account created. Check your inbox to confirm, then sign in.');
          setState(() => _isSignUp = false);
        }
      } else {
        await AuthService.instance.signIn(_emailCtrl.text, _passCtrl.text);
      }
    } catch (e) {
      if (mounted) setState(() => _error = _prettyError(e));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _forgotPassword() async {
    final email = _emailCtrl.text.trim();
    if (email.isEmpty) {
      setState(() => _error = 'Enter your email above first');
      return;
    }
    setState(() {
      _busy = true;
      _error = null;
      _info = null;
    });
    try {
      await AuthService.instance.resetPassword(email);
      if (mounted) setState(() => _info = 'Password reset email sent.');
    } catch (e) {
      if (mounted) setState(() => _error = _prettyError(e));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  String _prettyError(Object e) {
    final s = e.toString();
    if (s.contains('Invalid login')) return 'Wrong email or password.';
    if (s.contains('already registered')) {
      return 'That email is already in use.';
    }
    if (s.contains('Email not confirmed')) {
      return 'Please confirm your email first.';
    }
    return s.replaceFirst('AuthException(', '').replaceFirst(')', '');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 32),
            child: Form(
              key: _formKey,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  _buildLogo(),
                  const SizedBox(height: 28),
                  _buildToggle(),
                  const SizedBox(height: 22),
                  if (_isSignUp) ...[
                    _buildField(
                      controller: _nameCtrl,
                      label: 'Name',
                      icon: Icons.person_outline,
                      validator: (v) =>
                          (v == null || v.trim().isEmpty) ? 'Required' : null,
                    ),
                    const SizedBox(height: 12),
                  ],
                  _buildField(
                    controller: _emailCtrl,
                    label: 'Email',
                    icon: Icons.mail_outline,
                    keyboard: TextInputType.emailAddress,
                    validator: (v) {
                      if (v == null || v.trim().isEmpty) return 'Required';
                      if (!v.contains('@')) return 'Enter a valid email';
                      return null;
                    },
                  ),
                  const SizedBox(height: 12),
                  _buildField(
                    controller: _passCtrl,
                    label: 'Password',
                    icon: Icons.lock_outline,
                    obscure: _obscure,
                    suffix: IconButton(
                      icon: Icon(
                        _obscure ? Icons.visibility : Icons.visibility_off,
                        color: AppColors.textDim,
                        size: 18,
                      ),
                      onPressed: () => setState(() => _obscure = !_obscure),
                    ),
                    validator: (v) {
                      if (v == null || v.isEmpty) return 'Required';
                      if (_isSignUp && v.length < 6) {
                        return 'At least 6 characters';
                      }
                      return null;
                    },
                  ),
                  if (!_isSignUp) ...[
                    const SizedBox(height: 4),
                    Align(
                      alignment: Alignment.centerRight,
                      child: TextButton(
                        onPressed: _busy ? null : _forgotPassword,
                        child: const Text(
                          'Forgot password?',
                          style: TextStyle(
                              color: AppColors.accent, fontSize: 12),
                        ),
                      ),
                    ),
                  ],
                  const SizedBox(height: 12),
                  if (_error != null) _buildBanner(_error!, AppColors.err),
                  if (_info != null) _buildBanner(_info!, AppColors.ok),
                  const SizedBox(height: 4),
                  _buildSubmit(),
                  const SizedBox(height: 18),
                  Center(
                    child: GestureDetector(
                      onTap: _busy
                          ? null
                          : () => setState(() {
                                _isSignUp = !_isSignUp;
                                _error = null;
                                _info = null;
                              }),
                      child: RichText(
                        text: TextSpan(
                          style: const TextStyle(
                              color: AppColors.textDim, fontSize: 13),
                          children: [
                            TextSpan(
                              text: _isSignUp
                                  ? 'Already have an account? '
                                  : "Don't have an account? ",
                            ),
                            TextSpan(
                              text: _isSignUp ? 'Sign in' : 'Sign up',
                              style: const TextStyle(
                                  color: AppColors.accent,
                                  fontWeight: FontWeight.w700),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildLogo() {
    return Column(
      children: [
        Container(
          width: 78,
          height: 78,
          decoration: BoxDecoration(
            color: AppColors.bgCard,
            shape: BoxShape.circle,
            border: Border.all(color: AppColors.accent, width: 2),
          ),
          child: const Icon(Icons.sign_language,
              color: AppColors.accent, size: 38),
        ),
        const SizedBox(height: 14),
        const Text('LISTEN',
            style: TextStyle(
                color: AppColors.accent,
                fontWeight: FontWeight.w900,
                fontSize: 26,
                letterSpacing: 6)),
        const SizedBox(height: 4),
        const Text('Pakistani Sign Language',
            style: TextStyle(color: AppColors.textDim, fontSize: 13)),
      ],
    );
  }

  Widget _buildToggle() {
    Widget seg(String label, bool active, VoidCallback onTap) {
      return Expanded(
        child: GestureDetector(
          onTap: onTap,
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 180),
            height: 38,
            alignment: Alignment.center,
            decoration: BoxDecoration(
              color: active ? AppColors.accent : Colors.transparent,
              borderRadius: BorderRadius.circular(999),
            ),
            child: Text(label,
                style: TextStyle(
                    color: active
                        ? const Color(0xFF0B1020)
                        : AppColors.textDim,
                    fontWeight: FontWeight.w700,
                    fontSize: 13,
                    letterSpacing: 0.5)),
          ),
        ),
      );
    }

    return Container(
      padding: const EdgeInsets.all(4),
      decoration: BoxDecoration(
        color: AppColors.bgSoft,
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(children: [
        seg('Sign In', !_isSignUp, () => setState(() => _isSignUp = false)),
        seg('Sign Up', _isSignUp, () => setState(() => _isSignUp = true)),
      ]),
    );
  }

  Widget _buildField({
    required TextEditingController controller,
    required String label,
    required IconData icon,
    bool obscure = false,
    TextInputType? keyboard,
    String? Function(String?)? validator,
    Widget? suffix,
  }) {
    return TextFormField(
      controller: controller,
      obscureText: obscure,
      keyboardType: keyboard,
      validator: validator,
      autocorrect: false,
      enableSuggestions: false,
      style: const TextStyle(color: AppColors.text),
      decoration: InputDecoration(
        labelText: label,
        labelStyle: const TextStyle(color: AppColors.textDim),
        prefixIcon: Icon(icon, color: AppColors.textDim, size: 18),
        suffixIcon: suffix,
        filled: true,
        fillColor: AppColors.bgCard,
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: AppColors.border),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: AppColors.accent),
        ),
        errorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: AppColors.err),
        ),
        focusedErrorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: AppColors.err),
        ),
      ),
    );
  }

  Widget _buildBanner(String text, Color color) {
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: color.withAlpha(30),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withAlpha(120)),
      ),
      child: Text(text, style: TextStyle(color: color, fontSize: 12)),
    );
  }

  Widget _buildSubmit() {
    return ElevatedButton(
      onPressed: _busy ? null : _submit,
      style: ElevatedButton.styleFrom(
        backgroundColor: AppColors.accent,
        foregroundColor: const Color(0xFF0B1020),
        padding: const EdgeInsets.symmetric(vertical: 14),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
        ),
      ),
      child: _busy
          ? const SizedBox(
              height: 20,
              width: 20,
              child: CircularProgressIndicator(
                  strokeWidth: 2, color: Color(0xFF0B1020)),
            )
          : Text(_isSignUp ? 'Create Account' : 'Sign In',
              style: const TextStyle(
                  fontWeight: FontWeight.w800, fontSize: 15)),
    );
  }
}

import 'package:flutter/material.dart';

import '../services/auth_service.dart';
import '../services/progress_service.dart';
import '../services/settings_service.dart';
import '../services/streak_service.dart';
import '../theme/app_colors.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  @override
  void initState() {
    super.initState();
    SettingsService.instance.load();
    ProgressService.instance.load();
    StreakService.instance.load();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      body: SafeArea(
        child: AnimatedBuilder(
          animation: Listenable.merge([
            AuthService.instance,
            SettingsService.instance,
            ProgressService.instance,
            StreakService.instance,
          ]),
          builder: (_, __) {
            final auth = AuthService.instance;
            final settings = SettingsService.instance;
            final prog = ProgressService.instance;
            final streak = StreakService.instance;
            return ListView(
              padding: const EdgeInsets.all(18),
              children: [
                const SizedBox(height: 8),
                _buildAvatar(auth.displayName),
                const SizedBox(height: 12),
                Center(
                  child: Text(auth.displayName,
                      style: const TextStyle(
                          color: AppColors.text,
                          fontSize: 20,
                          fontWeight: FontWeight.w700)),
                ),
                const SizedBox(height: 4),
                Center(
                  child: Text(
                      auth.email.isEmpty ? 'No email' : auth.email,
                      style: const TextStyle(
                          color: AppColors.textDim, fontSize: 13)),
                ),
                const SizedBox(height: 24),
                Row(children: [
                  Expanded(
                      child: _statCard('${streak.streak}', 'Day Streak',
                          Icons.local_fire_department)),
                  const SizedBox(width: 12),
                  Expanded(
                      child: _statCard('${prog.learnedCount}',
                          'Signs Mastered', Icons.sign_language)),
                ]),
                const SizedBox(height: 24),
                _sectionLabel('SETTINGS'),
                const SizedBox(height: 10),
                _settingTile(
                  icon: Icons.language,
                  title: 'Language',
                  value: settings.langLabel,
                  onTap: _editLanguage,
                ),
                _settingTile(
                  icon: Icons.speed,
                  title: 'TTS Speed',
                  value: settings.ttsSpeedLabel,
                  onTap: _editTtsSpeed,
                ),
                _settingTile(
                  icon: Icons.tune,
                  title: 'Confidence Threshold',
                  value: '${(settings.confidence * 100).toInt()}%',
                  onTap: _editConfidence,
                ),
                _settingTile(
                  icon: Icons.dns,
                  title: 'Inference Server',
                  value: _shortHost(settings.serverUrl),
                  onTap: _editServerUrl,
                ),
                const SizedBox(height: 24),
                _sectionLabel('PROGRESS'),
                const SizedBox(height: 10),
                _settingTile(
                  icon: Icons.refresh,
                  title: 'Reset Learned Signs',
                  value: '${prog.learnedCount} learned',
                  onTap: _confirmResetProgress,
                ),
                _settingTile(
                  icon: Icons.restart_alt,
                  title: 'Reset Day Streak',
                  value: '${streak.streak} days',
                  onTap: _confirmResetStreak,
                ),
                const SizedBox(height: 24),
                _sectionLabel('ABOUT'),
                const SizedBox(height: 10),
                _settingTile(
                  icon: Icons.info_outline,
                  title: 'Version',
                  value: '1.0.0',
                  trailingArrow: false,
                ),
                _settingTile(
                  icon: Icons.code,
                  title: 'Model',
                  value: 'PSL Words v2 (64 classes)',
                  trailingArrow: false,
                ),
                const SizedBox(height: 24),
                _buildSignOut(),
                const SizedBox(height: 24),
              ],
            );
          },
        ),
      ),
    );
  }

  // ── tiles ──────────────────────────────────────────────────────────────

  Widget _buildAvatar(String name) {
    final initial =
        name.trim().isEmpty ? '?' : name.trim()[0].toUpperCase();
    return Center(
      child: Stack(
        alignment: Alignment.bottomRight,
        children: [
          Container(
            width: 84,
            height: 84,
            alignment: Alignment.center,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: AppColors.bgCard,
              border: Border.all(color: AppColors.accent, width: 2),
            ),
            child: Text(initial,
                style: const TextStyle(
                    color: AppColors.accent,
                    fontSize: 36,
                    fontWeight: FontWeight.w800)),
          ),
          GestureDetector(
            onTap: _editName,
            child: Container(
              width: 26,
              height: 26,
              decoration: BoxDecoration(
                color: AppColors.accent,
                shape: BoxShape.circle,
                border: Border.all(color: AppColors.bg, width: 2),
              ),
              child: const Icon(Icons.edit,
                  color: Color(0xFF0B1020), size: 12),
            ),
          ),
        ],
      ),
    );
  }

  Widget _statCard(String value, String label, IconData icon) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.bgCard,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        children: [
          Icon(icon, color: AppColors.accent, size: 22),
          const SizedBox(height: 8),
          Text(value,
              style: const TextStyle(
                  color: AppColors.text,
                  fontSize: 22,
                  fontWeight: FontWeight.w800)),
          const SizedBox(height: 2),
          Text(label,
              style:
                  const TextStyle(color: AppColors.textDim, fontSize: 11),
              textAlign: TextAlign.center),
        ],
      ),
    );
  }

  Widget _sectionLabel(String text) => Text(
        text,
        style: const TextStyle(
          color: AppColors.textDim,
          fontSize: 10,
          fontWeight: FontWeight.w700,
          letterSpacing: 1.6,
        ),
      );

  Widget _settingTile({
    required IconData icon,
    required String title,
    required String value,
    VoidCallback? onTap,
    bool trailingArrow = true,
  }) {
    return GestureDetector(
      onTap: onTap,
      behavior: HitTestBehavior.opaque,
      child: Container(
        margin: const EdgeInsets.only(bottom: 8),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 14),
        decoration: BoxDecoration(
          color: AppColors.bgCard,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: AppColors.border),
        ),
        child: Row(
          children: [
            Icon(icon, color: AppColors.textDim, size: 18),
            const SizedBox(width: 12),
            Expanded(
                child: Text(title,
                    style: const TextStyle(color: AppColors.text))),
            Text(value,
                style: const TextStyle(
                    color: AppColors.textDim, fontSize: 13)),
            if (trailingArrow && onTap != null) ...[
              const SizedBox(width: 4),
              const Icon(Icons.chevron_right,
                  color: AppColors.textDim, size: 16),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildSignOut() {
    return GestureDetector(
      onTap: _confirmSignOut,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 14),
        decoration: BoxDecoration(
          color: AppColors.bgCard,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: AppColors.err.withAlpha(80)),
        ),
        child: const Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.logout, color: AppColors.err, size: 18),
            SizedBox(width: 8),
            Text('Sign Out',
                style: TextStyle(
                    color: AppColors.err, fontWeight: FontWeight.w700)),
          ],
        ),
      ),
    );
  }

  // ── dialogs ────────────────────────────────────────────────────────────

  Future<void> _editName() async {
    final ctrl =
        TextEditingController(text: AuthService.instance.displayName);
    final newName = await showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: AppColors.bgCard,
        title:
            const Text('Edit Name', style: TextStyle(color: AppColors.text)),
        content: TextField(
          controller: ctrl,
          autofocus: true,
          style: const TextStyle(color: AppColors.text),
          decoration: const InputDecoration(
            hintText: 'Your name',
            hintStyle: TextStyle(color: AppColors.textDim),
            enabledBorder: UnderlineInputBorder(
                borderSide: BorderSide(color: AppColors.border)),
            focusedBorder: UnderlineInputBorder(
                borderSide: BorderSide(color: AppColors.accent)),
          ),
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(ctx),
              child: const Text('Cancel',
                  style: TextStyle(color: AppColors.textDim))),
          TextButton(
              onPressed: () => Navigator.pop(ctx, ctrl.text.trim()),
              child: const Text('Save',
                  style: TextStyle(color: AppColors.accent))),
        ],
      ),
    );
    if (newName == null || newName.isEmpty) return;
    try {
      await AuthService.instance.updateName(newName);
    } catch (e) {
      _toast('Could not update name: $e');
    }
  }

  Future<void> _editLanguage() async {
    final current = SettingsService.instance.lang;
    final next = await showModalBottomSheet<String>(
      context: context,
      backgroundColor: AppColors.bgCard,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(18)),
      ),
      builder: (ctx) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const SizedBox(height: 12),
            const Text('Spoken Language',
                style: TextStyle(
                    color: AppColors.text,
                    fontSize: 16,
                    fontWeight: FontWeight.w700)),
            const SizedBox(height: 12),
            _langOption(ctx, current, 'urdu', 'اردو (Urdu)',
                'Speak translations in Urdu'),
            _langOption(ctx, current, 'english', 'English',
                'Speak translations in English'),
            const SizedBox(height: 8),
          ],
        ),
      ),
    );
    if (next != null) {
      await SettingsService.instance.setLang(next);
    }
  }

  Widget _langOption(BuildContext ctx, String current, String value,
      String title, String subtitle) {
    final selected = current == value;
    return ListTile(
      onTap: () => Navigator.pop(ctx, value),
      leading: Icon(
          selected ? Icons.radio_button_checked : Icons.radio_button_off,
          color: selected ? AppColors.accent : AppColors.textDim),
      title: Text(title, style: const TextStyle(color: AppColors.text)),
      subtitle: Text(subtitle,
          style: const TextStyle(color: AppColors.textDim, fontSize: 12)),
    );
  }

  Future<void> _editTtsSpeed() async {
    double v = SettingsService.instance.ttsSpeed;
    await showModalBottomSheet(
      context: context,
      backgroundColor: AppColors.bgCard,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(18)),
      ),
      builder: (ctx) => StatefulBuilder(
        builder: (_, setLocal) => SafeArea(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 18, 20, 24),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('TTS Speed',
                    style: TextStyle(
                        color: AppColors.text,
                        fontSize: 16,
                        fontWeight: FontWeight.w700)),
                const SizedBox(height: 4),
                const Text(
                    'How fast spoken translations are read aloud.',
                    style: TextStyle(
                        color: AppColors.textDim, fontSize: 12)),
                const SizedBox(height: 12),
                Row(children: [
                  const Text('Slow',
                      style: TextStyle(
                          color: AppColors.textDim, fontSize: 12)),
                  Expanded(
                    child: Slider(
                      value: v,
                      min: 0.20,
                      max: 0.90,
                      divisions: 14,
                      label: v.toStringAsFixed(2),
                      activeColor: AppColors.accent,
                      onChanged: (nv) => setLocal(() => v = nv),
                    ),
                  ),
                  const Text('Fast',
                      style: TextStyle(
                          color: AppColors.textDim, fontSize: 12)),
                ]),
                Center(
                  child: Text('${v.toStringAsFixed(2)}x',
                      style: const TextStyle(
                          color: AppColors.accent,
                          fontSize: 14,
                          fontWeight: FontWeight.w700)),
                ),
                const SizedBox(height: 12),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: () async {
                      await SettingsService.instance.setTtsSpeed(v);
                      if (ctx.mounted) Navigator.pop(ctx);
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppColors.accent,
                      foregroundColor: const Color(0xFF0B1020),
                      padding: const EdgeInsets.symmetric(vertical: 12),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10),
                      ),
                    ),
                    child: const Text('Save',
                        style: TextStyle(fontWeight: FontWeight.w800)),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Future<void> _editConfidence() async {
    double v = SettingsService.instance.confidence;
    await showModalBottomSheet(
      context: context,
      backgroundColor: AppColors.bgCard,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(18)),
      ),
      builder: (ctx) => StatefulBuilder(
        builder: (_, setLocal) => SafeArea(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 18, 20, 24),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Confidence Threshold',
                    style: TextStyle(
                        color: AppColors.text,
                        fontSize: 16,
                        fontWeight: FontWeight.w700)),
                const SizedBox(height: 4),
                const Text(
                    'Predictions below this confidence are dimmed in the UI. Higher = stricter.',
                    style: TextStyle(
                        color: AppColors.textDim, fontSize: 12)),
                const SizedBox(height: 12),
                Slider(
                  value: v,
                  min: 0.30,
                  max: 0.95,
                  divisions: 13,
                  label: '${(v * 100).toInt()}%',
                  activeColor: AppColors.accent,
                  onChanged: (nv) => setLocal(() => v = nv),
                ),
                Center(
                  child: Text('${(v * 100).toInt()}%',
                      style: const TextStyle(
                          color: AppColors.accent,
                          fontSize: 14,
                          fontWeight: FontWeight.w700)),
                ),
                const SizedBox(height: 12),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: () async {
                      await SettingsService.instance.setConfidence(v);
                      if (ctx.mounted) Navigator.pop(ctx);
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppColors.accent,
                      foregroundColor: const Color(0xFF0B1020),
                      padding: const EdgeInsets.symmetric(vertical: 12),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10),
                      ),
                    ),
                    child: const Text('Save',
                        style: TextStyle(fontWeight: FontWeight.w800)),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  String _shortHost(String url) {
    try {
      final u = Uri.parse(url);
      return '${u.host}:${u.port}';
    } catch (_) {
      return url;
    }
  }

  Future<void> _editServerUrl() async {
    final ctrl =
        TextEditingController(text: SettingsService.instance.serverUrl);
    final next = await showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: AppColors.bgCard,
        title: const Text('Inference Server',
            style: TextStyle(color: AppColors.text)),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'WebSocket URL of the PSL server running on your computer.',
              style: TextStyle(color: AppColors.textDim, fontSize: 12),
            ),
            const SizedBox(height: 10),
            TextField(
              controller: ctrl,
              autofocus: true,
              keyboardType: TextInputType.url,
              autocorrect: false,
              style: const TextStyle(color: AppColors.text, fontSize: 13),
              decoration: const InputDecoration(
                hintText: 'ws://192.168.1.16:8000/ws/translate',
                hintStyle: TextStyle(color: AppColors.textDim, fontSize: 12),
                enabledBorder: UnderlineInputBorder(
                    borderSide: BorderSide(color: AppColors.border)),
                focusedBorder: UnderlineInputBorder(
                    borderSide: BorderSide(color: AppColors.accent)),
              ),
            ),
            const SizedBox(height: 8),
            const Text(
              'Default: ${SettingsService.defaultServerUrl}',
              style: TextStyle(color: AppColors.textDim, fontSize: 10),
            ),
          ],
        ),
        actions: [
          TextButton(
              onPressed: () async {
                await SettingsService.instance.resetServerUrl();
                if (ctx.mounted) Navigator.pop(ctx);
              },
              child: const Text('Reset',
                  style: TextStyle(color: AppColors.warn))),
          TextButton(
              onPressed: () => Navigator.pop(ctx),
              child: const Text('Cancel',
                  style: TextStyle(color: AppColors.textDim))),
          TextButton(
              onPressed: () => Navigator.pop(ctx, ctrl.text.trim()),
              child: const Text('Save',
                  style: TextStyle(color: AppColors.accent))),
        ],
      ),
    );
    if (next == null || next.isEmpty) return;
    if (!next.startsWith('ws://') && !next.startsWith('wss://')) {
      _toast('Server URL must start with ws:// or wss://');
      return;
    }
    await SettingsService.instance.setServerUrl(next);
    _toast('Server URL updated');
  }

  Future<void> _confirmResetProgress() async {
    final n = ProgressService.instance.learnedCount;
    final ok = await _confirm(
      title: 'Reset Learned Signs?',
      message:
          'This clears your $n signs-mastered progress. Cannot be undone.',
      destructive: true,
      okLabel: 'Reset',
    );
    if (ok) await ProgressService.instance.resetAll();
  }

  Future<void> _confirmResetStreak() async {
    final ok = await _confirm(
      title: 'Reset Day Streak?',
      message:
          'Your ${StreakService.instance.streak}-day streak will be cleared.',
      destructive: true,
      okLabel: 'Reset',
    );
    if (ok) await StreakService.instance.reset();
  }

  Future<void> _confirmSignOut() async {
    final ok = await _confirm(
      title: 'Sign Out?',
      message: 'You\'ll need to sign back in to use the app.',
      destructive: true,
      okLabel: 'Sign Out',
    );
    if (!ok) return;
    try {
      await AuthService.instance.signOut();
    } catch (e) {
      _toast('Sign out failed: $e');
    }
  }

  Future<bool> _confirm({
    required String title,
    required String message,
    required String okLabel,
    bool destructive = false,
  }) async {
    final res = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: AppColors.bgCard,
        title: Text(title, style: const TextStyle(color: AppColors.text)),
        content: Text(message,
            style: const TextStyle(color: AppColors.textDim)),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(ctx, false),
              child: const Text('Cancel',
                  style: TextStyle(color: AppColors.textDim))),
          TextButton(
              onPressed: () => Navigator.pop(ctx, true),
              child: Text(okLabel,
                  style: TextStyle(
                      color:
                          destructive ? AppColors.err : AppColors.accent,
                      fontWeight: FontWeight.w700))),
        ],
      ),
    );
    return res == true;
  }

  void _toast(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
        .showSnackBar(SnackBar(content: Text(msg)));
  }
}

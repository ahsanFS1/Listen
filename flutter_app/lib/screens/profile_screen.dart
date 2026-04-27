import 'package:flutter/material.dart';
import '../theme/app_colors.dart';

class ProfileScreen extends StatelessWidget {
  const ProfileScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(18),
          children: [
            const SizedBox(height: 8),
            // Avatar
            Center(
              child: Stack(
                alignment: Alignment.bottomRight,
                children: [
                  Container(
                    width: 84, height: 84,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: AppColors.bgCard,
                      border: Border.all(color: AppColors.accent, width: 2),
                    ),
                    child: const Icon(Icons.person,
                        color: AppColors.accent, size: 40),
                  ),
                  Container(
                    width: 26, height: 26,
                    decoration: BoxDecoration(
                      color: AppColors.accent,
                      shape: BoxShape.circle,
                      border: Border.all(color: AppColors.bg, width: 2),
                    ),
                    child: const Icon(Icons.edit,
                        color: Color(0xFF0B1020), size: 12),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 12),
            const Center(
              child: Text('PSL Learner',
                  style: TextStyle(
                      color: AppColors.text,
                      fontSize: 20,
                      fontWeight: FontWeight.w700)),
            ),
            const SizedBox(height: 4),
            const Center(
              child: Text('learner@listen.app',
                  style: TextStyle(color: AppColors.textDim, fontSize: 13)),
            ),
            const SizedBox(height: 24),

            // Stats
            Row(
              children: [
                Expanded(child: _statCard('0', 'Day Streak', Icons.local_fire_department)),
                const SizedBox(width: 12),
                Expanded(child: _statCard('0', 'Signs Mastered', Icons.sign_language)),
              ],
            ),
            const SizedBox(height: 24),

            // Settings
            _sectionLabel('SETTINGS'),
            const SizedBox(height: 10),
            _settingRow(Icons.language, 'Language', 'English / اردو'),
            _settingRow(Icons.speed, 'TTS Speed', 'Normal'),
            _settingRow(Icons.tune, 'Confidence Threshold', '70%'),
            const SizedBox(height: 24),

            _sectionLabel('ABOUT'),
            const SizedBox(height: 10),
            _settingRow(Icons.info_outline, 'Version', '1.0.0'),
            _settingRow(Icons.code, 'Model', 'PSL Words v2 (64 classes)'),
            const SizedBox(height: 24),

            // Sign out
            GestureDetector(
              onTap: () {},
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
                            color: AppColors.err,
                            fontWeight: FontWeight.w700)),
                  ],
                ),
              ),
            ),
          ],
        ),
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
              style: const TextStyle(
                  color: AppColors.textDim, fontSize: 11),
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

  Widget _settingRow(IconData icon, String title, String value) {
    return Container(
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
          const SizedBox(width: 4),
          const Icon(Icons.chevron_right,
              color: AppColors.textDim, size: 16),
        ],
      ),
    );
  }
}

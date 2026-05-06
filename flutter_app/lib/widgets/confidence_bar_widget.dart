import 'package:flutter/material.dart';
import '../theme/app_colors.dart';

class ConfidenceBar extends StatelessWidget {
  final double value; // 0.0 – 1.0
  final double height;

  const ConfidenceBar({super.key, required this.value, this.height = 6});

  @override
  Widget build(BuildContext context) {
    final clamped = value.clamp(0.0, 1.0);
    return LayoutBuilder(
      builder: (_, constraints) {
        final w = constraints.maxWidth;
        return Container(
          height: height,
          decoration: BoxDecoration(
            color: AppColors.border,
            borderRadius: BorderRadius.circular(999),
          ),
          child: Align(
            alignment: Alignment.centerLeft,
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 120),
              width: w * clamped,
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [Color(0xFF0099CC), AppColors.accent],
                ),
                borderRadius: BorderRadius.circular(999),
              ),
            ),
          ),
        );
      },
    );
  }
}

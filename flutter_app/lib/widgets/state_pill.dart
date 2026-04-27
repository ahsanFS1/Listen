import 'package:flutter/material.dart';
import '../ml/sign_pipeline.dart';
import '../theme/app_colors.dart';

class StatePill extends StatefulWidget {
  final SignState state;
  final bool hasHands;
  final bool cameraOn;

  const StatePill({
    super.key,
    required this.state,
    required this.hasHands,
    required this.cameraOn,
  });

  @override
  State<StatePill> createState() => _StatePillState();
}

class _StatePillState extends State<StatePill>
    with SingleTickerProviderStateMixin {
  late AnimationController _pulse;
  late Animation<double> _opacity;

  bool get _shouldAnimate =>
      widget.cameraOn &&
      (widget.state == SignState.predicting ||
       widget.state == SignState.signing);

  @override
  void initState() {
    super.initState();
    _pulse = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 700),
    );
    _opacity = Tween<double>(begin: 0.4, end: 1.0).animate(
      CurvedAnimation(parent: _pulse, curve: Curves.easeInOut),
    );
    // Don't start animating until actually needed
    if (_shouldAnimate) _pulse.repeat(reverse: true);
  }

  @override
  void didUpdateWidget(StatePill oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (_shouldAnimate && !_pulse.isAnimating) {
      _pulse.repeat(reverse: true);
    } else if (!_shouldAnimate && _pulse.isAnimating) {
      _pulse.stop();
      _pulse.value = 1.0;
    }
  }

  @override
  void dispose() {
    _pulse.dispose();
    super.dispose();
  }

  String get _label {
    if (!widget.cameraOn)   return 'OFFLINE';
    if (!widget.hasHands)   return 'SHOW HANDS';
    return switch (widget.state) {
      SignState.idle       => 'SCANNING',
      SignState.signing    => 'SIGNING',
      SignState.predicting => 'PREDICTING',
      SignState.committed  => 'COMMITTED',
      SignState.cooldown   => 'COOLDOWN',
    };
  }

  Color get _color {
    if (!widget.cameraOn) return AppColors.textDim;
    if (!widget.hasHands) return AppColors.textDim;
    return switch (widget.state) {
      SignState.idle       => AppColors.textDim,
      SignState.signing    => AppColors.warn,
      SignState.predicting => AppColors.accent,
      SignState.committed  => AppColors.ok,
      SignState.cooldown   => AppColors.err,
    };
  }

  @override
  Widget build(BuildContext context) {
    final color = _color;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
      decoration: BoxDecoration(
        color: AppColors.bgSoft,
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: color, width: 1),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          _shouldAnimate
              ? FadeTransition(opacity: _opacity, child: _dot(color))
              : _dot(color),
          const SizedBox(width: 8),
          Text(
            _label,
            style: TextStyle(
              color: color,
              fontWeight: FontWeight.w700,
              fontSize: 12,
              letterSpacing: 1.5,
            ),
          ),
        ],
      ),
    );
  }

  Widget _dot(Color color) => Container(
        width: 7, height: 7,
        decoration: BoxDecoration(color: color, shape: BoxShape.circle),
      );
}

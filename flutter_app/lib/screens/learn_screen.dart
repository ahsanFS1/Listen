import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

import '../data/signs.dart';
import '../services/progress_service.dart';
import '../theme/app_colors.dart';
import 'quiz_screen.dart';
import 'word_video_screen.dart';

/// Kept around because the alphabet screen still launches PSL Dictionary
/// for letter videos (no .mp4 bundle for letters yet). Word taps now use
/// the in-app video player instead.
Future<void> openPslUrl(BuildContext context, String url) async {
  final uri = Uri.parse(url);
  bool ok;
  try {
    ok = await launchUrl(uri, mode: LaunchMode.externalApplication);
  } catch (_) {
    ok = false;
  }
  if (!ok) {
    try {
      ok = await launchUrl(uri);
    } catch (_) {
      ok = false;
    }
  }
  if (!ok && context.mounted) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Could not open ${uri.toString()}')),
    );
  }
}

void openSignVideo(BuildContext context, SignInfo s) {
  Navigator.of(context).push(MaterialPageRoute(
    builder: (_) => WordVideoScreen(sign: s),
  ));
}

class LearnScreen extends StatefulWidget {
  const LearnScreen({super.key});

  @override
  State<LearnScreen> createState() => _LearnScreenState();
}

class _LearnScreenState extends State<LearnScreen> {
  String _search = '';

  @override
  void initState() {
    super.initState();
    ProgressService.instance.load();
  }

  Map<String, List<SignInfo>> get _grouped {
    final filtered = kSigns
        .where((s) =>
            _search.isEmpty ||
            s.english.toLowerCase().contains(_search.toLowerCase()) ||
            s.urdu.contains(_search))
        .toList();
    final map = <String, List<SignInfo>>{};
    for (final s in filtered) {
      map.putIfAbsent(s.category, () => []).add(s);
    }
    return map;
  }

  @override
  Widget build(BuildContext context) {
    final grouped = _grouped;
    return Scaffold(
      backgroundColor: AppColors.bg,
      body: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(18, 16, 18, 0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Master PSL',
                      style: TextStyle(
                          color: AppColors.text,
                          fontSize: 26,
                          fontWeight: FontWeight.w800)),
                  const SizedBox(height: 4),
                  const Text('Learn Pakistani Sign Language signs',
                      style:
                          TextStyle(color: AppColors.textDim, fontSize: 13)),
                  const SizedBox(height: 16),
                  Container(
                    height: 44,
                    decoration: BoxDecoration(
                      color: AppColors.bgCard,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: AppColors.border),
                    ),
                    child: TextField(
                      style: const TextStyle(color: AppColors.text),
                      decoration: const InputDecoration(
                        hintText: 'Search signs...',
                        hintStyle: TextStyle(color: AppColors.textDim),
                        prefixIcon:
                            Icon(Icons.search, color: AppColors.textDim),
                        border: InputBorder.none,
                        contentPadding: EdgeInsets.symmetric(vertical: 12),
                      ),
                      onChanged: (v) => setState(() => _search = v),
                    ),
                  ),
                  const SizedBox(height: 16),
                  _buildProgressCard(),
                  const SizedBox(height: 12),
                  _buildQuizCta(),
                  const SizedBox(height: 8),
                ],
              ),
            ),
            Expanded(
              child: grouped.isEmpty
                  ? const Center(
                      child: Text('No signs found',
                          style: TextStyle(color: AppColors.textDim)))
                  : ListView(
                      padding: const EdgeInsets.fromLTRB(18, 8, 18, 24),
                      children: grouped.entries
                          .map((e) => _CategoryCard(
                                category: e.key,
                                signs: e.value,
                              ))
                          .toList(),
                    ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildProgressCard() {
    return AnimatedBuilder(
      animation: ProgressService.instance,
      builder: (_, __) {
        final p = ProgressService.instance;
        return Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: AppColors.bgCard,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: AppColors.border),
          ),
          child: Row(
            children: [
              _CircleProgress(fraction: p.fraction),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('Overall Progress',
                        style: TextStyle(
                            color: AppColors.text,
                            fontWeight: FontWeight.w700)),
                    const SizedBox(height: 2),
                    Text('${p.learnedCount} / ${p.total} Signs',
                        style: const TextStyle(
                            color: AppColors.textDim, fontSize: 13)),
                    const SizedBox(height: 8),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(999),
                      child: LinearProgressIndicator(
                        value: p.fraction,
                        minHeight: 6,
                        backgroundColor: AppColors.bgSoft,
                        valueColor:
                            const AlwaysStoppedAnimation(AppColors.accent),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildQuizCta() {
    return GestureDetector(
      onTap: () => Navigator.of(context).push(MaterialPageRoute(
        builder: (_) => const QuizScreen(),
      )),
      child: Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          gradient: const LinearGradient(
            colors: [Color(0xFF06b6d4), Color(0xFF3b82f6)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          borderRadius: BorderRadius.circular(14),
        ),
        child: const Row(children: [
          Icon(Icons.quiz_outlined, color: Colors.white),
          SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Take the 5-Sign Quiz',
                    style: TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.w800,
                        fontSize: 15)),
                Text('Sign 5 prompts in front of the camera',
                    style: TextStyle(color: Colors.white70, fontSize: 12)),
              ],
            ),
          ),
          Icon(Icons.chevron_right, color: Colors.white),
        ]),
      ),
    );
  }
}

class _CategoryCard extends StatefulWidget {
  final String category;
  final List<SignInfo> signs;
  const _CategoryCard({required this.category, required this.signs});

  @override
  State<_CategoryCard> createState() => _CategoryCardState();
}

class _CategoryCardState extends State<_CategoryCard> {
  bool _expanded = false;

  String get _title =>
      widget.category[0].toUpperCase() + widget.category.substring(1);

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: ProgressService.instance,
      builder: (_, __) {
        final learnedInGroup = widget.signs
            .where((s) => ProgressService.instance.isLearned(s.id))
            .length;
        return Container(
          margin: const EdgeInsets.only(bottom: 10),
          decoration: BoxDecoration(
            color: AppColors.bgCard,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: AppColors.border),
          ),
          child: Column(
            children: [
              GestureDetector(
                onTap: () => setState(() => _expanded = !_expanded),
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Row(
                    children: [
                      Container(
                        width: 42,
                        height: 42,
                        decoration: BoxDecoration(
                          color: AppColors.bgSoft,
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(color: AppColors.border),
                        ),
                        child: Icon(_categoryIcon(widget.category),
                            color: AppColors.accent, size: 20),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(_title,
                                style: const TextStyle(
                                    color: AppColors.text,
                                    fontWeight: FontWeight.w700,
                                    fontSize: 15)),
                            Text('$learnedInGroup / ${widget.signs.length} learned',
                                style: const TextStyle(
                                    color: AppColors.textDim, fontSize: 12)),
                          ],
                        ),
                      ),
                      Icon(
                        _expanded
                            ? Icons.keyboard_arrow_up
                            : Icons.keyboard_arrow_down,
                        color: AppColors.textDim,
                      ),
                    ],
                  ),
                ),
              ),
              if (_expanded)
                Padding(
                  padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                  child: Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    children: widget.signs.map((s) {
                      final learned =
                          ProgressService.instance.isLearned(s.id);
                      return GestureDetector(
                        onTap: () => openSignVideo(context, s),
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 12, vertical: 8),
                          decoration: BoxDecoration(
                            color: learned
                                ? AppColors.accent.withValues(alpha: 0.15)
                                : AppColors.bgSoft,
                            borderRadius: BorderRadius.circular(8),
                            border: Border.all(
                              color: learned
                                  ? AppColors.accent.withValues(alpha: 0.6)
                                  : AppColors.border,
                            ),
                          ),
                          child: Column(
                            children: [
                              Row(mainAxisSize: MainAxisSize.min, children: [
                                Text(s.english,
                                    style: const TextStyle(
                                        color: AppColors.text,
                                        fontSize: 12,
                                        fontWeight: FontWeight.w600)),
                                const SizedBox(width: 4),
                                Icon(
                                  learned
                                      ? Icons.check_circle
                                      : Icons.play_circle_outline,
                                  color: learned
                                      ? AppColors.accent
                                      : AppColors.textDim,
                                  size: 13,
                                ),
                              ]),
                              Text(s.urdu,
                                  textDirection: TextDirection.rtl,
                                  style: const TextStyle(
                                      color: AppColors.accent, fontSize: 13)),
                            ],
                          ),
                        ),
                      );
                    }).toList(),
                  ),
                ),
            ],
          ),
        );
      },
    );
  }

  IconData _categoryIcon(String cat) {
    return switch (cat) {
      'greetings' => Icons.waving_hand,
      'animals' => Icons.pets,
      'transport' => Icons.directions_bus,
      'objects' => Icons.category,
      'bathroom' => Icons.water_drop,
      'places' => Icons.place,
      'expressions' => Icons.chat_bubble_outline,
      'actions' => Icons.gesture,
      'appearance' => Icons.face,
      'body' => Icons.accessibility_new,
      'alphabet' => Icons.abc,
      _ => Icons.sign_language,
    };
  }
}

class _CircleProgress extends StatelessWidget {
  final double fraction;
  const _CircleProgress({required this.fraction});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 48,
      height: 48,
      child: Stack(
        alignment: Alignment.center,
        children: [
          CircularProgressIndicator(
            value: fraction,
            backgroundColor: AppColors.bgSoft,
            color: AppColors.accent,
            strokeWidth: 4,
          ),
          Text(
            '${(fraction * 100).toInt()}%',
            style: const TextStyle(
                color: AppColors.text,
                fontSize: 11,
                fontWeight: FontWeight.w700),
          ),
        ],
      ),
    );
  }
}

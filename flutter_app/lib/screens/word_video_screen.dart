import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

import '../data/signs.dart';
import '../services/progress_service.dart';
import '../theme/app_colors.dart';
import 'fullscreen_video_screen.dart';

/// Plays the bundled PSL Dictionary demo video for [sign] and lets the
/// user mark / unmark it as learned. Replaces the previous behaviour of
/// kicking out to psl.org.pk in an external browser.
class WordVideoScreen extends StatefulWidget {
  final SignInfo sign;
  const WordVideoScreen({super.key, required this.sign});

  @override
  State<WordVideoScreen> createState() => _WordVideoScreenState();
}

class _WordVideoScreenState extends State<WordVideoScreen> {
  VideoPlayerController? _controller;
  String? _error;

  @override
  void initState() {
    super.initState();
    _initVideo();
  }

  Future<void> _initVideo() async {
    final asset = 'assets/videos/${widget.sign.id}.mp4';
    final c = VideoPlayerController.asset(asset);
    try {
      await c.initialize();
      await c.setLooping(true);
      await c.play();
      if (mounted) setState(() => _controller = c);
    } catch (e) {
      if (mounted) setState(() => _error = 'No video bundled for "${widget.sign.id}"');
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final s = widget.sign;
    return Scaffold(
      backgroundColor: AppColors.bg,
      appBar: AppBar(
        title: Text(s.english),
        backgroundColor: AppColors.bgCard,
      ),
      body: SafeArea(
        child: Column(
          children: [
            const SizedBox(height: 8),
            // Video stage
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: AspectRatio(
                aspectRatio: _controller?.value.isInitialized == true
                    ? _controller!.value.aspectRatio
                    : 16 / 9,
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(16),
                  child: Container(
                    color: Colors.black,
                    alignment: Alignment.center,
                    child: _error != null
                        ? Padding(
                            padding: const EdgeInsets.all(16),
                            child: Text(_error!,
                                textAlign: TextAlign.center,
                                style: const TextStyle(color: AppColors.textDim)),
                          )
                        : (_controller?.value.isInitialized == true
                            ? VideoPlayer(_controller!)
                            : const CircularProgressIndicator(
                                color: AppColors.accent)),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 16),
            // Word + urdu
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: AppColors.bgCard,
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: AppColors.border),
                ),
                child: Row(children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(s.english,
                            style: const TextStyle(
                                color: AppColors.text,
                                fontSize: 22,
                                fontWeight: FontWeight.w800)),
                        const SizedBox(height: 4),
                        Text(s.category,
                            style: const TextStyle(
                                color: AppColors.textDim, fontSize: 12)),
                      ],
                    ),
                  ),
                  Text(s.urdu,
                      textDirection: TextDirection.rtl,
                      style: const TextStyle(
                          color: AppColors.accent,
                          fontSize: 26,
                          fontWeight: FontWeight.w700)),
                ]),
              ),
            ),
            const Spacer(),
            // Action buttons
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
              child: AnimatedBuilder(
                animation: ProgressService.instance,
                builder: (_, __) {
                  final learned = ProgressService.instance.isLearned(s.id);
                  return Row(children: [
                    if (_controller != null) ...[
                      _CircleBtn(
                        icon: _controller!.value.isPlaying
                            ? Icons.pause
                            : Icons.play_arrow,
                        onTap: () {
                          setState(() {
                            _controller!.value.isPlaying
                                ? _controller!.pause()
                                : _controller!.play();
                          });
                        },
                      ),
                      const SizedBox(width: 12),
                      _CircleBtn(
                        icon: Icons.fullscreen,
                        onTap: () {
                          Navigator.of(context).push(MaterialPageRoute(
                            builder: (_) => FullscreenVideoScreen(
                              controller: _controller!,
                              title: s.english,
                            ),
                          ));
                        },
                      ),
                      const SizedBox(width: 12),
                    ],
                    Expanded(
                      child: ElevatedButton.icon(
                        onPressed: () async {
                          if (learned) {
                            await ProgressService.instance.unmark(s.id);
                          } else {
                            await ProgressService.instance.markLearned(s.id);
                          }
                        },
                        style: ElevatedButton.styleFrom(
                          backgroundColor: learned
                              ? AppColors.bgSoft
                              : AppColors.accent,
                          foregroundColor: learned
                              ? AppColors.text
                              : const Color(0xFF0B1020),
                          padding: const EdgeInsets.symmetric(vertical: 14),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                        ),
                        icon: Icon(learned ? Icons.check : Icons.bookmark_add),
                        label: Text(
                          learned ? 'Learned' : 'Mark as Learned',
                          style: const TextStyle(fontWeight: FontWeight.w800),
                        ),
                      ),
                    ),
                  ]);
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _CircleBtn extends StatelessWidget {
  final IconData icon;
  final VoidCallback onTap;
  const _CircleBtn({required this.icon, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 48, height: 48,
        decoration: BoxDecoration(
          color: AppColors.bgSoft,
          shape: BoxShape.circle,
          border: Border.all(color: AppColors.border),
        ),
        child: Icon(icon, color: AppColors.accent),
      ),
    );
  }
}

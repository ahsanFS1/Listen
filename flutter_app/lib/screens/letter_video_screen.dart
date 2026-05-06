import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:video_player/video_player.dart';

import '../data/alphabets.dart';
import '../theme/app_colors.dart';
import 'fullscreen_video_screen.dart';

/// Plays the bundled PSL Dictionary demo video for a single Urdu
/// letter. If the asset is missing (one letter on PSL's CDN returns
/// 403 so we never managed to bundle it), falls back to a button that
/// opens the public PSL Dict alphabet page in the browser.
class LetterVideoScreen extends StatefulWidget {
  final PslLetter letter;
  const LetterVideoScreen({super.key, required this.letter});

  @override
  State<LetterVideoScreen> createState() => _LetterVideoScreenState();
}

class _LetterVideoScreenState extends State<LetterVideoScreen> {
  VideoPlayerController? _controller;
  String? _error;

  @override
  void initState() {
    super.initState();
    _initVideo();
  }

  Future<void> _initVideo() async {
    final c = VideoPlayerController.asset(widget.letter.videoAsset);
    try {
      await c.initialize();
      await c.setLooping(true);
      await c.play();
      if (mounted) setState(() => _controller = c);
    } catch (_) {
      if (mounted) {
        setState(() =>
            _error = 'Video unavailable for "${widget.letter.roman}"');
      }
    }
  }

  Future<void> _openExternal() async {
    final uri = Uri.parse(widget.letter.pslUrl);
    try {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    } catch (_) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Could not open ${uri.toString()}')),
        );
      }
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final l = widget.letter;
    return Scaffold(
      backgroundColor: AppColors.bg,
      appBar: AppBar(
        title: Text(l.roman),
        backgroundColor: AppColors.bgCard,
      ),
      body: SafeArea(
        child: Column(
          children: [
            const SizedBox(height: 8),
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
                            padding: const EdgeInsets.all(20),
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                const Icon(Icons.cloud_off,
                                    color: AppColors.textDim, size: 40),
                                const SizedBox(height: 12),
                                Text(_error!,
                                    textAlign: TextAlign.center,
                                    style: const TextStyle(
                                        color: AppColors.textDim)),
                                const SizedBox(height: 16),
                                ElevatedButton.icon(
                                  onPressed: _openExternal,
                                  icon: const Icon(Icons.open_in_new),
                                  label: const Text('Open on PSL.org.pk'),
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: AppColors.accent,
                                    foregroundColor: const Color(0xFF0B1020),
                                  ),
                                ),
                              ],
                            ),
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
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: AppColors.bgCard,
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: AppColors.border),
                ),
                child: Row(
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(l.roman,
                              style: const TextStyle(
                                  color: AppColors.text,
                                  fontSize: 22,
                                  fontWeight: FontWeight.w800)),
                          const SizedBox(height: 4),
                          const Text('Urdu Alphabet',
                              style: TextStyle(
                                  color: AppColors.textDim, fontSize: 12)),
                        ],
                      ),
                    ),
                    Text(l.urdu,
                        textDirection: TextDirection.rtl,
                        style: const TextStyle(
                            color: AppColors.accent,
                            fontSize: 56,
                            fontWeight: FontWeight.w700)),
                  ],
                ),
              ),
            ),
            const Spacer(),
            if (_controller != null && _error == null)
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                child: Row(children: [
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: () {
                        setState(() {
                          _controller!.value.isPlaying
                              ? _controller!.pause()
                              : _controller!.play();
                        });
                      },
                      style: ElevatedButton.styleFrom(
                        backgroundColor: AppColors.accent,
                        foregroundColor: const Color(0xFF0B1020),
                        padding: const EdgeInsets.symmetric(vertical: 14),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                      icon: Icon(_controller!.value.isPlaying
                          ? Icons.pause
                          : Icons.play_arrow),
                      label: Text(
                        _controller!.value.isPlaying ? 'Pause' : 'Play',
                        style: const TextStyle(
                            fontWeight: FontWeight.w800, fontSize: 15),
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  SizedBox(
                    height: 48,
                    width: 48,
                    child: ElevatedButton(
                      onPressed: () {
                        Navigator.of(context).push(MaterialPageRoute(
                          builder: (_) => FullscreenVideoScreen(
                            controller: _controller!,
                            title: l.roman,
                          ),
                        ));
                      },
                      style: ElevatedButton.styleFrom(
                        backgroundColor: AppColors.bgSoft,
                        foregroundColor: AppColors.accent,
                        padding: EdgeInsets.zero,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                          side: const BorderSide(color: AppColors.border),
                        ),
                      ),
                      child: const Icon(Icons.fullscreen),
                    ),
                  ),
                ]),
              ),
          ],
        ),
      ),
    );
  }
}

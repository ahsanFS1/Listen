import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:video_player/video_player.dart';

import '../theme/app_colors.dart';

/// Full-screen, landscape-locked video viewer used by both the word and
/// alphabet players. The caller passes its already-initialised
/// [VideoPlayerController] so we don't re-buffer the asset.
///
/// While this route is alive the device is forced into landscape and
/// the system UI is hidden; both are reverted on dispose so we don't
/// leave the rest of the app in a weird state.
class FullscreenVideoScreen extends StatefulWidget {
  final VideoPlayerController controller;
  final String title;
  const FullscreenVideoScreen({
    super.key,
    required this.controller,
    required this.title,
  });

  @override
  State<FullscreenVideoScreen> createState() => _FullscreenVideoScreenState();
}

class _FullscreenVideoScreenState extends State<FullscreenVideoScreen> {
  bool _showControls = true;

  @override
  void initState() {
    super.initState();
    SystemChrome.setEnabledSystemUIMode(SystemUiMode.immersiveSticky);
    SystemChrome.setPreferredOrientations([
      DeviceOrientation.landscapeLeft,
      DeviceOrientation.landscapeRight,
    ]);
    widget.controller.addListener(_onControllerUpdate);
  }

  void _onControllerUpdate() {
    if (mounted) setState(() {});
  }

  @override
  void dispose() {
    widget.controller.removeListener(_onControllerUpdate);
    SystemChrome.setEnabledSystemUIMode(SystemUiMode.edgeToEdge);
    SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
    super.dispose();
  }

  void _togglePlay() {
    setState(() {
      widget.controller.value.isPlaying
          ? widget.controller.pause()
          : widget.controller.play();
    });
  }

  @override
  Widget build(BuildContext context) {
    final c = widget.controller;
    return Scaffold(
      backgroundColor: Colors.black,
      body: GestureDetector(
        onTap: () => setState(() => _showControls = !_showControls),
        child: Stack(
          fit: StackFit.expand,
          children: [
            Center(
              child: AspectRatio(
                aspectRatio: c.value.aspectRatio == 0
                    ? 16 / 9
                    : c.value.aspectRatio,
                child: VideoPlayer(c),
              ),
            ),
            if (_showControls) ...[
              Positioned(
                top: 0, left: 0, right: 0,
                child: SafeArea(
                  child: Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: const BoxDecoration(
                      gradient: LinearGradient(
                        begin: Alignment.topCenter, end: Alignment.bottomCenter,
                        colors: [Colors.black87, Colors.transparent],
                      ),
                    ),
                    child: Row(children: [
                      IconButton(
                        icon: const Icon(Icons.close, color: Colors.white),
                        onPressed: () => Navigator.of(context).pop(),
                      ),
                      Expanded(
                        child: Text(widget.title,
                            style: const TextStyle(
                                color: Colors.white,
                                fontWeight: FontWeight.w700,
                                fontSize: 16)),
                      ),
                    ]),
                  ),
                ),
              ),
              Positioned(
                left: 0, right: 0, bottom: 0,
                child: SafeArea(
                  child: Container(
                    padding: const EdgeInsets.fromLTRB(16, 8, 16, 12),
                    decoration: const BoxDecoration(
                      gradient: LinearGradient(
                        begin: Alignment.bottomCenter, end: Alignment.topCenter,
                        colors: [Colors.black87, Colors.transparent],
                      ),
                    ),
                    child: Column(mainAxisSize: MainAxisSize.min, children: [
                      VideoProgressIndicator(
                        c,
                        allowScrubbing: true,
                        colors: const VideoProgressColors(
                          playedColor: AppColors.accent,
                          bufferedColor: Colors.white24,
                          backgroundColor: Colors.white12,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Row(children: [
                        IconButton(
                          iconSize: 36,
                          icon: Icon(
                            c.value.isPlaying
                                ? Icons.pause_circle_filled
                                : Icons.play_circle_filled,
                            color: AppColors.accent,
                          ),
                          onPressed: _togglePlay,
                        ),
                        const SizedBox(width: 8),
                        Text(
                          _fmt(c.value.position),
                          style: const TextStyle(color: Colors.white, fontSize: 12),
                        ),
                        const Text(' / ',
                            style: TextStyle(color: Colors.white54, fontSize: 12)),
                        Text(
                          _fmt(c.value.duration),
                          style: const TextStyle(color: Colors.white70, fontSize: 12),
                        ),
                      ]),
                    ]),
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  static String _fmt(Duration d) {
    final m = d.inMinutes.remainder(60).toString().padLeft(2, '0');
    final s = d.inSeconds.remainder(60).toString().padLeft(2, '0');
    return '$m:$s';
  }
}

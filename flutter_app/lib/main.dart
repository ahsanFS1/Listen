import 'package:camera_android/camera_android.dart';
import 'package:camera_platform_interface/camera_platform_interface.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'app.dart';
import 'services/auth_service.dart';
import 'services/settings_service.dart';
import 'services/streak_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Force Camera2 backend on Android — CameraX does JPEG compression on
  // Samsung devices which kills frame throughput.
  if (defaultTargetPlatform == TargetPlatform.android) {
    CameraPlatform.instance = AndroidCamera();
  }

  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);

  await AuthService.initSupabase();
  AuthService.instance.start();
  await SettingsService.instance.load();
  await StreakService.instance.touchToday();

  runApp(const ListenApp());
}

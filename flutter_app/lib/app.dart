import 'package:flutter/material.dart';
import 'theme/app_colors.dart';
import 'screens/translate_screen.dart';
import 'screens/learn_screen.dart';
import 'screens/dictionary_screen.dart';
import 'screens/profile_screen.dart';

class ListenApp extends StatelessWidget {
  const ListenApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Listen PSL',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        scaffoldBackgroundColor: AppColors.bg,
        colorScheme: const ColorScheme.dark(
          surface: AppColors.bg,
          primary: AppColors.accent,
          onPrimary: Color(0xFF0B1020),
          secondary: AppColors.accent,
        ),
        appBarTheme: const AppBarTheme(
          backgroundColor: AppColors.bg,
          foregroundColor: AppColors.text,
          elevation: 0,
        ),
        textTheme: const TextTheme(
          bodyMedium: TextStyle(color: AppColors.text),
          bodySmall:  TextStyle(color: AppColors.textDim),
        ),
      ),
      home: const _MainShell(),
    );
  }
}

class _MainShell extends StatefulWidget {
  const _MainShell();

  @override
  State<_MainShell> createState() => _MainShellState();
}

class _MainShellState extends State<_MainShell> {
  int _tab = 0;
  // Track which tabs have been visited so we only build them on first visit
  final Set<int> _visited = {0};

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      body: Stack(
        children: [
          _tabBody(0, const TranslateScreen()),
          _tabBody(1, const LearnScreen()),
          _tabBody(2, const DictionaryScreen()),
          _tabBody(3, const ProfileScreen()),
        ],
      ),
      bottomNavigationBar: _buildNavBar(),
    );
  }

  // Only build a tab when first visited; hide (not destroy) when inactive
  Widget _tabBody(int idx, Widget screen) {
    if (!_visited.contains(idx)) {
      return const SizedBox.shrink();
    }
    return Offstage(offstage: _tab != idx, child: screen);
  }

  Widget _buildNavBar() {
    return Container(
      decoration: const BoxDecoration(
        color: AppColors.bgCard,
        border: Border(top: BorderSide(color: AppColors.border, width: 1)),
      ),
      child: SafeArea(
        top: false,
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 6),
          child: Row(
            children: [
              _navItem(0, Icons.videocam_outlined,  Icons.videocam,  'Translate'),
              _navItem(1, Icons.school_outlined,    Icons.school,    'Learn'),
              _navItem(2, Icons.menu_book_outlined, Icons.menu_book, 'Dictionary'),
              _navItem(3, Icons.person_outline,     Icons.person,    'Profile'),
            ],
          ),
        ),
      ),
    );
  }

  Widget _navItem(int idx, IconData icon, IconData activeIcon, String label) {
    final active = _tab == idx;
    return Expanded(
      child: GestureDetector(
        onTap: () => setState(() {
          _tab = idx;
          _visited.add(idx); // mark as visited so it gets built
        }),
        behavior: HitTestBehavior.opaque,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              active ? activeIcon : icon,
              color: active ? AppColors.accent : AppColors.textDim,
              size: 24,
            ),
            const SizedBox(height: 3),
            Text(
              label,
              style: TextStyle(
                color: active ? AppColors.accent : AppColors.textDim,
                fontSize: 10,
                fontWeight: active ? FontWeight.w700 : FontWeight.normal,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

import 'package:flutter/material.dart';

import '../data/alphabets.dart';
import '../data/signs.dart';
import '../theme/app_colors.dart';
import 'learn_screen.dart' show openPslUrl;

const String _kPslDictionaryHome = 'https://psl.org.pk/dictionary';

class DictionaryScreen extends StatefulWidget {
  const DictionaryScreen({super.key});

  @override
  State<DictionaryScreen> createState() => _DictionaryScreenState();
}

class _DictionaryScreenState extends State<DictionaryScreen>
    with SingleTickerProviderStateMixin {
  late final TabController _tab;
  String _search = '';

  @override
  void initState() {
    super.initState();
    _tab = TabController(length: 2, vsync: this);
    _tab.addListener(() {
      if (mounted) setState(() {});
    });
  }

  @override
  void dispose() {
    _tab.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      body: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildHeader(),
            _buildSearch(),
            const SizedBox(height: 8),
            _buildTabBar(),
            Expanded(
              child: TabBarView(
                controller: _tab,
                children: [
                  _WordsList(search: _search),
                  _AlphabetsList(search: _search),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader() => Padding(
        padding: const EdgeInsets.fromLTRB(18, 16, 18, 8),
        child: Row(children: [
          const Expanded(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text('Dictionary',
                  style: TextStyle(
                      color: AppColors.text,
                      fontSize: 26,
                      fontWeight: FontWeight.w800)),
              SizedBox(height: 2),
              Text('Powered by psl.org.pk',
                  style: TextStyle(color: AppColors.textDim, fontSize: 12)),
            ]),
          ),
          GestureDetector(
            onTap: () => openPslUrl(context, _kPslDictionaryHome),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(
                color: AppColors.bgSoft,
                borderRadius: BorderRadius.circular(999),
                border: Border.all(color: AppColors.border),
              ),
              child: const Row(mainAxisSize: MainAxisSize.min, children: [
                Icon(Icons.open_in_new, color: AppColors.accent, size: 14),
                SizedBox(width: 6),
                Text('PSL Site',
                    style: TextStyle(
                        color: AppColors.accent,
                        fontSize: 12,
                        fontWeight: FontWeight.w700)),
              ]),
            ),
          ),
        ]),
      );

  Widget _buildSearch() => Padding(
        padding: const EdgeInsets.symmetric(horizontal: 18),
        child: Container(
          height: 44,
          decoration: BoxDecoration(
            color: AppColors.bgCard,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: AppColors.border),
          ),
          child: TextField(
            style: const TextStyle(color: AppColors.text),
            decoration: const InputDecoration(
              hintText: 'Search words or letters...',
              hintStyle: TextStyle(color: AppColors.textDim),
              prefixIcon: Icon(Icons.search, color: AppColors.textDim),
              border: InputBorder.none,
              contentPadding: EdgeInsets.symmetric(vertical: 12),
            ),
            onChanged: (v) => setState(() => _search = v),
          ),
        ),
      );

  Widget _buildTabBar() => Padding(
        padding: const EdgeInsets.fromLTRB(18, 12, 18, 4),
        child: Container(
          padding: const EdgeInsets.all(4),
          decoration: BoxDecoration(
            color: AppColors.bgSoft,
            borderRadius: BorderRadius.circular(999),
            border: Border.all(color: AppColors.border),
          ),
          child: TabBar(
            controller: _tab,
            indicator: BoxDecoration(
              color: AppColors.accent,
              borderRadius: BorderRadius.circular(999),
            ),
            indicatorSize: TabBarIndicatorSize.tab,
            dividerColor: Colors.transparent,
            labelColor: const Color(0xFF0B1020),
            unselectedLabelColor: AppColors.textDim,
            labelStyle: const TextStyle(
                fontWeight: FontWeight.w700, fontSize: 13, letterSpacing: 0.5),
            tabs: const [
              Tab(text: 'Words'),
              Tab(text: 'Alphabets'),
            ],
          ),
        ),
      );
}

class _WordsList extends StatelessWidget {
  final String search;
  const _WordsList({required this.search});

  @override
  Widget build(BuildContext context) {
    final q = search.toLowerCase().trim();
    final items = kSigns
        .where((s) =>
            q.isEmpty ||
            s.english.toLowerCase().contains(q) ||
            s.urdu.contains(q) ||
            s.id.toLowerCase().contains(q))
        .toList()
      ..sort((a, b) => a.english.compareTo(b.english));

    if (items.isEmpty) {
      return const Center(
          child: Text('No matching words',
              style: TextStyle(color: AppColors.textDim)));
    }

    return ListView.separated(
      padding: const EdgeInsets.fromLTRB(18, 12, 18, 24),
      itemCount: items.length,
      separatorBuilder: (_, __) => const SizedBox(height: 8),
      itemBuilder: (ctx, i) {
        final s = items[i];
        return _DictRow(
          left: s.english,
          right: s.urdu,
          subtitle: s.category,
          onTap: () => openPslUrl(ctx, s.pslUrl),
        );
      },
    );
  }
}

class _AlphabetsList extends StatelessWidget {
  final String search;
  const _AlphabetsList({required this.search});

  @override
  Widget build(BuildContext context) {
    final q = search.toLowerCase().trim();
    final items = kPslLetters
        .where((l) =>
            q.isEmpty ||
            l.name.toLowerCase().contains(q) ||
            l.roman.toLowerCase().contains(q) ||
            l.urdu.contains(q))
        .toList();

    if (items.isEmpty) {
      return const Center(
          child: Text('No matching letters',
              style: TextStyle(color: AppColors.textDim)));
    }

    return GridView.builder(
      padding: const EdgeInsets.fromLTRB(18, 12, 18, 24),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 3,
        mainAxisSpacing: 10,
        crossAxisSpacing: 10,
        childAspectRatio: 0.95,
      ),
      itemCount: items.length,
      itemBuilder: (ctx, i) {
        final l = items[i];
        return GestureDetector(
          onTap: () => openPslUrl(ctx, l.pslUrl),
          child: Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: AppColors.bgCard,
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: AppColors.border),
            ),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(l.urdu,
                    textDirection: TextDirection.rtl,
                    style: const TextStyle(
                        color: AppColors.accent,
                        fontSize: 36,
                        fontWeight: FontWeight.w700)),
                const SizedBox(height: 4),
                Text(l.roman,
                    style: const TextStyle(
                        color: AppColors.text,
                        fontSize: 12,
                        fontWeight: FontWeight.w700)),
              ],
            ),
          ),
        );
      },
    );
  }
}

class _DictRow extends StatelessWidget {
  final String left;
  final String right;
  final String? subtitle;
  final VoidCallback onTap;
  const _DictRow({
    required this.left,
    required this.right,
    required this.onTap,
    this.subtitle,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
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
                  Text(left,
                      style: const TextStyle(
                          color: AppColors.text,
                          fontSize: 15,
                          fontWeight: FontWeight.w700)),
                  if (subtitle != null && subtitle!.isNotEmpty) ...[
                    const SizedBox(height: 2),
                    Text(subtitle!,
                        style: const TextStyle(
                            color: AppColors.textDim, fontSize: 11)),
                  ],
                ],
              ),
            ),
            Text(right,
                textDirection: TextDirection.rtl,
                style: const TextStyle(
                    color: AppColors.accent,
                    fontSize: 18,
                    fontWeight: FontWeight.w700)),
            const SizedBox(width: 10),
            const Icon(Icons.open_in_new,
                color: AppColors.textDim, size: 16),
          ],
        ),
      ),
    );
  }
}

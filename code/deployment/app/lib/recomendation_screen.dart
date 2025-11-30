import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:music_rec_system/features/player/data/providers/song_providers.dart';
import 'package:music_rec_system/features/player/presentation/pages/player_page.dart';
import 'package:music_rec_system/widgets/neon_particles.dart';
import 'package:music_rec_system/widgets/wave_circle_shader.dart';

class RecommendationScreen extends ConsumerStatefulWidget {
  const RecommendationScreen({super.key});

  @override
  ConsumerState<RecommendationScreen> createState() => _RecommendationScreenState();
}

class _RecommendationScreenState extends ConsumerState<RecommendationScreen>
    with SingleTickerProviderStateMixin {
  final String _fullText = "Get music recommendations";
  String _visibleText = "";
  bool _isTyping = true;

  bool _showCursor = true;
  Timer? _typingTimer;
  Timer? _cursorTimer;

  late final AnimationController _buttonController;
  late final Animation<double> _buttonOpacity;

  bool _showSearching = false;

  final Color neonBlue = const Color(0xFF66CCFF);

  @override
  void initState() {
    super.initState();

    _buttonController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 400),
    );

    _buttonOpacity = CurvedAnimation(
      parent: _buttonController,
      curve: Curves.easeInOut,
    );

    _startTypingEffect();
    _startCursorBlink();
  }

  void _startCursorBlink() {
    _cursorTimer = Timer.periodic(const Duration(milliseconds: 500), (_) {
      if (!mounted) return;
      setState(() => _showCursor = !_showCursor);
    });
  }

  void _startTypingEffect() {
    int index = 0;

    _typingTimer = Timer.periodic(const Duration(milliseconds: 75), (timer) {
      if (!mounted) {
        timer.cancel();
        return;
      }

      if (index <= _fullText.length) {
        setState(() {
          _visibleText = _fullText.substring(0, index);
        });
        index++;
      } else {
        timer.cancel();
        setState(() => _isTyping = false);
        _buttonController.forward();
      }
    });
  }

  void _onGetRecommendations() {
    if (_showSearching) return;

    setState(() => _showSearching = true);

    // Вызываем startSession через провайдер
    ref.read(initialSongsProvider.future).then((_) {
      if (!mounted) return;
      
      // Переходим на PlayerPage после загрузки
      Navigator.of(context).pushReplacement(_fadeRoute());
    }).catchError((error) {
      if (!mounted) return;
      
      setState(() => _showSearching = false);
      
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Ошибка загрузки: $error'),
          backgroundColor: Colors.red,
        ),
      );
    });
  }

  PageRouteBuilder _fadeRoute() {
    return PageRouteBuilder(
      transitionDuration: const Duration(milliseconds: 700),
      reverseTransitionDuration: const Duration(milliseconds: 400),
      pageBuilder: (_, __, ___) => PlayerPage(),
      transitionsBuilder: (_, animation, __, child) {
        final curved = CurvedAnimation(parent: animation, curve: Curves.easeInOut);
        return FadeTransition(
          opacity: curved,
          child: child,
        );
      },
    );
  }

  @override
  void dispose() {
    _typingTimer?.cancel();
    _cursorTimer?.cancel();
    _buttonController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final displayText = _visibleText + (_showCursor ? "|" : " ");

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          /// --- НЕОНОВАЯ ВОЛНА ---
          
          
          WaveCircleShader(),
          NeonParticles(),

          /// --- ОСНОВНОЙ КОНТЕНТ ---
          Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  displayText,
                  style: TextStyle(
                    fontFamily: "VT323",
                    color: neonBlue,
                    fontSize: 32,
                    letterSpacing: 1.8,
                    shadows: [
                      Shadow(color: neonBlue.withOpacity(0.7), blurRadius: 12),
                    ],
                  ),
                ),

                const SizedBox(height: 40),

                FadeTransition(
                  opacity: _buttonOpacity,
                  child: ScaleTransition(
                    scale: _buttonOpacity,
                    child: Container(
                      decoration: BoxDecoration(
                        boxShadow: [
                          BoxShadow(
                            color: Colors.white.withOpacity(0.9),
                            blurRadius: 15,
                            spreadRadius: 1,
                          ),
                        ],
                      ),
                      child: ElevatedButton(
                        onPressed: _isTyping ? null : _onGetRecommendations,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.white,
                          foregroundColor: Colors.black,
                          padding: const EdgeInsets.symmetric(
                            vertical: 14,
                            horizontal: 28,
                          ),
                        ),
                        child: const Text(
                          "Get recommendations",
                          style: TextStyle(fontSize: 18, fontFamily: "VT323"),
                        ),
                      ),
                    ),
                  ),
                ),

                const SizedBox(height: 30),

                if (_showSearching)
                  Text(
                    "Searching for music...",
                    style: TextStyle(
                      fontFamily: "VT323",
                      color: Colors.white,
                      fontSize: 18,
                      shadows: [
                        Shadow(
                          color: neonBlue.withOpacity(0.5),
                          blurRadius: 10,
                        ),
                      ],
                    ),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}






import 'dart:math';

import 'package:flutter/material.dart';

class NeonParticles extends StatefulWidget {
  const NeonParticles({super.key});

  @override
  State<NeonParticles> createState() => _NeonParticlesState();
}

class _NeonParticlesState extends State<NeonParticles>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;
  final Random _rnd = Random();

  final int particleCount = 35;
  late List<_Particle> particles;

  @override
  void initState() {
    super.initState();

    particles = List.generate(particleCount, (_) => _Particle.random());

    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 55), // ещё более плавное цикличное движение
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (_, __) {
        return CustomPaint(
          painter: _ParticlesPainter(
            particles: particles,
            progress: _controller.value,
          ),
          size: Size.infinite,
        );
      },
    );
  }
}

/// ------------ ЛОГИКА ЧАСТИЦ ----------------

class _Particle {
  final double baseX;
  final double baseY;
  final double size;
  final double opacity;
  final Color color;
  final double phase;
  final int speedMultiplier;
  final double wobbleAmplitude;
  final int wobbleFreq;
  final int twinkleFreq;

  _Particle({
    required this.baseX,
    required this.baseY,
    required this.size,
    required this.opacity,
    required this.color,
    required this.phase,
    required this.speedMultiplier,
    required this.wobbleAmplitude,
    required this.wobbleFreq,
    required this.twinkleFreq,
  });

  factory _Particle.random() {
    final rnd = Random();
    final colors = [
      const Color(0xFFCFF5FF),
      const Color(0xFF9BE7FF),
      const Color(0xFF6FD9FF),
    ];

    return _Particle(
      baseX: rnd.nextDouble(),
      baseY: rnd.nextDouble(),
      size: rnd.nextDouble() * 2.0 + 0.8,
      opacity: rnd.nextDouble() * 0.65 + 0.35,
      color: colors[rnd.nextInt(colors.length)],
      phase: rnd.nextDouble(),
      speedMultiplier: 1, // один полный подъём за цикл
      wobbleAmplitude: rnd.nextDouble() * 0.02 + 0.005,
      wobbleFreq: rnd.nextInt(3) + 1,
      twinkleFreq: rnd.nextInt(3) + 1,
    );
  }
}

class _ParticlesPainter extends CustomPainter {
  final List<_Particle> particles;
  final double progress;
  final Color color = const Color(0xFF55CFFF);

  _ParticlesPainter({required this.particles, required this.progress});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint();
    const double twoPi = pi * 2;

    for (var p in particles) {
      double loop = progress + p.phase;

      // Цикличное вертикальное перемещение (летим вверх -> уменьшаем y)
      double travel = (loop * p.speedMultiplier) % 1.0;
      double dy = (p.baseY - travel + 1.0) % 1.0;

      // Небольшой горизонтальный "дрейф" синусом
      double wobbleAngle = twoPi * (loop * p.wobbleFreq);
      double dx = (p.baseX + sin(wobbleAngle) * p.wobbleAmplitude + 1.0) % 1.0;

      // Твинклинг
      double twinkle = 0.6 + 0.4 * sin(twoPi * (loop * p.twinkleFreq));
      double alpha = p.opacity * twinkle;

      final Offset center = Offset(dx * size.width, dy * size.height);

      paint
        ..color = p.color.withOpacity(alpha * 0.8)
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 7);
      canvas.drawCircle(center, p.size * 2.0, paint);

      paint
        ..maskFilter = null
        ..color = Colors.white.withOpacity(alpha);
      canvas.drawCircle(center, p.size * 0.7, paint);
    }
  }

  @override
  bool shouldRepaint(covariant _ParticlesPainter oldDelegate) => true;
}

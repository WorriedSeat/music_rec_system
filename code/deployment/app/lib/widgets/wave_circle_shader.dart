import 'dart:ui' as ui;
import 'package:flutter/material.dart';

class WaveCircleShader extends StatefulWidget {
  const WaveCircleShader({super.key});
  @override
  State<WaveCircleShader> createState() => _WaveCircleShaderState();
}

class _WaveCircleShaderState extends State<WaveCircleShader>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl;
  ui.FragmentShader? _shader;

  @override
  void initState() {
    super.initState();
    // ИСПРАВЛЕНО: добавлен duration
    _ctrl = AnimationController(vsync: this, duration: const Duration(seconds: 20))
      ..repeat();
    _loadShader();
  }

  Future<void> _loadShader() async {
    final program = await ui.FragmentProgram.fromAsset('assets/shaders/sound.frag');
    setState(() => _shader = program.fragmentShader());
  }

  @override
  Widget build(BuildContext context) {
    if (_shader == null) {
      return const Scaffold(body: ColoredBox(color: Colors.black));
    }

    return Scaffold(
      body: AnimatedBuilder(
        animation: _ctrl,
        builder: (_, __) {
          final size = MediaQuery.of(context).size;

          _shader!.setFloat(0, size.width);
          _shader!.setFloat(1, size.height);

          // Делаем заведомо бесшовный цикл:
          // _ctrl.value идёт от 0 до 1, умножаем его на 2π и шейдер
          // воспринимает iTime как фазу в радианах. Все sin(.. * iTime)
          // внутри шейдера используют ЦЕЛЫЕ множители, поэтому при
          // переходе с 2π обратно в 0 картинка совпадает.
          const double twoPi = 6.283185307179586;
          _shader!.setFloat(2, _ctrl.value * twoPi);

          return CustomPaint(
            size: Size.infinite,
            painter: _ShaderPainter(_shader!),
          );
        },
      ),
    );
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }
}

class _ShaderPainter extends CustomPainter {
  final ui.FragmentShader shader;
  const _ShaderPainter(this.shader);

  @override
  void paint(Canvas canvas, Size size) {
    // Сначала рисуем ЧИСТО ЧЁРНЫЙ фон
    canvas.drawRect(Offset.zero & size, Paint()..color = Colors.black);
    // Потом шейдер сверху
    canvas.drawRect(Offset.zero & size, Paint()..shader = shader);
  }

  @override
  bool shouldRepaint(_) => true;
}
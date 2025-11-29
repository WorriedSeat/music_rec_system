#include <flutter/runtime_effect.glsl>

uniform vec2 iResolution;
uniform float iTime;
out vec4 fragColor;

// Бирюзовый цвет, как на скриншотах
vec3 lineColor() {
    return vec3(0.0, 0.9, 0.85);
}

void main() {
    // Нормализованные координаты [-1;1] с поправкой на аспект
    vec2 uv = (FlutterFragCoord().xy / iResolution.xy) * 2.0 - 1.0;
    uv.x *= iResolution.x / iResolution.y;

    float angle = atan(uv.y, uv.x);
    float len   = length(uv);
    float t     = iTime;

    // Базовый радиус и амплитуда "жеванности" окружности
    float baseRadius = 0.6;
    float amp        = 0.10;

    // Толщина одной линии (уменьшили для более тонких полос)
    float thickness  = 0.004;

    // Сколько "следов" (несколько колец одно над другим)
    const int TRAILS = 8;

    vec3 col = vec3(0.0); // чёрный фон

    for (int i = 0; i < TRAILS; i++) {
        float fi = float(i);

        // Чем дальше след, тем он тусклее
        float trailAlpha = 1.0 - fi / float(TRAILS);
        trailAlpha *= 0.8;

        // Временной сдвиг для каждого следа (эффект "шлейфа")
        float tt = t - fi * 0.10;

        // Волна вдоль окружности: сумма нескольких гармоник.
        // Для ЧИСТОГО зацикливания t бежит от 0 до 2π, поэтому
        // множители при t выбираем ЦЕЛЫМИ, чтобы при t = 0 и t = 2π
        // аргументы синусов совпадали.
        float wave =
            sin(angle * 2.0 + tt * 2.0) * 0.45 +
            sin(angle * 4.0 - tt * 1.0) * 0.25 +
            sin(angle * 6.0 + tt * 3.0) * 0.18;

        float r = baseRadius + amp * wave;

        // Немного раздвинем следы по радиусу, чтобы линии не склеивались
        r += fi * 0.012;

        float dist = abs(len - r);

        // Узкая линия
        float line = 1.0 - smoothstep(0.0, thickness, dist);

        // Мягкое свечение вокруг линии (слегка сузили, чтобы не размывало тонкую линию)
        float glow = exp(-160.0 * dist);

        vec3 c = lineColor();
        vec3 contrib = c * (line * 1.4 + glow * 0.9) * trailAlpha;

        col += contrib;
    }

    // Лёгкая общая пульсация яркости (тоже с целым множителем по времени)
    col *= 1.0 + 0.10 * sin(t * 2.0);

    // Гамма-коррекция
    col = pow(col, vec3(0.9));

    fragColor = vec4(col, 1.0);
}
# Примеры API запросов и ответов

## 1. Start Session

### Запрос (POST /start_session)

**URL:** `http://your-api-url/start_session`

**Headers:**
```
Content-Type: application/json
```

**Body:**
```json
{
  "user_id": "user_123"
}
```

**Пример в Dart:**
```dart
final requestBody = {
  'user_id': 'user_123',
};

final response = await http.post(
  Uri.parse("$baseUrl/start_session"),
  headers: {
    'Content-Type': 'application/json',
  },
  body: jsonEncode(requestBody),
);
```

### Ответ (200 OK)

```json
{
  "session_id": "user_123",
  "songs": [
    {
      "track_id": 26,
      "color_hex": "#FF66FF"
    },
    {
      "track_id": 43,
      "color_hex": "#55CFFF"
    },
    {
      "track_id": 50,
      "color_hex": "#66FFAA"
    },
    {
      "track_id": 71,
      "color_hex": "#FFCC55"
    },
    {
      "track_id": 81,
      "color_hex": "#808080"
    }
  ],
  "message": "Session started. Listen and provide feedback!"
}
```

---

## 2. Fetch Recommendations

### Запрос (POST /recommend)

**URL:** `http://your-api-url/recommend`

**Headers:**
```
Content-Type: application/json
```

**Body:**
```json
{
  "session_id": "user_123",
  "songs": [
    {
      "track_id": 26,
      "played_ratio": 1.0,
      "action": 1
    },
    {
      "track_id": 43,
      "played_ratio": 0.85,
      "action": 1
    },
    {
      "track_id": 50,
      "played_ratio": 0.5,
      "action": 0
    },
    {
      "track_id": 71,
      "played_ratio": 0.2,
      "action": -1
    },
    {
      "track_id": 81,
      "played_ratio": 0.75,
      "action": 0
    }
  ]
}
```

**Пример в Dart:**
```dart
final requestBody = {
  'session_id': 'user_123',
  'songs': [
    CallbackEntity(trackId: 26, playedRatio: 1.0, action: 1).toJson(),
    CallbackEntity(trackId: 43, playedRatio: 0.85, action: 1).toJson(),
    CallbackEntity(trackId: 50, playedRatio: 0.5, action: 0).toJson(),
    CallbackEntity(trackId: 71, playedRatio: 0.2, action: -1).toJson(),
    CallbackEntity(trackId: 81, playedRatio: 0.75, action: 0).toJson(),
  ],
};

final response = await http.post(
  Uri.parse("$baseUrl/recommend"),
  headers: {
    'Content-Type': 'application/json',
  },
  body: jsonEncode(requestBody),
);
```

### Ответ (200 OK)

```json
{
  "songs": [
    {
      "track_id": 92,
      "color_hex": "#FF66FF"
    },
    {
      "track_id": 105,
      "color_hex": "#55CFFF"
    },
    {
      "track_id": 118,
      "color_hex": "#66FFAA"
    },
    {
      "track_id": 134,
      "color_hex": "#FFCC55"
    },
    {
      "track_id": 147,
      "color_hex": "#808080"
    }
  ]
}
```

---

## Поля и их значения

### SongFeedback (для /recommend)
- `track_id` (int) - ID трека
- `played_ratio` (float) - Доля прослушанного (0.0 - 1.0)
  - `0.0` - не прослушано
  - `1.0` - прослушано полностью
- `action` (int) - Действие пользователя:
  - `1` - лайк
  - `0` - нейтрально (не выбрано)
  - `-1` - дизлайк

### SongResponse (ответ)
- `track_id` (int) - ID трека
- `color_hex` (string) - Цвет в формате HEX (например, "#FF66FF")

---

## Пример полного цикла

1. **Начало сессии:**
   ```json
   POST /start_session
   {"user_id": "user_123"}
   ```
   → Получаем 5 начальных песен

2. **Пользователь прослушал песни:**
   - Песня 1: прослушано 100%, лайк (action: 1)
   - Песня 2: прослушано 85%, лайк (action: 1)
   - Песня 3: прослушано 50%, нейтрально (action: 0)
   - Песня 4: прослушано 20%, дизлайк (action: -1)
   - Песня 5: прослушано 75%, нейтрально (action: 0)

3. **Запрос новых рекомендаций:**
   ```json
   POST /recommend
   {
     "session_id": "user_123",
     "songs": [
       {"track_id": 26, "played_ratio": 1.0, "action": 1},
       {"track_id": 43, "played_ratio": 0.85, "action": 1},
       {"track_id": 50, "played_ratio": 0.5, "action": 0},
       {"track_id": 71, "played_ratio": 0.2, "action": -1},
       {"track_id": 81, "played_ratio": 0.75, "action": 0}
     ]
   }
   ```
   → Получаем 5 новых рекомендаций на основе истории


import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:music_rec_system/features/player/data/datasources/song_data_source.dart';
import 'package:music_rec_system/features/player/data/models/session_response.dart';
import 'package:music_rec_system/features/player/domain/callback_entity.dart';
import 'package:music_rec_system/features/player/domain/song_entity.dart';

// Провайдер для SongDataSource
final songDataSourceProvider = Provider<SongDataSource>((ref) {
  return SongDataSource();
});

// Провайдер для session_id
final sessionIdProvider = StateProvider<String?>((ref) => null);

// Провайдер для получения начальных песен (старт сессии)
final initialSongsProvider = FutureProvider<SessionResponse>((ref) async {
  final dataSource = ref.watch(songDataSourceProvider);
  final userId = 'user_${DateTime.now().millisecondsSinceEpoch}';
  final response = await dataSource.startSession(userId);
  // Сохраняем session_id
  ref.read(sessionIdProvider.notifier).state = response.sessionId;
  return response;
});

// Провайдер для получения рекомендаций на основе истории
final recommendationsProvider =
    FutureProvider.family<List<SongEntity>, RecommendationsParams>((ref, params) async {
  final dataSource = ref.watch(songDataSourceProvider);
  return await dataSource.callback(
    sessionId: params.sessionId,
    history: params.history,
  );
});

// Параметры для запроса рекомендаций
class RecommendationsParams {
  final String sessionId;
  final List<CallbackEntity> history;

  RecommendationsParams({
    required this.sessionId,
    required this.history,
  });

  @override
  bool operator == (Object other) =>
      identical(this, other) ||
      other is RecommendationsParams &&
          runtimeType == other.runtimeType &&
          sessionId == other.sessionId &&
          history.length == other.history.length;

  @override
  int get hashCode => sessionId.hashCode ^ history.length.hashCode;
}


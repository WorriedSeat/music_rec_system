import 'package:music_rec_system/features/player/domain/song_entity.dart';

class SessionResponse {
  final String sessionId;
  final List<SongEntity> songs;

  SessionResponse({
    required this.sessionId,
    required this.songs,
  });
}


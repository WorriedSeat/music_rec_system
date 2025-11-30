import 'dart:convert';

import 'package:http/http.dart' as http;
import 'package:music_rec_system/core/baseUrl/base_url.dart';
import 'package:music_rec_system/features/player/data/models/session_response.dart';
import 'package:music_rec_system/features/player/data/models/song_model.dart';
import 'package:music_rec_system/features/player/domain/callback_entity.dart';
import 'package:music_rec_system/features/player/domain/song_entity.dart';

class SongDataSource {
  Future<SessionResponse> startSession(String userId) async {
    final uri = Uri.parse("$baseUrl/start_session");
    
    final requestBody = {
      'user_id': userId,
    };
    
    final response = await http.post(
      uri,
      headers: {
        'Content-Type': 'application/json',
      },
      body: jsonEncode(requestBody),
    );
    
    if (response.statusCode == 200) {
      print(response.body);
      final data = jsonDecode(response.body);
      // API возвращает {"songs": [...], "session_id": "..."}
      final songsList = data['songs'] as List;
      final songs = songsList.map((json) => SongModel.fromJson(json)).toList();
      return SessionResponse(
        sessionId: data['session_id'] as String,
        songs: songs,
      );
    } else {
      throw Exception('Failed to start session: ${response.statusCode}');
    }
  }

  Future<List<SongEntity>> callback({
    required String sessionId,
    required List<CallbackEntity> history,
  }) async {
    final uri = Uri.parse("$baseUrl/recommend");
    
    final requestBody = {
      'session_id': sessionId,
      'songs': history.map((song) => song.toJson()).toList(),
    };
    
    final response = await http.post(
      uri,
      headers: {
        'Content-Type': 'application/json',
      },
      body: jsonEncode(requestBody),
    );
    
    if (response.statusCode == 200) {
      print(response.body);
      final data = jsonDecode(response.body);
      // API возвращает {"songs": [...]}
      final songsList = data['songs'] as List;
      return songsList.map((json) => SongModel.fromJson(json)).toList();
    } else {
      throw Exception('Failed send callback: ${response.statusCode}');
    }
  }

}

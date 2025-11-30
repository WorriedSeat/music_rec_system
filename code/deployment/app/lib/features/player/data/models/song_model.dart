import 'package:music_rec_system/features/player/domain/song_entity.dart';

class SongModel extends SongEntity {
  SongModel({required int trackId, required String colorHex, required int track_length})
    : super(colorHex: colorHex, trackId: trackId, track_length: track_length);

  factory SongModel.fromJson(Map<String, dynamic> json) {
    return SongModel(
      trackId: json['track_id'] ?? json['trackId'],
      colorHex: json['color_hex'] ?? json['colorHex'],
      track_length: json['track_length_seconds']
    );
  }
}

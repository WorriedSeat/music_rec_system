class CallbackEntity {
  final int trackId;
  final double playedRatio;
  final int action;

  CallbackEntity({
    required this.trackId,
    required this.action,
    required this.playedRatio,
  });

  Map<String, dynamic> toJson() {
    return {
      'track_id': trackId,
      'played_ratio': playedRatio,
      'action': action,
    };
  }
}

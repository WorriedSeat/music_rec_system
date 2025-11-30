import 'dart:async';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:music_rec_system/features/player/data/providers/song_providers.dart';
import 'package:music_rec_system/features/player/domain/callback_entity.dart';
import 'package:music_rec_system/features/player/domain/song_entity.dart';
import 'package:music_rec_system/widgets/neon_particles.dart';
import 'package:music_rec_system/widgets/wave_circle_shader.dart';

class Track {
  final String id;
  final double duration;
  final Color background;
  final Color primary;

  Track({
    required this.id,
    required this.duration,
    required this.background,
    required this.primary,
  });

  factory Track.fromJson(Map<String, dynamic> json) {
    return Track(
      id: json["id"],
      duration: double.parse(json["duration"]),
      background: _parseColor(json["backgroud_color"]),
      primary: _parseColor(json["primary_color"]),
    );
  }

  static Color _parseColor(String name) {
    switch (name) {
      case "red":
        return Color(0xFFFF66FF);
      case "blue":
        return Color(0xFF55CFFF);
      case "green":
        return Color(0xFF66FFAA);
      case "yellow":
        return Color(0xFFFFCC55);
      case "black":
        return Colors.black;
      case "white":
        return Colors.white;
    }
    return Colors.white;
  }
}

class PlayerPage extends ConsumerStatefulWidget {
  const PlayerPage({super.key});

  @override
  ConsumerState<PlayerPage> createState() => _MainPageState();
}

class _MainPageState extends ConsumerState<PlayerPage> {
  List<SongEntity>? songs;
  int currentIndex = 0;

  bool _isLoadingNewSongs = false;

  double _currentPos = 0;
  bool _isPlaying = true;
  int? _currentAction; // null, 1 (лайк), или -1 (дизлайк)

  // История прослушивания для текущей песни
  int? _currentTrackId;

  Timer? _playTimer;
  Timer? _visualizerTimer;
  final Random _random = Random();

  // История всех прослушанных песен
  final List<CallbackEntity> _playHistory = [];

  // Флаг чтобы не делать параллельные fetch-запросы
  bool _isFetching = false;

  Track get currentTrack {
    if (songs == null || songs!.isEmpty) {
      return Track(
        id: '0',
        duration: 0,
        background: Colors.black,
        primary: Colors.white,
      );
    }
    final song = songs![currentIndex];
    return Track(
      id: song.trackId.toString(),
      duration: song.track_length.toDouble(),
      background: Colors.black,
      primary: _hexToColor(song.colorHex),
    );
  }

  Color _hexToColor(String hex) {
    try {
      return Color(
        int.parse(hex.replaceFirst('#', ''), radix: 16) + 0xFF000000,
      );
    } catch (e) {
      return Colors.black;
    }
  }

  @override
  void initState() {
    super.initState();
    _loadSongs();
    _startVisualizer();
  }

  void _loadSongs() {
    ref
        .read(initialSongsProvider.future)
        .then((sessionResponse) {
          if (mounted) {
            setState(() {
              songs = sessionResponse.songs;
              if (sessionResponse.songs.isNotEmpty) {
                _currentTrackId = sessionResponse.songs[0].trackId;
                if (_isPlaying) {
                  _startPlaying();
                }
              }
            });
          }
        })
        .catchError((error) {
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(content: Text('Ошибка загрузки песен: $error')),
            );
          }
        });
  }

  /// Сохраняет текущую песню в историю (локально)
  void _saveCurrentTrack() {
    if (_currentTrackId == null || songs == null || songs!.isEmpty) return;

    final duration = currentTrack.duration;
    final playedRatio = duration > 0
        ? (_currentPos / duration).clamp(0.0, 1.0)
        : 0.0;
    final action = _currentAction ?? 0; // 0 если не выбрано

    _playHistory.add(
      CallbackEntity(
        trackId: _currentTrackId!,
        playedRatio: playedRatio,
        action: action,
      ),
    );
  }

  /// Получает новые рекомендации на основе истории
  Future<void> _fetchNewRecommendations() async {
    if (_isFetching) return;
    final sessionId = ref.read(sessionIdProvider);
    if (sessionId == null || _playHistory.isEmpty) return;

    _isFetching = true;

    // 🔥 Показываем индикатор
    if (mounted) setState(() => _isLoadingNewSongs = true);

    try {
      final dataSource = ref.read(songDataSourceProvider);
      final newSongs = await dataSource.callback(
        sessionId: sessionId,
        history: _playHistory,
      );

      if (mounted && newSongs.isNotEmpty) {
        setState(() {
          songs = newSongs;
          currentIndex = 0;
          _currentPos = 0;
          _currentAction = null;
          _currentTrackId = newSongs[0].trackId;
        });

        _playHistory.clear();

        if (_isPlaying) {
          _startPlaying();
        }
      } else {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Новые рекомендации не найдены')),
          );
        }
      }
    } catch (error) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Ошибка отправки ответа: $error')),
        );
      }
    } finally {
      _isFetching = false;

      // 🔥 Скрываем индикатор
      if (mounted) setState(() => _isLoadingNewSongs = false);
    }
  }

  @override
  void dispose() {
    _visualizerTimer?.cancel();
    _playTimer?.cancel();
    super.dispose();
  }

  void _startVisualizer() {
    _visualizerTimer?.cancel();
    _visualizerTimer = Timer.periodic(Duration(milliseconds: 10), (timer) {
      if (_isPlaying && mounted) setState(() {});
    });
  }

  void _startPlaying() {
    _playTimer?.cancel();
    _playTimer = Timer.periodic(Duration(seconds: 1), (timer) {
      // если не играем — таймер остановим
      if (!_isPlaying) {
        timer.cancel();
        return;
      }

      // если достигли конца трека — обрабатываем окончание
      if (_currentPos >= currentTrack.duration) {
        timer.cancel();
        _handleTrackEnd();
        return;
      }

      setState(() => _currentPos++);
    });
  }

  void _pausePlaying() => _playTimer?.cancel();

  /// Обработать окончание текущего трека (дослушан до конца)
  Future<void> _handleTrackEnd() async {
    if (songs == null || songs!.isEmpty) return;

    _saveCurrentTrack();

    final isLast = currentIndex == songs!.length - 1;

    if (isLast) {
      // не переключаемся автоматически на первый — запускаем подгрузку
      setState(() {
        _isPlaying = false; // остановим воспроизведение, пока подгружаем
      });

      await _fetchNewRecommendations();
      return;
    } else {
      // обычный переход на следующий трек
      _goToNextTrack();
    }
  }

  /// Внутренняя функция перехода на следующий трек (без сохранения истории)
  void _goToNextTrack() {
    if (songs == null || songs!.isEmpty) return;
    // безопасный инкремент (мы уже знаем, что не последний)
    if (currentIndex < songs!.length - 1) {
      setState(() {
        currentIndex++;
        _currentPos = 0;
        _currentAction = null;
        _currentTrackId = songs![currentIndex].trackId;
      });

      if (_isPlaying) {
        _startPlaying();
      }
    }
  }

  /// Нажатие кнопки Next
  Future<void> _nextTrack() async {
    if (songs == null || songs!.isEmpty) return;

    // Сохраняем текущую песню перед переходом
    _saveCurrentTrack();

    final wasLastTrack = currentIndex == songs!.length - 1;

    if (wasLastTrack) {
      // Если текущая — последняя, не переходим на первый. Подгружаем новые треки.
      setState(() {
        _isPlaying = false; // при подгрузке останавливаем проигрывание
      });
      await _fetchNewRecommendations();
      return;
    }

    // Иначе — просто переходим
    _goToNextTrack();
  }

  void _prevTrack() {
    if (songs == null || songs!.isEmpty) return;

    // Сохраняем текущую песню перед переходом
    _saveCurrentTrack();

    setState(() {
      currentIndex = (currentIndex - 1 + songs!.length) % songs!.length;
      _currentPos = 0;
      _currentAction = null; // Сбрасываем лайк/дизлайк
      _currentTrackId = songs![currentIndex].trackId;
    });

    if (_isPlaying) _startPlaying();
  }

  void _toggleLike() {
    setState(() {
      if (_currentAction == 1) {
        _currentAction = null;
      } else {
        _currentAction = 1;
      }
    });
  }

  void _toggleDislike() {
    setState(() {
      if (_currentAction == -1) {
        _currentAction = null;
      } else {
        _currentAction = -1;
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final songsAsync = ref.watch(initialSongsProvider);

    return songsAsync.when(
      data: (sessionResponse) {
        if (songs == null) {
          WidgetsBinding.instance.addPostFrameCallback((_) {
            if (mounted) {
              setState(() {
                songs = sessionResponse.songs;
                if (sessionResponse.songs.isNotEmpty) {
                  _currentTrackId = sessionResponse.songs[0].trackId;
                }
              });
            }
          });
        }
        return _buildPlayer(context);
      },
      loading: () => _buildLoading(context),
      error: (error, stack) => _buildError(context, error),
    );
  }

  Widget _buildLoading(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: const Center(child: CircularProgressIndicator(color: Colors.white)),
    );
  }

  Widget _buildError(BuildContext context, Object error) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.error_outline, color: Colors.red, size: 48),
            const SizedBox(height: 16),
            Text(
              'Ошибка загрузки',
              style: TextStyle(color: Colors.white, fontSize: 18),
            ),
            const SizedBox(height: 8),
            Text(
              error.toString(),
              style: TextStyle(color: Colors.white70, fontSize: 14),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 24),
            ElevatedButton(
              onPressed: () {
                ref.invalidate(initialSongsProvider);
                _loadSongs();
              },
              child: const Text('Повторить'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPlayer(BuildContext context) {
    if (songs == null || songs!.isEmpty) {
      return _buildLoading(context);
    }

    return Scaffold(
      backgroundColor: currentTrack.background,
      body: Stack(
        children: [
          if (_isLoadingNewSongs)
            Positioned.fill(
              child: Container(
                color: Colors.black.withOpacity(0.4),
                child: const Center(
                  child: CircularProgressIndicator(color: Colors.white),
                ),
              ),
            ),
          WaveCircleShader(),
          NeonParticles(),
          Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const SizedBox(height: 100),

              /// --- ВИЗУАЛИЗАТОР ---
              SizedBox(
                height: 80,
                child: Center(
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: List.generate(7, (index) {
                      return AnimatedContainer(
                        duration: Duration(milliseconds: 80),
                        width: 8,
                        height: _isPlaying ? _random.nextInt(80) + 20 : 20,
                        margin: EdgeInsets.symmetric(horizontal: 4),
                        decoration: BoxDecoration(
                          boxShadow: [
                            BoxShadow(
                              color: currentTrack.primary,
                              blurRadius: 15,
                            ),
                          ],
                          color: currentTrack.primary,
                          borderRadius: BorderRadius.circular(4),
                        ),
                      );
                    }),
                  ),
                ),
              ),

              /// --- ВРЕМЯ ---
              Padding(
                padding: const EdgeInsets.fromLTRB(40, 100, 40, 0),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      _formatTime(_currentPos),
                      style: TextStyle(color: Colors.white70, fontSize: 14),
                    ),
                    Text(
                      _formatTime(currentTrack.duration),
                      style: TextStyle(color: Colors.white70, fontSize: 14),
                    ),
                  ],
                ),
              ),

              /// --- СЛАЙДЕР ---
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 30.0),
                child: Slider(
                  value: _currentPos,
                  min: 0,
                  max: currentTrack.duration,
                  onChanged: (value) => setState(() => _currentPos = value),
                  activeColor: Colors.white,
                  inactiveColor: Colors.white24,
                  thumbColor: Colors.white,
                ),
              ),

              SizedBox(height: 30),

              /// --- КНОПКИ ПРОИГРЫВАТЕЛЯ ---
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  IconButton(
                    onPressed: _toggleDislike,
                    icon: Icon(
                      _currentAction == -1
                          ? Icons.thumb_down
                          : Icons.thumb_down_alt_outlined,
                      size: 35,
                    ),
                    color: Colors.white,
                  ),

                  SizedBox(width: 25),

                  /// PREV
                  IconButton(
                    icon: Icon(
                      Icons.skip_previous,
                      size: 50,
                      color: Colors.white,
                    ),
                    onPressed: _prevTrack,
                  ),

                  SizedBox(width: 20),

                  /// PLAY / PAUSE
                  Container(
                    decoration: BoxDecoration(
                      boxShadow: [
                        BoxShadow(color: currentTrack.primary, blurRadius: 15),
                      ],
                      color: currentTrack.primary,
                      shape: BoxShape.circle,
                    ),
                    child: IconButton(
                      icon: Icon(
                        _isPlaying ? Icons.pause : Icons.play_arrow,
                        size: 40,
                        color: Colors.white,
                      ),
                      onPressed: () {
                        setState(() => _isPlaying = !_isPlaying);

                        if (_isPlaying) {
                          _startPlaying();
                        } else {
                          _pausePlaying();
                        }
                      },
                    ),
                  ),

                  SizedBox(width: 20),

                  /// NEXT
                  IconButton(
                    icon: Icon(Icons.skip_next, size: 50, color: Colors.white),
                    onPressed: () {
                      // кнопка может вызывать async метод
                      _nextTrack();
                    },
                  ),

                  SizedBox(width: 25),

                  IconButton(
                    onPressed: _toggleLike,
                    icon: Icon(
                      _currentAction == 1
                          ? Icons.favorite
                          : Icons.favorite_outline,
                      size: 35,
                    ),
                    color: Colors.white,
                  ),
                ],
              ),
            ],
          ),
        ],
      ),
    );
  }

  /// Формат времени
  String _formatTime(double seconds) {
    int minutes = (seconds / 60).floor();
    int remainingSeconds = (seconds % 60).round();
    return '${minutes.toString().padLeft(2, '0')}:${remainingSeconds.toString().padLeft(2, '0')}';
  }
}

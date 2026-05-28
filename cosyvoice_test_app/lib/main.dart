import 'dart:io';
import 'package:audioplayers/audioplayers.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'pipeline/constants.dart';
import 'pipeline/cosyvoice_pipeline.dart';

void main() {
  runApp(const CosyVoiceApp());
}

class CosyVoiceApp extends StatelessWidget {
  const CosyVoiceApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'CosyVoice TTS',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF1A73E8),
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      home: const CosyVoiceHomePage(),
    );
  }
}

class CosyVoiceHomePage extends StatefulWidget {
  const CosyVoiceHomePage({super.key});

  @override
  State<CosyVoiceHomePage> createState() => _CosyVoiceHomePageState();
}

class _CosyVoiceHomePageState extends State<CosyVoiceHomePage> {
  final CosyVoicePipeline _pipeline = CosyVoicePipeline();
  final TextEditingController _textController =
      TextEditingController(text: defaultTtsText);
  final AudioPlayer _audioPlayer = AudioPlayer();

  String _modelDir = '';
  String _refWavPath = '';
  String _promptText = '';
  String _status = 'Ready';
  String _outputPath = '';
  Map<String, double> _timings = {};
  double _rtf = 0.0;
  double _audioDuration = 0.0;
  bool _isGenerating = false;
  bool _isPlaying = false;
  Map<String, bool> _modelStatus = {};

  @override
  void dispose() {
    _textController.dispose();
    _audioPlayer.dispose();
    _pipeline.dispose();
    super.dispose();
  }

  Future<void> _pickModelDir() async {
    final result = await FilePicker.getDirectoryPath(
      dialogTitle: 'Select CosyVoice Model Directory',
    );
    if (result != null) {
      setState(() {
        _modelDir = result;
        _status = 'Model dir: $result';
        _modelStatus = _pipeline.checkModels(result);
      });
    }
  }

  Future<void> _loadModelDir(String path) async {
    setState(() {
      _status = 'Loading models...';
      _modelDir = path;
    });
    try {
      await _pipeline.init(path);
      setState(() {
        _status = 'Models loaded successfully';
        _modelStatus = _pipeline.checkModels(path);
      });
    } catch (e) {
      setState(() {
        _status = 'Error loading models: $e';
      });
    }
  }

  Future<void> _loadModels() async {
    if (_modelDir.isEmpty) {
      // Try default Android path
      if (await Directory(defaultModelDir).exists()) {
        await _loadModelDir(defaultModelDir);
      } else {
        setState(() {
          _status = 'Default dir not found. Please select model directory.';
        });
        await _pickModelDir();
        if (_modelDir.isNotEmpty) {
          await _loadModelDir(_modelDir);
        }
      }
    } else {
      await _loadModelDir(_modelDir);
    }
  }

  Future<void> _pickRefWav() async {
    final result = await FilePicker.pickFiles(
      type: FileType.audio,
      dialogTitle: 'Select Reference Audio (WAV)',
      allowMultiple: false,
    );
    if (result != null && result.files.single.path != null) {
      setState(() {
        _refWavPath = result.files.single.path!;
        _status = 'Reference: ${result.files.single.name}';
      });
    }
  }

  Future<void> _generate() async {
    if (!_pipeline.isInitialized) {
      setState(() => _status = 'Load models first!');
      return;
    }
    if (_refWavPath.isEmpty) {
      setState(() => _status = 'Select reference audio first!');
      return;
    }
    if (_textController.text.isEmpty) {
      setState(() => _status = 'Enter text to synthesize!');
      return;
    }

    setState(() {
      _isGenerating = true;
      _status = 'Generating...';
      _timings = {};
      _rtf = 0.0;
      _audioDuration = 0.0;
      _outputPath = '';
    });

    try {
      final result = await _pipeline.generate(
        _textController.text,
        _refWavPath,
        _promptText,
      );

      setState(() {
        _outputPath = result.outputPath;
        _audioDuration = result.audioDuration;
        _timings = result.timings;
        _rtf = result.rtf;
        _status = 'Done! ${result.audioDuration.toStringAsFixed(2)}s audio, ${result.speechTokenCount} tokens';
        _isGenerating = false;
      });
    } catch (e, stackTrace) {
      setState(() {
        _status = 'Error: $e';
        _isGenerating = false;
      });
      debugPrint('Generation error: $e\n$stackTrace');
    }
  }

  Future<void> _playAudio() async {
    if (_outputPath.isEmpty) return;
    await _audioPlayer.stop();
    setState(() => _isPlaying = true);
    try {
      await _audioPlayer.play(DeviceFileSource(_outputPath));
    } catch (e) {
      setState(() => _isPlaying = false);
      debugPrint('Playback error: $e');
      return;
    }
    _audioPlayer.onPlayerComplete.listen((_) {
      if (mounted) setState(() => _isPlaying = false);
    });
  }

  Future<void> _stopAudio() async {
    await _audioPlayer.stop();
    setState(() => _isPlaying = false);
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(
        title: const Text('CosyVoice3 ONNX TTS'),
        backgroundColor: theme.colorScheme.primaryContainer,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // ── Model Directory ──
            _sectionTitle('Model Directory'),
            Row(
              children: [
                Expanded(
                  child: Text(
                    _modelDir.isEmpty
                        ? 'No directory selected'
                        : _modelDir,
                    style: theme.textTheme.bodySmall,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                const SizedBox(width: 8),
                OutlinedButton(
                  onPressed: _pickModelDir,
                  child: const Text('Browse'),
                ),
                const SizedBox(width: 8),
                FilledButton(
                  onPressed: _isGenerating ? null : _loadModels,
                  child: const Text('Load Models'),
                ),
              ],
            ),
            if (_modelStatus.isNotEmpty) ...[
              const SizedBox(height: 8),
              _modelStatusWidget(),
            ],
            const SizedBox(height: 16),

            // ── Reference Audio ──
            _sectionTitle('Reference Audio'),
            Row(
              children: [
                Expanded(
                  child: Text(
                    _refWavPath.isEmpty
                        ? 'No reference audio selected'
                        : _refWavPath
                            .split(Platform.pathSeparator)
                            .last,
                    style: theme.textTheme.bodySmall,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                const SizedBox(width: 8),
                OutlinedButton(
                  onPressed: _pickRefWav,
                  child: const Text('Browse'),
                ),
              ],
            ),
            const SizedBox(height: 8),
            TextField(
              decoration: const InputDecoration(
                labelText: 'Prompt Text (optional, auto-detected from filename)',
                border: OutlineInputBorder(),
                isDense: true,
              ),
              onChanged: (v) => _promptText = v,
            ),
            const SizedBox(height: 16),

            // ── TTS Text ──
            _sectionTitle('Text to Synthesize'),
            TextField(
              controller: _textController,
              maxLines: 3,
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
                hintText: 'Enter text...',
              ),
            ),
            const SizedBox(height: 16),

            // ── Generate Button ──
            FilledButton.icon(
              onPressed: _isGenerating ? null : _generate,
              icon: _isGenerating
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.play_arrow),
              label: Text(_isGenerating ? 'Generating...' : 'Generate'),
            ),
            const SizedBox(height: 16),

            // ── Status ──
            _sectionTitle('Status'),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: theme.colorScheme.surfaceContainerHighest,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                _status,
                style: theme.textTheme.bodyMedium,
              ),
            ),
            const SizedBox(height: 16),

            // ── Playback ──
            if (_outputPath.isNotEmpty) ...[
              _sectionTitle('Playback'),
              Row(
                children: [
                  FilledButton.icon(
                    onPressed: _isPlaying ? _stopAudio : _playAudio,
                    icon: Icon(_isPlaying ? Icons.stop : Icons.play_circle),
                    label: Text(_isPlaying ? 'Stop' : 'Play'),
                  ),
                  const SizedBox(width: 12),
                  Text(
                    'Duration: ${_audioDuration.toStringAsFixed(2)}s',
                    style: theme.textTheme.bodyMedium,
                  ),
                ],
              ),
              const SizedBox(height: 16),
            ],

            // ── Timing ──
            if (_timings.isNotEmpty) ...[
              _sectionTitle('Timing & RTF'),
              _timingTable(),
            ],
          ],
        ),
      ),
    );
  }

  Widget _sectionTitle(String title) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Text(
        title,
        style: Theme.of(context).textTheme.titleSmall?.copyWith(
              fontWeight: FontWeight.bold,
            ),
      ),
    );
  }

  Widget _modelStatusWidget() {
    final theme = Theme.of(context);
    return Wrap(
      spacing: 6,
      runSpacing: 4,
      children: _modelStatus.entries.map((e) {
        final loaded = e.value;
        return Chip(
          label: Text(
            e.key,
            style: theme.textTheme.labelSmall,
          ),
          avatar: Icon(
            loaded ? Icons.check_circle : Icons.cancel,
            size: 14,
            color: loaded ? Colors.green : Colors.red,
          ),
          visualDensity: VisualDensity.compact,
        );
      }).toList(),
    );
  }

  Widget _timingTable() {
    final theme = Theme.of(context);
    return Table(
      columnWidths: const {
        0: FlexColumnWidth(2),
        1: FlexColumnWidth(1.5),
        2: FlexColumnWidth(1.5),
      },
      children: [
        TableRow(
          decoration: BoxDecoration(
            color: theme.colorScheme.surfaceContainerHighest,
          ),
          children: [
            _tableCell('Stage', bold: true),
            _tableCell('Time (s)', bold: true),
            _tableCell('RTF', bold: true),
          ],
        ),
        for (final entry in _timings.entries)
          TableRow(children: [
            _tableCell(entry.key),
            _tableCell(entry.value.toStringAsFixed(2)),
            _tableCell(
              _audioDuration > 0
                  ? (entry.value / _audioDuration).toStringAsFixed(3)
                  : '-',
            ),
          ]),
        TableRow(
          decoration: BoxDecoration(
            color: theme.colorScheme.surfaceContainerHighest,
          ),
          children: [
            _tableCell('TOTAL', bold: true),
            _tableCell(
                ((_timings['llm'] ?? 0) +
                            (_timings['flow'] ?? 0) +
                            (_timings['hift'] ?? 0))
                        .toStringAsFixed(2),
                bold: true),
            _tableCell(_rtf.toStringAsFixed(3), bold: true),
          ],
        ),
      ],
    );
  }

  Widget _tableCell(String text, {bool bold = false}) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
      child: Text(
        text,
        style: TextStyle(
          fontWeight: bold ? FontWeight.bold : FontWeight.normal,
          fontSize: 13,
        ),
      ),
    );
  }
}

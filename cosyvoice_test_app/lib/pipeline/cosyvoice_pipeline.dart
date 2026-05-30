import 'dart:io';
import 'dart:typed_data';
import 'audio_utils.dart';
import 'constants.dart';
import 'flow_inference.dart';
import 'hift_inference.dart';
import 'llm_inference.dart';
import 'pipeline_logger.dart';
import 'preprocessing.dart';

/// Main pipeline orchestrator tying all stages together.
class CosyVoicePipeline {
  final Preprocessor _preprocessor = Preprocessor();
  final LlmInference _llm = LlmInference();
  final FlowInference _flow = FlowInference();
  final HiftInference _hift = HiftInference();

  bool _isInitialized = false;
  String _onnxDir = '';
  String _modelDir = '';
  String _tokenizerDir = '';

  bool get isInitialized => _isInitialized;

  /// Initialize pipeline by loading all ONNX models.
  /// [modelDir] — root directory containing onnx_models/ and pretrained_models/
  Future<void> init(String modelDir) async {
    _onnxDir = '$modelDir/onnx_models';
    _modelDir =
        '$modelDir/pretrained_models/Fun-CosyVoice3-0.5B';
    _tokenizerDir =
        '$modelDir/pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN';

    // Verify directories exist
    if (!await Directory(_onnxDir).exists()) {
      throw Exception('ONNX models directory not found: $_onnxDir');
    }
    if (!await Directory(_modelDir).exists()) {
      throw Exception('Model directory not found: $_modelDir');
    }
    if (!await Directory(_tokenizerDir).exists()) {
      throw Exception('Tokenizer directory not found: $_tokenizerDir');
    }

    // Load preprocessing models
    await _preprocessor.load(_onnxDir, _modelDir, _tokenizerDir);

    // Load LLM models
    await _llm.load(_onnxDir);

    // Load flow models
    await _flow.load(_onnxDir);

    // Load HiFT vocoder
    await _hift.load(_onnxDir);

    _isInitialized = true;

    // Verify tokenizer produces correct token IDs
    pLog('═══ TOKENIZER VERIFICATION ═══');
    final test1 = _preprocessor.tokenizerEncode('You are a helpful assistant.');
    pLog('  "You are a helpful assistant." → $test1');
    pLog('  Expected: [2610, 525, 264, 10950, 17847, 13]');
    final test2 = _preprocessor.tokenizerEncode('안녕하세요, 반갑습니다.');
    pLog('  "안녕하세요, 반갑습니다." → $test2');
    pLog('  Expected: [126246, 144370, 91145, 11, 63757, 138685, 38231, 13]');
    final test3 = _preprocessor.tokenizerEncode('You are a helpful assistant.<|endofprompt|>안녕하세요 저는 오늘 이렇게 만나서 정말 반갑습니다.');
    pLog('  full prompt → $test3');
    pLog('  Expected: [2610, 525, 264, 10950, 17847, 13, 151646, 126246, 144370, 91145, 134561, 133857, 130653, 142353, 26698, 134247, 63757, 138685, 38231, 13]');
  }

  /// Check which model files exist in the model directory.
  Map<String, bool> checkModels(String modelDir) {
    final onnxDir = '$modelDir/onnx_models';
    final modelDirPretrained =
        '$modelDir/pretrained_models/Fun-CosyVoice3-0.5B';
    final tokenizerDir =
        '$modelDir/pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN';

    final models = <String, bool>{
      'mel_16k_128bin.onnx': File('$onnxDir/mel_16k_128bin.onnx').existsSync(),
      'mel_24k_80bin.onnx': File('$onnxDir/mel_24k_80bin.onnx').existsSync(),
      'fbank_16k_80bin.onnx':
          File('$onnxDir/fbank_16k_80bin.onnx').existsSync(),
      'speech_tokenizer_v3.onnx':
          File('$modelDirPretrained/speech_tokenizer_v3.onnx').existsSync(),
      'campplus.onnx':
          File('$modelDirPretrained/campplus.onnx').existsSync(),
      'llm_embed.onnx': File('$onnxDir/llm_embed_int4_gather.onnx').existsSync() ||
          File('$onnxDir/llm_embed.onnx').existsSync(),
      'llm_initial.onnx': File('$onnxDir/llm_initial.onnx').existsSync() ||
          File('$onnxDir/llm_initial_int8.onnx').existsSync(),
      'llm_decode_int8.onnx':
          File('$onnxDir/llm_decode_int8.onnx').existsSync() ||
              File('$onnxDir/llm_decode.onnx').existsSync(),
      'flow_prep_mobile.onnx':
          File('$onnxDir/flow_prep_mobile.onnx').existsSync() ||
              File('$onnxDir/flow_prep.onnx').existsSync(),
      'dit_estimator_int8_ffn.onnx':
          File('$onnxDir/dit_estimator_int8_ffn.onnx').existsSync() ||
              File('$onnxDir/dit_estimator_mobile.onnx').existsSync() ||
              File('$onnxDir/dit_estimator.onnx').existsSync(),
      'hift.onnx': File('$onnxDir/hift.onnx').existsSync(),
      'tokenizer.json':
          File('$tokenizerDir/tokenizer.json').existsSync(),
    };
    return models;
  }

  /// Run full TTS pipeline.
  ///
  /// [text] — text to synthesize
  /// [refWavPath] — path to reference WAV file
  /// [promptText] — prompt text for the reference audio (optional, auto-detected)
  ///
  /// Returns PipelineResult with audio path and timing info.
  Future<PipelineResult> generate(
    String text,
    String refWavPath,
    String promptText,
  ) async {
    if (!_isInitialized) throw Exception('Pipeline not initialized');

    final timings = <String, double>{};
    final sw = Stopwatch();

    // ── Resolve prompt text ──
    var resolvedPrompt = promptText;
    if (resolvedPrompt.isEmpty) {
      // Auto-detect from reference filename
      final basename = File(refWavPath).path
          .split(Platform.pathSeparator)
          .last
          .replaceAll('.wav', '');
      final detected = refPromptMap[basename];
      if (detected != null) {
        resolvedPrompt =
            'You are a helpful assistant.<|endofprompt|>$detected';
      } else {
        resolvedPrompt =
            'You are a helpful assistant.<|endofprompt|>$defaultTtsText';
      }
    } else if (!resolvedPrompt.contains('<|endofprompt|>')) {
      resolvedPrompt =
          'You are a helpful assistant.<|endofprompt|>$resolvedPrompt';
    }

    // ── Stage 1: Preprocessing ──
    pLog('═══ STAGE 1: PREPROCESSING ═══');
    pLog('Prompt text: $resolvedPrompt');
    pLog('TTS text: $text');
    sw.reset();
    sw.start();
    final preprocData =
        await _preprocessor.run(refWavPath, resolvedPrompt, text);
    sw.stop();
    timings['preprocessing'] = sw.elapsedMilliseconds / 1000.0;
    pLog('  prompt_text_tokens (${preprocData.promptTextTokens.length}): ${shortList(preprocData.promptTextTokens)}');
    pLog('  tts_text_tokens (${preprocData.ttsTextTokens.length}): ${shortList(preprocData.ttsTextTokens)}');
    pLog('  speech_tokens (${preprocData.speechTokens.length}): ${shortList(preprocData.speechTokens)}');
    pLog('  speaker_embedding: ${preprocData.speakerEmbedding.length} values');
    pLog('  prompt_speech_feat: ${preprocData.promptSpeechFeat.length} values, shape=${preprocData.promptFeatShape}, featLen=${preprocData.promptFeatLen}');

    // Release preprocessing sessions to free ~970MB (data already extracted into preprocData)
    pLog('  Releasing preprocessing sessions...');
    await _preprocessor.dispose();

    // ── Stage 2: LLM Inference ──
    pLog('═══ STAGE 2: LLM INFERENCE ═══');
    sw.reset();
    sw.start();
    final speechTokens = await _llm.run(preprocData);
    sw.stop();
    timings['llm'] = sw.elapsedMilliseconds / 1000.0;
    pLog('  Generated speech tokens (${speechTokens.length}): ${shortList(speechTokens)}');

    if (speechTokens.isEmpty) {
      throw Exception('LLM produced no speech tokens');
    }

    // Release LLM sessions to free memory for Flow (~803MB freed)
    pLog('  Releasing LLM sessions...');
    await _llm.dispose();

    // ── Stage 3: Flow/DiT Inference ──
    pLog('═══ STAGE 3: FLOW/DIT ═══');
    pLog('  prompt_tokens: ${preprocData.speechTokens.length}, generated: ${speechTokens.length}');
    sw.reset();
    sw.start();
    final melOutput = await _flow.run(
      speechTokens: speechTokens,
      promptTokens: preprocData.speechTokens,
      promptSpeechFeat: preprocData.promptSpeechFeat,
      promptFeatShape: preprocData.promptFeatShape,
      speakerEmbedding: preprocData.speakerEmbedding,
    );
    sw.stop();
    timings['flow'] = sw.elapsedMilliseconds / 1000.0;

    // Release Flow sessions to free memory for HiFT
    pLog('  Releasing Flow sessions...');
    await _flow.dispose();

    // Determine mel length (output is (1, 80, T_mel) flat)
    final melLen = melOutput.length ~/ melDim;
    pLog('  mel_output: ${melOutput.length} values, melLen=$melLen frames');

    // ── Stage 4: HiFT Vocoder ──
    pLog('═══ STAGE 4: HIFT VOCODER ═══');
    sw.reset();
    sw.start();
    final audio = await _hift.run(melOutput, melLen);
    sw.stop();
    timings['hift'] = sw.elapsedMilliseconds / 1000.0;
    pLog('  audio: ${audio.length} samples, duration=${(audio.length / sampleRate).toStringAsFixed(3)}s');

    // ── Stage 5: Save Output ──
    // Normalize audio: scale to [-0.95, 0.95] range
    double maxAbs = 0.0;
    for (int i = 0; i < audio.length; i++) {
      final abs = audio[i].abs();
      if (abs > maxAbs) maxAbs = abs;
    }
    pLog('  max_amplitude: $maxAbs');
    final normalizedAudio = Float32List(audio.length);
    if (maxAbs > 1e-6) {
      final scale = 0.95 / maxAbs;
      for (int i = 0; i < audio.length; i++) {
        normalizedAudio[i] = audio[i] * scale;
      }
    }

    final outputDir = await getOutputDir();
    final outputPath = '$outputDir/cosyvoice_output.wav';
    await writeWav(outputPath, normalizedAudio, sampleRate);

    final audioDuration = audio.length / sampleRate;
    final totalInference =
        (timings['llm'] ?? 0) + (timings['flow'] ?? 0) + (timings['hift'] ?? 0);
    final totalRtf = audioDuration > 0 ? totalInference / audioDuration : 0.0;
    final llmRtf = audioDuration > 0
        ? (timings['llm'] ?? 0) / audioDuration
        : 0.0;
    final flowRtf = audioDuration > 0
        ? (timings['flow'] ?? 0) / audioDuration
        : 0.0;
    pLog('═══ RESULT: ${audioDuration.toStringAsFixed(3)}s audio, ${speechTokens.length} speech tokens, total mel=$melLen ═══');
    pLog('  RTF total=${totalRtf.toStringAsFixed(2)} | LLM=${llmRtf.toStringAsFixed(2)} (${(timings['llm'] ?? 0).toStringAsFixed(2)}s) | Flow=${flowRtf.toStringAsFixed(2)} (${(timings['flow'] ?? 0).toStringAsFixed(2)}s) | HiFT=${(timings['hift'] ?? 0).toStringAsFixed(2)}s | Preproc=${(timings['preprocessing'] ?? 0).toStringAsFixed(2)}s');

    return PipelineResult(
      outputPath: outputPath,
      audioDuration: audioDuration,
      timings: timings,
      rtf: totalRtf,
      speechTokenCount: speechTokens.length,
    );
  }

  /// Release all model sessions.
  Future<void> dispose() async {
    await _preprocessor.dispose();
    await _llm.dispose();
    await _flow.dispose();
    await _hift.dispose();
    _isInitialized = false;
  }
}

/// Result of pipeline execution.
class PipelineResult {
  final String outputPath;
  final double audioDuration;
  final Map<String, double> timings;
  final double rtf;
  final int speechTokenCount;

  PipelineResult({
    required this.outputPath,
    required this.audioDuration,
    required this.timings,
    required this.rtf,
    required this.speechTokenCount,
  });
}

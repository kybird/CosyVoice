import 'dart:io';
import 'dart:typed_data';
import 'package:onnxruntime_v2/onnxruntime_v2.dart';
import 'audio_utils.dart';
import 'bpe_tokenizer.dart';
import 'constants.dart';
import 'pipeline_logger.dart';
import 'tensor_utils.dart';

/// Preprocessing stage: feature extraction, speech tokenization,
/// speaker embedding, and text tokenization.
class Preprocessor {
  late OrtSession _mel16kSession;
  late OrtSession _mel24kSession;
  late OrtSession _fbankSession;
  late OrtSession _speechTokenizerSession;
  late OrtSession _campplusSession;
  late BpeTokenizer _tokenizer;

  bool _isLoaded = false;

  bool get isLoaded => _isLoaded;

  /// Load all preprocessing ONNX models.
  /// [onnxDir] — onnx_models/ directory (mel, fbank)
  /// [modelDir] — pretrained_models/Fun-CosyVoice3-0.5B/ (speech_tokenizer, campplus)
  /// [tokenizerPath] — CosyVoice-BlankEN/ directory (tokenizer.json)
  Future<void> load(String onnxDir, String modelDir, String tokenizerPath) async {
    final opts = OrtSessionOptions()
      ..setSessionGraphOptimizationLevel(GraphOptimizationLevel.ortEnableAll)
      ..setIntraOpNumThreads(1)
      ..setInterOpNumThreads(1);

    // Mel 16k (128-bin whisper mel for speech tokenizer)
    _mel16kSession = await _loadSession('$onnxDir/mel_16k_128bin.onnx', opts);

    // Mel 24k (80-bin matcha mel for flow prompt)
    _mel24kSession = await _loadSession('$onnxDir/mel_24k_80bin.onnx', opts);

    // Fbank 16k (80-bin kaldi fbank for campplus)
    _fbankSession = await _loadSession('$onnxDir/fbank_16k_80bin.onnx', opts);

    // Speech tokenizer — lives in pretrained_models/, not onnx_models/
    _speechTokenizerSession =
        await _loadSession('$modelDir/speech_tokenizer_v3.onnx', opts);

    // Campplus (speaker embedding) — lives in pretrained_models/, not onnx_models/
    _campplusSession = await _loadSession('$modelDir/campplus.onnx', opts);

    // BPE tokenizer
    _tokenizer = BpeTokenizer();
    await _tokenizer.load('$tokenizerPath/tokenizer.json');

    _isLoaded = true;
  }

  Future<OrtSession> _loadSession(String path, OrtSessionOptions opts) async {
    final file = File(path);
    if (!await file.exists()) {
      throw Exception('Model not found: $path');
    }
    // Use fromBuffer for cross-platform compatibility (Windows path issues)
    if (Platform.isWindows) {
      final bytes = await file.readAsBytes();
      return OrtSession.fromBuffer(bytes, opts);
    }
    return OrtSession.fromFile(file, opts);
  }

  /// Extract speech tokens from reference audio.
  /// Returns list of token IDs.
  Future<List<int>> extractSpeechTokens(String wavPath) async {
    final speech = await loadWav(wavPath, targetSr: 16000);
    // ONNX mel extraction (whisper 128-bin)
    final waveform = Float32List.fromList(speech);

    final runOpts = OrtRunOptions();
    final inputs = {
      'waveform': OrtValueTensor.createTensorWithDataList(
          waveform, [1, waveform.length]),
    };

    final melOutputs = _mel16kSession.run(runOpts, inputs);
    final melFeat = flattenToFloat32(melOutputs[0]!.value);
    // melFeat shape: (1, 128, T) flattened
    // Need to figure out T from the output
    final outputShape = _getOutputShape(melOutputs[0]!.value);
    final melFrames = outputShape.length >= 3 ? outputShape[2] : (melFeat.length ~/ 128);

    // Speech tokenizer
    final spInputs = {
      _speechTokenizerSession.inputNames[0]: OrtValueTensor.createTensorWithDataList(
          melFeat, outputShape.length >= 3 ? [1, 128, melFrames] : [1, 128]),
      _speechTokenizerSession.inputNames[1]:
          OrtValueTensor.createTensorWithDataList(Int32List.fromList([melFrames]), [1]),
    };
    final spOutputs = _speechTokenizerSession.run(runOpts, spInputs);
    final tokensFlat = flattenToInt32(spOutputs[0]!.value);
    return tokensFlat.toList();
  }

  /// Extract speaker embedding from reference audio.
  /// Returns Float32List of shape (192,).
  Future<Float32List> extractSpeakerEmbedding(String wavPath) async {
    final speech = await loadWav(wavPath, targetSr: 16000);

    // ONNX kaldi fbank extraction
    final runOpts = OrtRunOptions();
    final inputs = {
      'waveform': OrtValueTensor.createTensorWithDataList(
          Float32List.fromList(speech), [1, speech.length]),
    };
    final fbankOutputs = _fbankSession.run(runOpts, inputs);
    final feat = flattenToFloat32(fbankOutputs[0]!.value);
    final featShape = _getOutputShape(fbankOutputs[0]!.value);
    // feat shape: (T_frames, 80)
    final tFrames = featShape.isNotEmpty ? featShape[0] : (feat.length ~/ 80);
    final dim = featShape.length >= 2 ? featShape[1] : 80;

    // Mean subtraction
    final mean = Float32List(dim);
    for (int i = 0; i < tFrames; i++) {
      for (int j = 0; j < dim; j++) {
        mean[j] += feat[i * dim + j];
      }
    }
    for (int j = 0; j < dim; j++) {
      mean[j] /= tFrames;
    }
    for (int i = 0; i < tFrames; i++) {
      for (int j = 0; j < dim; j++) {
        feat[i * dim + j] -= mean[j];
      }
    }

    // Campplus
    final cpInputs = {
      _campplusSession.inputNames[0]: OrtValueTensor.createTensorWithDataList(
          feat, [1, tFrames, dim]),
    };
    final cpOutputs = _campplusSession.run(runOpts, cpInputs);
    return flattenToFloat32(cpOutputs[0]!.value);
  }

  /// Extract mel spectrogram features for the flow model prompt (24kHz, 80-bin).
  /// Returns Float32List with shape [1, T, 80].
  Future<Float32List> extractPromptSpeechFeat(String wavPath) async {
    final speech = await loadWav(wavPath, targetSr: sampleRate);

    final runOpts = OrtRunOptions();
    final inputs = {
      'waveform': OrtValueTensor.createTensorWithDataList(
          Float32List.fromList(speech), [1, speech.length]),
    };
    final outputs = _mel24kSession.run(runOpts, inputs);
    return flattenToFloat32(outputs[0]!.value);
  }

  /// Tokenize text using BPE tokenizer.
  List<int> tokenizeText(String text) {
    return _tokenizer.encode(text);
  }

  /// Public accessor for tokenizer verification.
  List<int> tokenizerEncode(String text) {
    return _tokenizer.encode(text);
  }

  /// Run full preprocessing and return all needed inputs.
  Future<PreprocData> run(
      String refWav, String promptText, String ttsText) async {
    // Extract prompt speech features (24kHz mel)
    final promptSpeechFeat = await extractPromptSpeechFeat(refWav);
    final featShape = _inferMel24kShape(promptSpeechFeat);
    pLog('[Preproc] mel_24k output: ${promptSpeechFeat.length} values, shape=$featShape', tag: 'Preproc');

    // Extract speech tokens (16kHz)
    final speechTokens = await extractSpeechTokens(refWav);
    pLog('[Preproc] speech_tokens: ${speechTokens.length} tokens', tag: 'Preproc');

    // Align lengths: token_mel_ratio = 2
    final tokenLen = minInt(
        featShape[1] ~/ tokenMelRatio, speechTokens.length);
    final alignedFeatLen = tokenLen * tokenMelRatio;
    pLog('[Preproc] aligned: tokenLen=$tokenLen, alignedFeatLen=$alignedFeatLen (raw_speech_tokens=${speechTokens.length}, raw_feat_T=${featShape[1]})', tag: 'Preproc');

    // Trim prompt_speech_feat to aligned length
    final trimmedFeat = Float32List(1 * alignedFeatLen * melDim);
    for (int i = 0; i < alignedFeatLen * melDim; i++) {
      trimmedFeat[i] = promptSpeechFeat[i];
    }
    final alignedTokens = speechTokens.sublist(0, tokenLen);

    // Speaker embedding
    final speakerEmbedding = await extractSpeakerEmbedding(refWav);
    pLog('[Preproc] speaker_embedding: ${speakerEmbedding.length} values', tag: 'Preproc');

    // Text tokenization
    final promptTextTokens = tokenizeText(promptText);
    final ttsTextTokens = tokenizeText(ttsText);
    pLog('[Preproc] prompt="$promptText"', tag: 'Preproc');
    pLog('[Preproc] prompt_text_tokens (${promptTextTokens.length}): ${shortList(promptTextTokens)}', tag: 'Preproc');
    pLog('[Preproc] tts_text_tokens (${ttsTextTokens.length}): ${shortList(ttsTextTokens)}', tag: 'Preproc');

    return PreprocData(
      promptTextTokens: promptTextTokens,
      ttsTextTokens: ttsTextTokens,
      speechTokens: alignedTokens,
      speakerEmbedding: speakerEmbedding,
      promptSpeechFeat: trimmedFeat,
      promptFeatLen: alignedFeatLen,
      promptFeatShape: [1, alignedFeatLen, melDim],
    );
  }

  /// Release all sessions.
  Future<void> dispose() async {
    await _mel16kSession.release();
    await _mel24kSession.release();
    await _fbankSession.release();
    await _speechTokenizerSession.release();
    await _campplusSession.release();
    _isLoaded = false;
  }

  List<int> _getOutputShape(dynamic value) {
    if (value is List) {
      return getShape(value);
    }
    return [];
  }

  /// Infer shape of mel_24k output from flat data.
  /// Since mel_24k outputs (1, T, 80), we know dim0=1, dim2=80.
  List<int> _inferMel24kShape(Float32List flat) {
    final tFrames = flat.length ~/ melDim;
    return [1, tFrames, melDim];
  }
}

/// Data produced by preprocessing stage.
class PreprocData {
  final List<int> promptTextTokens;
  final List<int> ttsTextTokens;
  final List<int> speechTokens;
  final Float32List speakerEmbedding;
  final Float32List promptSpeechFeat;
  final int promptFeatLen;
  final List<int> promptFeatShape;

  PreprocData({
    required this.promptTextTokens,
    required this.ttsTextTokens,
    required this.speechTokens,
    required this.speakerEmbedding,
    required this.promptSpeechFeat,
    required this.promptFeatLen,
    required this.promptFeatShape,
  });
}

int minInt(int a, int b) => a < b ? a : b;

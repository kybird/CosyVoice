import 'dart:io';
import 'dart:typed_data';
import 'package:onnxruntime_v2/onnxruntime_v2.dart';
import 'constants.dart';
import 'pipeline_logger.dart';
import 'tensor_utils.dart';
import 'preprocessing.dart';

/// LLM autoregressive decoding using ONNX Runtime.
/// Embeddings and logits via llm_embed.onnx.
class LlmInference {
  late OrtSession _embedSession;
  late OrtSession _initialSession;
  late OrtSession _decodeSession;

  // Dummy inputs for llm_embed.onnx (unused outputs computed but discarded)
  final Int64List _dummyTokenIds = Int64List.fromList([0]);
  final Int64List _dummySpeechIds = Int64List.fromList([0]);
  late Float32List _dummyHidden;

  bool _isLoaded = false;
  bool get isLoaded => _isLoaded;

  /// Load LLM ONNX sessions.
  /// [onnxDir] is the directory containing llm_*.onnx files.
  Future<void> load(String onnxDir) async {
    final opts = OrtSessionOptions()
      ..setSessionGraphOptimizationLevel(GraphOptimizationLevel.ortEnableAll)
      ..setIntraOpNumThreads(4)
      ..setInterOpNumThreads(1);

    // Load llm_embed.onnx (embed_tokens + speech_embedding + llm_decoder)
    _embedSession = await _loadSession('$onnxDir/llm_embed.onnx', opts);

    // Load llm_initial.onnx (prefill) — prefer FP32 for quality
    var initialPath = '$onnxDir/llm_initial.onnx';
    if (!await File(initialPath).exists()) {
      initialPath = '$onnxDir/llm_initial_int8.onnx';
    }
    _initialSession = await _loadSession(initialPath, opts);

    // Load llm_decode_int8.onnx (decode step)
    var decodePath = '$onnxDir/llm_decode_int8.onnx';
    if (!await File(decodePath).exists()) {
      decodePath = '$onnxDir/llm_decode.onnx';
    }
    _decodeSession = await _loadSession(decodePath, opts);

    _dummyHidden = Float32List(hiddenSize); // zeros

    _isLoaded = true;
  }

  Future<OrtSession> _loadSession(String path, OrtSessionOptions opts) async {
    final file = File(path);
    if (!await file.exists()) {
      throw Exception('LLM model not found: $path');
    }
    if (Platform.isWindows) {
      final bytes = await file.readAsBytes();
      return OrtSession.fromBuffer(bytes, opts);
    }
    return OrtSession.fromFile(file, opts);
  }

  /// Look up text token embeddings via llm_embed.onnx.
  /// Returns Float32List of shape (N, hiddenSize).
  Future<Float32List> _embedTokens(List<int> tokenIds) async {
    final ids = Int64List.fromList(tokenIds);
    final runOpts = OrtRunOptions();
    final inputs = {
      'token_ids': OrtValueTensor.createTensorWithDataList(ids, [ids.length]),
      'speech_ids':
          OrtValueTensor.createTensorWithDataList(_dummySpeechIds, [1]),
      'hidden_state': OrtValueTensor.createTensorWithDataList(
          _dummyHidden, [1, 1, hiddenSize]),
    };

    // Request specific output by name
    final outputs = _embedSession.run(runOpts, inputs);
    // Output 0 = text_emb (N, 896)
    return flattenToFloat32(outputs[0]!.value);
  }

  /// Get embedding for speech token(s) via llm_embed.onnx.
  /// Returns Float32List of shape (1, N, hiddenSize).
  Future<Float32List> _speechEmbedBatch(List<int> tokenIds) async {
    final ids = Int64List.fromList(tokenIds);
    final runOpts = OrtRunOptions();
    final inputs = {
      'token_ids':
          OrtValueTensor.createTensorWithDataList(_dummyTokenIds, [1]),
      'speech_ids':
          OrtValueTensor.createTensorWithDataList(ids, [ids.length]),
      'hidden_state': OrtValueTensor.createTensorWithDataList(
          _dummyHidden, [1, 1, hiddenSize]),
    };

    final outputs = _embedSession.run(runOpts, inputs);
    // Output 1 = speech_emb (N, 896)
    final flat = flattenToFloat32(outputs[1]!.value);
    // Reshape to (1, N, 896)
    return flat;
  }

  /// Apply linear decoder to hidden state to get logits via ONNX.
  /// hiddenState: flat Float32List of length hiddenSize (896)
  /// Returns Float32List of length llmVocabSize (6761)
  Future<Float32List> _decodeHidden(Float32List hiddenState) async {
    final runOpts = OrtRunOptions();
    final inputs = {
      'token_ids':
          OrtValueTensor.createTensorWithDataList(_dummyTokenIds, [1]),
      'speech_ids':
          OrtValueTensor.createTensorWithDataList(_dummySpeechIds, [1]),
      'hidden_state': OrtValueTensor.createTensorWithDataList(
          hiddenState, [1, 1, hiddenSize]),
    };

    final outputs = _embedSession.run(runOpts, inputs);
    // Release unused outputs (text_emb, speech_emb)
    (outputs[0] as OrtValueTensor).release();
    (outputs[1] as OrtValueTensor).release();
    // Output 2 = logits (1, 1, 6761)
    final logits = flattenToFloat32(outputs[2]!.value);
    (outputs[2] as OrtValueTensor).release();
    return logits;
  }

  /// Run LLM autoregressive decoding.
  /// Returns list of speech token IDs (filtered).
  Future<List<int>> run(PreprocData preprocData) async {
    final promptTextTokens = preprocData.promptTextTokens;
    final ttsTextTokens = preprocData.ttsTextTokens;
    final speechTokens = preprocData.speechTokens;

    // Build LLM input: [sos_emb, text_emb, task_id_emb, prompt_speech_emb]
    final allTextTokens = [...promptTextTokens, ...ttsTextTokens];
    final textEmb = await _embedTokens(allTextTokens); // (N, hiddenSize)

    // SOS + task_id embeddings
    final sosEmb = await _speechEmbedBatch([sosToken]); // (1, 1, hiddenSize)
    final taskIdEmb =
        await _speechEmbedBatch([taskIdToken]); // (1, 1, hiddenSize)

    // Prompt speech token embeddings
    final promptSpeechEmb =
        await _speechEmbedBatch(speechTokens); // (1, N_speech, hiddenSize)

    // Concatenate: sos(1) + text(N_text) + task_id(1) + speech(N_speech)
    final seqLen = 1 + allTextTokens.length + 1 + speechTokens.length;
    final lmInput = Float32List(1 * seqLen * hiddenSize);

    // Fill sos
    for (int j = 0; j < hiddenSize; j++) {
      lmInput[j] = sosEmb[j];
    }
    // Fill text embeddings
    int offset = hiddenSize;
    for (int i = 0; i < allTextTokens.length; i++) {
      for (int j = 0; j < hiddenSize; j++) {
        lmInput[offset + i * hiddenSize + j] =
            textEmb[i * hiddenSize + j];
      }
    }
    offset += allTextTokens.length * hiddenSize;
    // Fill task_id
    for (int j = 0; j < hiddenSize; j++) {
      lmInput[offset + j] = taskIdEmb[j];
    }
    offset += hiddenSize;
    // Fill prompt speech embeddings
    for (int i = 0; i < speechTokens.length; i++) {
      for (int j = 0; j < hiddenSize; j++) {
        lmInput[offset + i * hiddenSize + j] =
            promptSpeechEmb[i * hiddenSize + j];
      }
    }

    // Attention mask: all ones
    final attentionMask = Int64List(seqLen);
    for (int i = 0; i < seqLen; i++) {
      attentionMask[i] = 1;
    }

    // ── Prefill via llm_initial.onnx ──
    pLog('[LLM] Building input: sos(1) + text(${allTextTokens.length}) + taskid(1) + speech(${speechTokens.length}) = seqLen=$seqLen', tag: 'LLM');
    final runOpts = OrtRunOptions();
    final initInputs = {
      'inputs_embeds': OrtValueTensor.createTensorWithDataList(
          lmInput, [1, seqLen, hiddenSize]),
      'attention_mask': OrtValueTensor.createTensorWithDataList(
          attentionMask, [1, seqLen]),
    };

    final initialOutputs = _initialSession.run(runOpts, initInputs);
    // initialOutputs[0] = hidden_state (1, seq_len, 896)
    // initialOutputs[1..48] = KV cache (past_key_0, past_value_0, ..., past_key_23, past_value_23)
    final hiddenState = flattenToFloat32(initialOutputs[0]!.value);

    // Extract last hidden state for first token decoding
    final lastHidden = Float32List(hiddenSize);
    for (int j = 0; j < hiddenSize; j++) {
      lastHidden[j] = hiddenState[(seqLen - 1) * hiddenSize + j];
    }

    // Decode first token
    final logits = await _decodeHidden(lastHidden); // (1, 1, 6761)
    final logp = logSoftmax(Float32List.fromList(logits.sublist(0, llmVocabSize)));

    final textLen = ttsTextTokens.length;
    final minLen = textLen * minTokenTextRatio;
    final maxLen = textLen * maxTokenTextRatio;

    // Stop token set: tokens in range [speechTokenSize, speechTokenSize + 200)
    final stopTokenIds = Set<int>.from(
        List<int>.generate(200, (i) => speechTokenSize + i));

    final outTokens = <int>[];
    int curSilentCount = 0;
    const maxSilentCount = 5;

    // Decode first token
    int topId = topKSample(logp, outTokens,
        topK: samplingTopK, repPenalty: repetitionPenalty);
    pLog('[LLM] First token: $topId', tag: 'LLM');
    if (stopTokenIds.contains(topId)) {
      pLog('[LLM] WARNING: EOS at first token!', tag: 'LLM');
      return _filterTokens(outTokens);
    }
    outTokens.add(topId);

    // KV cache from initial outputs — keep as OrtValue for zero-copy reuse
    // Each layer has past_key and past_value (indices 1..48)
    List<OrtValue> kvCacheOrt = [];
    for (int i = 1; i < initialOutputs.length; i++) {
      kvCacheOrt.add(initialOutputs[i]!);
    }

    // ── Autoregressive decode loop ──
    for (int step = 1; step < maxLen; step++) {
      // Get embedding for last predicted token
      final tokenEmb = await _speechEmbedBatch([topId]); // (1, 1, hiddenSize)

      // Build decode inputs
      final position = seqLen + step - 1;
      final positionIds = Int64List.fromList([position]);

      final decodeInputs = <String, OrtValue>{};
      decodeInputs['inputs_embeds'] = OrtValueTensor.createTensorWithDataList(
          tokenEmb, [1, 1, hiddenSize]);
      decodeInputs['position_ids'] =
          OrtValueTensor.createTensorWithDataList(positionIds, [1, 1]);

      // Add KV cache as inputs (zero-copy: pass OrtValue pointers directly)
      for (int layerIdx = 0; layerIdx < numLayers; layerIdx++) {
        decodeInputs['past_key_${layerIdx}_in'] = kvCacheOrt[layerIdx * 2];
        decodeInputs['past_value_${layerIdx}_in'] = kvCacheOrt[layerIdx * 2 + 1];
      }

      final decodeOutputs = _decodeSession.run(runOpts, decodeInputs);

      // Get hidden state (small: 896 floats)
      final hidden = flattenToFloat32(decodeOutputs[0]!.value);
      (decodeOutputs[0] as OrtValueTensor).release();

      // Release small input OrtValues created this step
      (decodeInputs['inputs_embeds'] as OrtValueTensor).release();
      (decodeInputs['position_ids'] as OrtValueTensor).release();

      // Update KV cache: release old OrtValues, keep new ones (zero-copy)
      for (final ort in kvCacheOrt) {
        (ort as OrtValueTensor).release();
      }
      kvCacheOrt = <OrtValue>[];
      for (int i = 1; i < decodeOutputs.length; i++) {
        kvCacheOrt.add(decodeOutputs[i]!);
      }

      // Decode token
      final stepLogits = await _decodeHidden(hidden);
      var stepLogp = logSoftmax(
          Float32List.fromList(stepLogits.sublist(0, llmVocabSize)));

      // Ignore EOS before min_len
      if (step < minLen) {
        stepLogp[eosToken] = double.negativeInfinity;
      }

      topId = topKSample(stepLogp, outTokens,
          topK: samplingTopK, repPenalty: repetitionPenalty);

      if (stopTokenIds.contains(topId)) {
        pLog('[LLM] Stop token $topId at step $step', tag: 'LLM');
        break;
      }

      // Silent token filtering
      if (silentTokens.contains(topId)) {
        curSilentCount++;
        if (curSilentCount > maxSilentCount) {
          continue;
        }
      } else {
        curSilentCount = 0;
      }

      outTokens.add(topId);

      if ((step + 1) % 20 == 0) {
        pLog('[LLM] Step ${step + 1}/$maxLen, tokens=${outTokens.length}, last=${outTokens.last}', tag: 'LLM');
      }
    }

    // Release remaining KV cache OrtValues
    for (final ort in kvCacheOrt) {
      (ort as OrtValueTensor).release();
    }

    pLog('[LLM] Done: ${outTokens.length} raw tokens, ${_filterTokens(outTokens).length} after filter', tag: 'LLM');
    return _filterTokens(outTokens);
  }

  /// Remove excess silent tokens.
  List<int> _filterTokens(List<int> tokens) {
    final result = <int>[];
    int silentCount = 0;
    for (final t in tokens) {
      if (silentTokens.contains(t)) {
        silentCount++;
        if (silentCount <= 5) {
          result.add(t);
        }
      } else {
        silentCount = 0;
        result.add(t);
      }
    }
    return result;
  }

  /// Release all sessions.
  Future<void> dispose() async {
    await _embedSession.release();
    await _initialSession.release();
    await _decodeSession.release();
    _isLoaded = false;
  }
}

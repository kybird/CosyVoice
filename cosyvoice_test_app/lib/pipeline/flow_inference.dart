import 'dart:io';
import 'dart:typed_data';
import 'package:onnxruntime_v2/onnxruntime_v2.dart';
import 'constants.dart';
import 'pipeline_logger.dart';
import 'tensor_utils.dart';

/// Flow matching inference using flow_prep_mobile.onnx + dit_estimator.
/// Implements the ODE solver (Euler method) with Classifier-Free Guidance.
class FlowInference {
  late OrtSession _flowPrepSession;
  late OrtSession _ditSession;

  bool _isLoaded = false;
  bool get isLoaded => _isLoaded;

  /// Load flow ONNX sessions.
  Future<void> load(String onnxDir) async {
    final opts = OrtSessionOptions()
      ..setSessionGraphOptimizationLevel(GraphOptimizationLevel.ortEnableAll)
      ..setIntraOpNumThreads(4)
      ..setInterOpNumThreads(1);

    // Flow prep
    var prepPath = '$onnxDir/flow_prep_mobile.onnx';
    if (!await File(prepPath).exists()) {
      prepPath = '$onnxDir/flow_prep.onnx';
    }
    _flowPrepSession = await _loadSession(prepPath, opts);

    // DiT estimator — prefer INT8 FFN
    var ditPath = '$onnxDir/dit_estimator_int8_ffn.onnx';
    if (!await File(ditPath).exists()) {
      ditPath = '$onnxDir/dit_estimator_mobile.onnx';
      if (!await File(ditPath).exists()) {
        ditPath = '$onnxDir/dit_estimator.onnx';
      }
    }
    // DiT is the main bottleneck — use more threads
    final ditOpts = OrtSessionOptions()
      ..setSessionGraphOptimizationLevel(GraphOptimizationLevel.ortEnableAll)
      ..setIntraOpNumThreads(0) // use all cores
      ..setInterOpNumThreads(1);
    _ditSession = await _loadSession(ditPath, ditOpts);

    _isLoaded = true;
  }

  Future<OrtSession> _loadSession(String path, OrtSessionOptions opts) async {
    final file = File(path);
    if (!await file.exists()) {
      throw Exception('Flow model not found: $path');
    }
    if (Platform.isWindows) {
      final bytes = await file.readAsBytes();
      return OrtSession.fromBuffer(bytes, opts);
    }
    return OrtSession.fromFile(file, opts);
  }

  /// Run flow matching inference.
  ///
  /// [speechTokens] — generated speech tokens from LLM
  /// [promptTokens] — speech tokens from reference audio
  /// [promptSpeechFeat] — mel features (1, T_prompt, 80) flat
  /// [promptFeatShape] — shape of promptSpeechFeat
  /// [speakerEmbedding] — (192,) flat
  ///
  /// Returns mel spectrogram as Float32List (1, 80, T_mel) flat.
  Future<Float32List> run({
    required List<int> speechTokens,
    required List<int> promptTokens,
    required Float32List promptSpeechFeat,
    required List<int> promptFeatShape,
    required Float32List speakerEmbedding,
  }) async {
    // Prepare inputs for flow_prep_mobile.onnx
    final allTokens = [...promptTokens, ...speechTokens];
    // Clip token IDs to [0, speechTokenSize - 1]
    final clippedTokens = Int64List.fromList(
        allTokens.map((t) => t.clamp(0, speechTokenSize - 1)).toList());

    final spkEmb = Float32List.fromList(speakerEmbedding);

    final runOpts = OrtRunOptions();
    final prepInputs = {
      'token_ids': OrtValueTensor.createTensorWithDataList(
          clippedTokens, [1, clippedTokens.length]),
      'speaker_emb':
          OrtValueTensor.createTensorWithDataList(spkEmb, [1, spkEmbedDim]),
      'prompt_feat': OrtValueTensor.createTensorWithDataList(
          promptSpeechFeat, promptFeatShape),
    };

    final prepOutputs = _flowPrepSession.run(runOpts, prepInputs);
    // mu: (1, 80, T_mel), spks: (1, 80), cond: (1, 80, T_mel)
    final mu = flattenToFloat32(prepOutputs[0]!.value);
    final spks = flattenToFloat32(prepOutputs[1]!.value);
    final cond = flattenToFloat32(prepOutputs[2]!.value);

    // Determine shapes
    final muShape = _getOutputShape(prepOutputs[0]!.value);
    final totalMelLen = muShape.length >= 3 ? muShape[2] : (mu.length ~/ melDim);
    final promptMelLen = promptFeatShape[1];

    pLog('[Flow] flow_prep output: mu=${mu.length} values, shape=$muShape, totalMelLen=$totalMelLen, promptMelLen=$promptMelLen, newMelLen=${totalMelLen - promptMelLen}', tag: 'Flow');

    // ── ODE Solver (Euler method) ──
    // Random noise
    var x = randomNormal(mu.length, seed: 0);

    // Time schedule (cosine)
    final tSpanCosine = cosineSchedule(nTimesteps);

    final mask = Float32List(totalMelLen);
    for (int i = 0; i < totalMelLen; i++) {
      mask[i] = 1.0;
    }

    // Pre-allocate zero buffers for unconditional
    final zerosMu = Float32List(mu.length);
    final zerosCond = Float32List(cond.length);
    final zerosSpks = Float32List(spkDim);

    var tVal = tSpanCosine[0];
    var dt = tSpanCosine[1] - tSpanCosine[0];

    for (int step = 1; step < tSpanCosine.length; step++) {
      final tArr = Float32List.fromList([tVal]);

      // Conditional call
      final condInputs = {
        'x': OrtValueTensor.createTensorWithDataList(
            x, [1, melDim, totalMelLen]),
        'mask': OrtValueTensor.createTensorWithDataList(
            mask, [1, 1, totalMelLen]),
        'mu': OrtValueTensor.createTensorWithDataList(
            mu, [1, melDim, totalMelLen]),
        't': OrtValueTensor.createTensorWithDataList(tArr, [1]),
        'spks': OrtValueTensor.createTensorWithDataList(spks, [1, spkDim]),
        'cond': OrtValueTensor.createTensorWithDataList(
            cond, [1, melDim, totalMelLen]),
      };
      final ditCondOut = _ditSession.run(runOpts, condInputs);
      final vCond = flattenToFloat32(ditCondOut[0]!.value);

      // Unconditional call
      final uncondInputs = {
        'x': OrtValueTensor.createTensorWithDataList(
            x, [1, melDim, totalMelLen]),
        'mask': OrtValueTensor.createTensorWithDataList(
            mask, [1, 1, totalMelLen]),
        'mu': OrtValueTensor.createTensorWithDataList(
            zerosMu, [1, melDim, totalMelLen]),
        't': OrtValueTensor.createTensorWithDataList(tArr, [1]),
        'spks':
            OrtValueTensor.createTensorWithDataList(zerosSpks, [1, spkDim]),
        'cond': OrtValueTensor.createTensorWithDataList(
            zerosCond, [1, melDim, totalMelLen]),
      };
      final ditUncondOut = _ditSession.run(runOpts, uncondInputs);
      final vUncond = flattenToFloat32(ditUncondOut[0]!.value);

      // CFG: dphi_dt = v_cond + guidanceScale * (v_cond - v_uncond)
      final dphiDt = Float32List(x.length);
      for (int i = 0; i < x.length; i++) {
        dphiDt[i] = vCond[i] + guidanceScale * (vCond[i] - vUncond[i]);
      }

      // Euler step: x += dphi_dt * dt
      for (int i = 0; i < x.length; i++) {
        x[i] += dphiDt[i] * dt;
      }

      // Advance time
      tVal += dt;
      if (step < tSpanCosine.length - 1) {
        dt = tSpanCosine[step + 1] - tVal;
      }
    }

    // Extract only the new mel frames (skip prompt portion)
    // x is flat (1, 80, totalMelLen)
    // Extract x[:, :, promptMelLen:]
    final newMelLen = totalMelLen - promptMelLen;
    final melOutput = Float32List(melDim * newMelLen);
    for (int c = 0; c < melDim; c++) {
      for (int t = 0; t < newMelLen; t++) {
        melOutput[c * newMelLen + t] =
            x[c * totalMelLen + (promptMelLen + t)];
      }
    }

    return melOutput;
  }

  List<int> _getOutputShape(dynamic value) {
    if (value is List) return getShape(value);
    return [];
  }

  /// Release all sessions.
  Future<void> dispose() async {
    await _flowPrepSession.release();
    await _ditSession.release();
    _isLoaded = false;
  }
}

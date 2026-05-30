import 'dart:io';
import 'dart:typed_data';
import 'package:onnxruntime_v2/onnxruntime_v2.dart';
import 'constants.dart';
import 'tensor_utils.dart';

/// HiFT vocoder: mel spectrogram -> audio waveform via ONNX Runtime.
class HiftInference {
  late OrtSession _session;

  bool _isLoaded = false;
  bool get isLoaded => _isLoaded;

  /// Load HiFT ONNX model.
  Future<void> load(String onnxDir) async {
    final opts = OrtSessionOptions()
      ..setSessionGraphOptimizationLevel(GraphOptimizationLevel.ortEnableAll)
      ..setIntraOpNumThreads(4)
      ..setInterOpNumThreads(1);

    _session = await _loadSession('$onnxDir/hift.onnx', opts);
    _isLoaded = true;
  }

  Future<OrtSession> _loadSession(String path, OrtSessionOptions opts) async {
    final file = File(path);
    if (!await file.exists()) {
      throw Exception('HiFT model not found: $path');
    }
    if (Platform.isWindows) {
      final bytes = await file.readAsBytes();
      return OrtSession.fromBuffer(bytes, opts);
    }
    return OrtSession.fromFile(file, opts);
  }

  /// Convert mel spectrogram to audio.
  ///
  /// [melSpectrogram] — Float32List with shape (1, 80, T_mel)
  /// [melLen] — T_mel dimension
  ///
  /// Returns Float32List of audio samples at 24kHz.
  Future<Float32List> run(Float32List melSpectrogram, int melLen) async {
    final runOpts = OrtRunOptions();
    final inputs = {
      'speech_feat': OrtValueTensor.createTensorWithDataList(
          melSpectrogram, [1, melDim, melLen]),
    };

    final outputs = _session.run(runOpts, inputs);
    final audio = flattenToFloat32(outputs[0]!.value);
    (outputs[0] as OrtValueTensor).release();
    (inputs['speech_feat'] as OrtValueTensor).release();
    return audio;
  }

  /// Release session.
  Future<void> dispose() async {
    await _session.release();
    _isLoaded = false;
  }
}

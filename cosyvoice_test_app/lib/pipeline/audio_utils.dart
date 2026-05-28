import 'dart:io';
import 'dart:typed_data';
import 'package:path_provider/path_provider.dart';

/// WAV file header structure (RIFF/WAVE, mono, float32).
/// Supports reading and writing WAV files.

/// Read a WAV file and return mono float32 samples at the file's native sample rate.
/// Handles 16-bit PCM and 32-bit float formats.
Future<Float32List> readWav(String path) async {
  final file = File(path);
  final bytes = await file.readAsBytes();
  final data = bytes.buffer.asByteData(bytes.offsetInBytes, bytes.lengthInBytes);

  // Parse RIFF header
  final riff = String.fromCharCodes(bytes.sublist(0, 4));
  if (riff != 'RIFF') throw Exception('Not a WAV file: $path');

  final wave = String.fromCharCodes(bytes.sublist(8, 12));
  if (wave != 'WAVE') throw Exception('Not a WAV file: $path');

  int offset = 12;
  int audioFormat = 0;
  int numChannels = 1;
  int bitsPerSample = 16;
  int dataSize = 0;
  int dataOffset = 0;

  while (offset < bytes.length - 8) {
    final chunkId = String.fromCharCodes(bytes.sublist(offset, offset + 4));
    final chunkSize = data.getUint32(offset + 4, Endian.little);

    if (chunkId == 'fmt ') {
      audioFormat = data.getUint16(offset + 8, Endian.little);
      numChannels = data.getUint16(offset + 10, Endian.little);
      // sampleRate read but not used here (caller reads it separately)
      // data.getUint32(offset + 12, Endian.little);
      bitsPerSample = data.getUint16(offset + 22, Endian.little);
    } else if (chunkId == 'data') {
      dataSize = chunkSize;
      dataOffset = offset + 8;
      break;
    }

    offset += 8 + chunkSize;
    if (chunkSize.isOdd) offset++; // padding byte
  }

  if (dataSize == 0) throw Exception('No data chunk in WAV: $path');

  // Decode samples
  Float32List samples;
  final numSamples = dataSize ~/ (bitsPerSample ~/ 8);

  if (audioFormat == 3) {
    // IEEE float32
    samples = Float32List(numSamples);
    for (int i = 0; i < numSamples; i++) {
      samples[i] = data.getFloat32(dataOffset + i * 4, Endian.little);
    }
  } else if (audioFormat == 1 && bitsPerSample == 16) {
    // 16-bit PCM
    samples = Float32List(numSamples);
    for (int i = 0; i < numSamples; i++) {
      final int16 = data.getInt16(dataOffset + i * 2, Endian.little);
      samples[i] = int16 / 32768.0;
    }
  } else if (audioFormat == 1 && bitsPerSample == 32) {
    // 32-bit PCM int
    samples = Float32List(numSamples);
    for (int i = 0; i < numSamples; i++) {
      final int32 = data.getInt32(dataOffset + i * 4, Endian.little);
      samples[i] = int32 / 2147483648.0;
    }
  } else {
    throw Exception(
        'Unsupported WAV format: audioFormat=$audioFormat bits=$bitsPerSample');
  }

  // Convert stereo to mono
  if (numChannels > 1) {
    final monoLen = samples.length ~/ numChannels;
    final mono = Float32List(monoLen);
    for (int i = 0; i < monoLen; i++) {
      double sum = 0.0;
      for (int ch = 0; ch < numChannels; ch++) {
        sum += samples[i * numChannels + ch];
      }
      mono[i] = sum / numChannels;
    }
    return mono;
  }

  return samples;
}

/// Write a WAV file (mono, 16-bit PCM, 24kHz).
Future<void> writeWav(String path, Float32List samples, int sampleRate) async {
  // Convert float32 [-1.0, 1.0] to int16 [-32768, 32767]
  final int16Data = Int16List(samples.length);
  for (int i = 0; i < samples.length; i++) {
    final clamped = samples[i].clamp(-1.0, 1.0);
    int16Data[i] = (clamped * 32767).round();
  }

  final dataSize = int16Data.length * 2;
  final fileSize = 36 + dataSize;

  final header = ByteData(44);
  // RIFF header
  header.setUint32(0, _stringToUint32('RIFF'), Endian.little);
  header.setUint32(4, fileSize, Endian.little);
  header.setUint32(8, _stringToUint32('WAVE'), Endian.little);
  // fmt chunk
  header.setUint32(12, _stringToUint32('fmt '), Endian.little);
  header.setUint32(16, 16, Endian.little); // chunk size
  header.setUint16(20, 1, Endian.little); // PCM
  header.setUint16(22, 1, Endian.little); // mono
  header.setUint32(24, sampleRate, Endian.little);
  header.setUint32(28, sampleRate * 2, Endian.little); // byte rate
  header.setUint16(32, 2, Endian.little); // block align
  header.setUint16(34, 16, Endian.little); // bits per sample
  // data chunk
  header.setUint32(36, _stringToUint32('data'), Endian.little);
  header.setUint32(40, dataSize, Endian.little);

  final bytes = BytesBuilder();
  bytes.add(header.buffer.asUint8List());
  bytes.add(int16Data.buffer.asUint8List());

  final file = File(path);
  await file.writeAsBytes(bytes.toBytes());
}

int _stringToUint32(String s) {
  return s.codeUnitAt(0) |
      (s.codeUnitAt(1) << 8) |
      (s.codeUnitAt(2) << 16) |
      (s.codeUnitAt(3) << 24);
}

/// Simple linear interpolation resampling.
Float32List resampleLinear(Float32List samples, int fromRate, int toRate) {
  if (fromRate == toRate) return samples;

  final ratio = toRate / fromRate;
  final newLength = (samples.length * ratio).round();
  final result = Float32List(newLength);

  for (int i = 0; i < newLength; i++) {
    final srcPos = i / ratio;
    final srcIdx = srcPos.floor();
    final frac = srcPos - srcIdx;
    final idx0 = srcIdx.clamp(0, samples.length - 1);
    final idx1 = (srcIdx + 1).clamp(0, samples.length - 1);
    result[i] = samples[idx0] * (1.0 - frac) + samples[idx1] * frac;
  }
  return result;
}

/// Trim silence from audio.
Float32List trimSilence(
  Float32List samples, {
  double threshold = 0.01,
  int paddingMs = 50,
  int sampleRate = 16000,
}) {
  int first = 0;
  int last = samples.length - 1;

  for (int i = 0; i < samples.length; i++) {
    if (samples[i].abs() > threshold) {
      first = i;
      break;
    }
  }
  for (int i = samples.length - 1; i >= 0; i--) {
    if (samples[i].abs() > threshold) {
      last = i;
      break;
    }
  }

  final padSamples = (paddingMs * sampleRate / 1000).round();
  first = (first - padSamples).clamp(0, samples.length - 1);
  last = (last + padSamples).clamp(0, samples.length - 1);

  return samples.sublist(first, last + 1);
}

/// Load WAV, resample to target rate, mono, optionally trim silence.
Future<Float32List> loadWav(
  String path, {
  int targetSr = 16000,
  bool trimSilenceFlag = true,
}) async {
  // Read raw samples
  final raw = await readWav(path);
  // We need to get the source sample rate from the file
  // Re-read just to get SR (simplified: just read and resample)
  final bytes = await File(path).readAsBytes();
  final data = bytes.buffer.asByteData(bytes.offsetInBytes, bytes.lengthInBytes);
  int sr = 16000;
  int offset = 12;
  while (offset < bytes.length - 8) {
    final chunkId = String.fromCharCodes(bytes.sublist(offset, offset + 4));
    final chunkSize = data.getUint32(offset + 4, Endian.little);
    if (chunkId == 'fmt ') {
      sr = data.getUint32(offset + 12, Endian.little);
      break;
    }
    offset += 8 + chunkSize;
    if (chunkSize.isOdd) offset++;
  }

  Float32List result = raw;
  if (sr != targetSr) {
    result = resampleLinear(result, sr, targetSr);
  }
  if (trimSilenceFlag) {
    result = trimSilence(result, sampleRate: targetSr);
  }
  return result;
}

/// Get output directory for generated audio.
Future<String> getOutputDir() async {
  final dir = await getTemporaryDirectory();
  return dir.path;
}

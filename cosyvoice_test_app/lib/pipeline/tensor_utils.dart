import 'dart:math' as math;
import 'dart:typed_data';

/// Numerically stable softmax over a Float32List of logits.
Float32List softmax(Float32List logits) {
  final result = Float32List(logits.length);
  double maxVal = -double.infinity;
  for (int i = 0; i < logits.length; i++) {
    if (logits[i] > maxVal) maxVal = logits[i];
  }
  double sumExp = 0.0;
  for (int i = 0; i < logits.length; i++) {
    final expVal = math.exp(logits[i] - maxVal);
    result[i] = expVal;
    sumExp += expVal;
  }
  for (int i = 0; i < result.length; i++) {
    result[i] /= sumExp;
  }
  return result;
}

/// Numerically stable log-softmax over a Float32List of logits.
Float32List logSoftmax(Float32List logits) {
  final result = Float32List(logits.length);
  double maxVal = -double.infinity;
  for (int i = 0; i < logits.length; i++) {
    if (logits[i] > maxVal) maxVal = logits[i];
  }
  double logSumExp = 0.0;
  for (int i = 0; i < logits.length; i++) {
    logSumExp += math.exp(logits[i] - maxVal);
  }
  logSumExp = maxVal + math.log(logSumExp);
  for (int i = 0; i < logits.length; i++) {
    result[i] = logits[i] - logSumExp;
  }
  return result;
}

/// Generate random normal (Gaussian) samples using Box-Muller.
Float32List randomNormal(int size, {int? seed}) {
  final rand = math.Random(seed ?? DateTime.now().microsecondsSinceEpoch);
  final result = Float32List(size);
  for (int i = 0; i < size; i += 2) {
    final u1 = 1.0 - rand.nextDouble();
    final u2 = 1.0 - rand.nextDouble();
    final mag = math.sqrt(-2.0 * math.log(u1));
    result[i] = mag * math.cos(2.0 * math.pi * u2);
    if (i + 1 < size) {
      result[i + 1] = mag * math.sin(2.0 * math.pi * u2);
    }
  }
  return result;
}

/// Flatten a nested `List<dynamic>` to Float32List.
Float32List flattenToFloat32(dynamic value) {
  if (value is Float32List) return value;
  if (value is List) {
    final flat = <double>[];
    _flattenRecursive(value, flat);
    return Float32List.fromList(flat);
  }
  throw Exception('Expected Float32List or List, got ${value.runtimeType}');
}

void _flattenRecursive(List data, List<double> out) {
  for (final element in data) {
    if (element is List) {
      _flattenRecursive(element, out);
    } else if (element is num) {
      out.add(element.toDouble());
    }
  }
}

/// Flatten a nested `List<dynamic>` to Int32List.
Int32List flattenToInt32(dynamic value) {
  if (value is Int32List) return value;
  if (value is List) {
    final flat = <int>[];
    _flattenRecursiveInt(value, flat);
    return Int32List.fromList(flat);
  }
  throw Exception('Expected Int32List or List, got ${value.runtimeType}');
}

void _flattenRecursiveInt(List data, List<int> out) {
  for (final element in data) {
    if (element is List) {
      _flattenRecursiveInt(element, out);
    } else if (element is int) {
      out.add(element);
    }
  }
}

/// Flatten a nested `List<dynamic>` to Int64List.
Int64List flattenToInt64(dynamic value) {
  if (value is Int64List) return value;
  if (value is List) {
    final flat = <int>[];
    _flattenRecursiveInt(value, flat);
    return Int64List.fromList(flat);
  }
  throw Exception('Expected Int64List or List, got ${value.runtimeType}');
}

/// Get shape of a nested list.
List<int> getShape(dynamic value) {
  final shape = <int>[];
  dynamic current = value;
  while (current is List && current.isNotEmpty) {
    shape.add(current.length);
    current = current[0];
  }
  return shape;
}

/// Create a 1D Float32List filled with zeros.
Float32List zerosFloat32(int size) => Float32List(size);

/// Create a 1D Int64List filled with ones.
Int64List onesInt64(int size) {
  final data = Int64List(size);
  for (int i = 0; i < size; i++) {
    data[i] = 1;
  }
  return data;
}

/// Cosine schedule: 1 - cos(t * pi/2)
List<double> cosineSchedule(int nSteps) {
  final tSpan = List<double>.generate(nSteps + 1, (i) => i / nSteps);
  return tSpan.map((t) => 1.0 - math.cos(t * 0.5 * math.pi)).toList();
}

/// Multinomial sampling from probabilities, returns index.
int multinomialSample(Float32List probs, math.Random rand) {
  final r = rand.nextDouble();
  double cumProb = 0.0;
  for (int i = 0; i < probs.length; i++) {
    cumProb += probs[i];
    if (cumProb >= r) return i;
  }
  return probs.length - 1;
}

/// Top-k sampling with repetition penalty.
/// Returns sampled token ID.
int topKSample(
  Float32List logits,
  List<int> decodedTokens, {
  int topK = 10,
  double repPenalty = 1.2,
  int? seed,
}) {
  final rand = math.Random(seed ?? DateTime.now().microsecondsSinceEpoch);
  final logitsCopy = Float32List.fromList(logits);

  // Apply repetition penalty to recently generated tokens
  if (decodedTokens.isNotEmpty && repPenalty > 1.0) {
    final recent = <int>{};
    final start = math.max(0, decodedTokens.length - 20);
    for (int i = start; i < decodedTokens.length; i++) {
      recent.add(decodedTokens[i]);
    }
    for (final tokId in recent) {
      if (tokId < logitsCopy.length) {
        if (logitsCopy[tokId] > 0) {
          logitsCopy[tokId] /= repPenalty;
        } else {
          logitsCopy[tokId] *= repPenalty;
        }
      }
    }
  }

  final prob = softmax(logitsCopy);

  // Get sorted indices by probability (descending)
  final indices = List<int>.generate(prob.length, (i) => i);
  indices.sort((a, b) => prob[b].compareTo(prob[a]));

  final candidates = <int>[];
  double cumProb = 0.0;
  for (int i = 0; i < indices.length; i++) {
    if (cumProb < 0.8 && candidates.length < topK) {
      cumProb += prob[indices[i]];
      candidates.add(indices[i]);
    } else {
      break;
    }
  }
  if (candidates.isEmpty) candidates.add(indices[0]);

  // Compute weights for candidates
  final candLogits = Float32List(candidates.length);
  for (int i = 0; i < candidates.length; i++) {
    candLogits[i] = logitsCopy[candidates[i]];
  }
  final weights = softmax(candLogits);

  final idx = multinomialSample(weights, rand);
  return candidates[idx];
}

/// Element-wise in-place tensor operations for Float32List.
void tensorAddInPlace(Float32List a, Float32List b) {
  for (int i = 0; i < a.length; i++) {
    a[i] += b[i];
  }
}

void tensorScaleInPlace(Float32List a, double scale) {
  for (int i = 0; i < a.length; i++) {
    a[i] *= scale;
  }
}

void tensorSubtract(Float32List a, Float32List b, Float32List out) {
  for (int i = 0; i < a.length; i++) {
    out[i] = a[i] - b[i];
  }
}

/// Reshape flat Float32List to nested list given shape.
/// Only supports 2D and 3D reshapes.
List<dynamic> reshapeFlat(Float32List flat, List<int> shape) {
  if (shape.length == 1) {
    return flat.toList();
  }
  if (shape.length == 2) {
    final rows = shape[0];
    final cols = shape[1];
    final result = <List<double>>[];
    for (int i = 0; i < rows; i++) {
      final row = <double>[];
      for (int j = 0; j < cols; j++) {
        row.add(flat[i * cols + j]);
      }
      result.add(row);
    }
    return result;
  }
  // 3D
  final d0 = shape[0];
  final d1 = shape[1];
  final d2 = shape[2];
  final result = <List<List<double>>>[];
  for (int i = 0; i < d0; i++) {
    final mat = <List<double>>[];
    for (int j = 0; j < d1; j++) {
      final row = <double>[];
      for (int k = 0; k < d2; k++) {
        row.add(flat[i * d1 * d2 + j * d2 + k]);
      }
      mat.add(row);
    }
    result.add(mat);
  }
  return result;
}

import 'dart:convert';
import 'dart:io';

/// BPE Tokenizer for CosyVoice3 (Qwen2 style, byte-level BPE).
/// Parses HuggingFace tokenizer.json format.
class BpeTokenizer {
  late final Map<String, int> _vocab;
  late final List<List<String>> _merges;
  late final Map<String, int> _specialTokens;
  late final Map<int, String> _idToToken;
  late final List<String> _sortedSpecialTokens;

  // Cached GPT-2 byte-to-unicode mappings
  static final List<String> _byteToChar = _buildByteToChar();
  static final Map<int, int> _charToByte = _buildCharToByte();

  static List<String> _buildByteToChar() {
    // Match Python's bytes_to_unicode() exactly:
    // bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    // = range(33,127) + range(161,173) + range(174,256)
    final bs = <int>[
      for (int i = 33; i <= 126; i++) i, // 94 values: !"#$%&'()*+...
      for (int i = 161; i <= 172; i++) i, // 12 values: ¡¢£¤¥¦§¨©ª«¬
      for (int i = 174; i <= 255; i++) i, // 82 values: ®¯°±²³...
    ];
    final bsSet = bs.toSet();

    // GPT-2 mapping: good byte b → char(b), bad byte b → char(256+n)
    final mapping = List<String>.filled(256, '');
    int n = 0;
    for (int b = 0; b < 256; b++) {
      if (bsSet.contains(b)) {
        mapping[b] = String.fromCharCode(b);
      } else {
        mapping[b] = String.fromCharCode(256 + n);
        n++;
      }
    }
    return mapping;
  }

  static Map<int, int> _buildCharToByte() {
    final mapping = <int, int>{};
    for (int b = 0; b < 256; b++) {
      mapping[_byteToChar[b].codeUnitAt(0)] = b;
    }
    return mapping;
  }

  /// Load tokenizer from tokenizer.json file.
  Future<void> load(String tokenizerJsonPath) async {
    final file = File(tokenizerJsonPath);
    if (!await file.exists()) {
      throw Exception('Tokenizer file not found: $tokenizerJsonPath');
    }

    final content = await file.readAsString();
    final json = jsonDecode(content) as Map<String, dynamic>;

    // Parse model section
    final model = json['model'] as Map<String, dynamic>;
    final vocabRaw = model['vocab'] as Map<String, dynamic>;
    _vocab = vocabRaw.map((k, v) => MapEntry(k, v as int));

    final mergesRaw = model['merges'] as List<dynamic>;
    _merges = mergesRaw.map((m) {
      if (m is String) {
        final parts = m.split(' ');
        return parts.length == 2 ? parts : <String>[];
      } else if (m is List) {
        return m.map((e) => e.toString()).toList();
      }
      return <String>[];
    }).where((pair) => pair.length == 2).toList();

    // Build reverse vocab
    _idToToken = _vocab.map((k, v) => MapEntry(v, k));

    // Parse added_tokens (special tokens)
    _specialTokens = {};
    final addedTokens = json['added_tokens'] as List<dynamic>;
    for (final token in addedTokens) {
      final t = token as Map<String, dynamic>;
      final content = t['content'] as String;
      final id = t['id'] as int;
      if (t['special'] == true) {
        _specialTokens[content] = id;
      }
      _vocab.putIfAbsent(content, () => id);
      _idToToken.putIfAbsent(id, () => content);
    }

    // Sort special tokens by length descending for greedy matching
    _sortedSpecialTokens = _specialTokens.keys.toList()
      ..sort((a, b) => b.length.compareTo(a.length));

    print('[Tokenizer] Loaded: ${_vocab.length} vocab, ${_merges.length} merges, ${_specialTokens.length} special tokens');
    print('[Tokenizer] Special tokens: ${_specialTokens.keys.take(10).toList()}...');
  }

  /// Encode text to token IDs.
  List<int> encode(String text) {
    final segments = _splitSpecialTokens(text);
    final List<int> allIds = [];
    for (final segment in segments) {
      if (segment.isSpecial) {
        final id = _specialTokens[segment.text];
        if (id != null) {
          allIds.add(id);
        }
      } else {
        allIds.addAll(_encodeText(segment.text));
      }
    }
    return allIds;
  }

  /// Decode token IDs back to text.
  String decode(List<int> ids, {bool skipSpecial = true}) {
    final buf = StringBuffer();
    for (final id in ids) {
      final token = _idToToken[id];
      if (token == null) continue;
      if (skipSpecial && _specialTokens.containsValue(id)) continue;
      buf.write(_decodeToken(token));
    }
    return buf.toString();
  }

  List<int> _encodeText(String text) {
    // Step 1: Pre-tokenize using GPT-2 regex (Split pre-tokenizer from tokenizer.json)
    final chunks = _preTokenize(text);

    final List<int> allIds = [];
    for (final chunk in chunks) {
      // Step 2: Byte-level encoding: convert chunk to bytes, then to byte-level unicode chars
      final bytes = utf8.encode(chunk);
      final byteChars = bytes.map((b) => _byteToChar[b]).toList();
      if (byteChars.isEmpty) continue;

      // Step 3: Apply BPE merges to this chunk
      final bpeTokens = _applyBpe(byteChars);

      // Step 4: Map to IDs
      for (final token in bpeTokens) {
        final id = _vocab[token];
        if (id != null) {
          allIds.add(id);
        }
      }
    }
    return allIds;
  }

  /// GPT-2 pre-tokenizer regex: splits text into chunks before BPE.
  /// Pattern from tokenizer.json (expanded (?i:...) for Dart compatibility):
  ///   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
  static final RegExp _preTokenPattern = RegExp(
    r"'(?:[sStTdDmM]|[rR][eE]|[vV][eE]|[lL][lL])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    unicode: true,
    dotAll: true,
  );

  /// Split text into pre-tokenized chunks using GPT-2 regex.
  List<String> _preTokenize(String text) {
    final matches = _preTokenPattern.allMatches(text);
    return matches.map((m) => m.group(0)!).toList();
  }

  /// Apply BPE merges to a list of tokens.
  List<String> _applyBpe(List<String> tokens) {
    if (tokens.length <= 1) return tokens;

    var current = List<String>.from(tokens);

    for (final merge in _merges) {
      final left = merge[0];
      final right = merge[1];
      final merged = '$left$right';

      int i = 0;
      while (i < current.length - 1) {
        if (current[i] == left && current[i + 1] == right) {
          current[i] = merged;
          current.removeAt(i + 1);
        } else {
          i++;
        }
      }
    }

    return current;
  }

  /// Split text into special token and regular text segments.
  List<_Segment> _splitSpecialTokens(String text) {
    if (_sortedSpecialTokens.isEmpty || text.isEmpty) {
      return [_Segment(text, false)];
    }

    final segments = <_Segment>[];
    var remaining = text;

    while (remaining.isNotEmpty) {
      bool found = false;
      for (final special in _sortedSpecialTokens) {
        if (remaining.startsWith(special)) {
          segments.add(_Segment(special, true));
          remaining = remaining.substring(special.length);
          found = true;
          break;
        }
      }
      if (!found) {
        int nextSpecialPos = remaining.length;
        for (final special in _sortedSpecialTokens) {
          final idx = remaining.indexOf(special);
          if (idx > 0 && idx < nextSpecialPos) {
            nextSpecialPos = idx;
          }
        }
        segments.add(_Segment(remaining.substring(0, nextSpecialPos), false));
        remaining = remaining.substring(nextSpecialPos);
      }
    }

    return segments;
  }

  /// Decode a byte-level BPE token back to a string.
  String _decodeToken(String token) {
    final bytes = <int>[];
    for (int i = 0; i < token.length; i++) {
      final byteVal = _charToByte[token.codeUnitAt(i)];
      if (byteVal != null) {
        bytes.add(byteVal);
      }
    }
    try {
      return utf8.decode(bytes);
    } catch (_) {
      return String.fromCharCodes(bytes);
    }
  }
}

class _Segment {
  final String text;
  final bool isSpecial;
  _Segment(this.text, this.isSpecial);
}

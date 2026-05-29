// CosyVoice3 model constants from cosyvoice3.yaml
// All dimensions, sampling parameters, and token IDs

const int hiddenSize = 896;
const int numLayers = 24;
const int numHeads = 14;
const int numKvHeads = 2;
const int headDim = 64;
const int speechTokenSize = 6561;
const int llmVocabSize = 6761; // speechTokenSize + 200
const int llmInputSize = 896;
const int llmOutputSize = 896;
const int spkEmbedDim = 192;
const int sampleRate = 24000;
const int tokenFrameRate = 25;
const int tokenMelRatio = 2;
const int preLookaheadLen = 3;
const int inputFrameRate = 50;
const int melDim = 80;
const int spkDim = 80;

// Flow/CFG parameters
const double guidanceScale = 0.7;
const int nTimesteps = 4;

// CFG skip: skip unconditional pass on steps where t < threshold.
// Flow Matching t=1 is pure noise (CFG critical), t=0 is final (CFG negligible).
// Set to 0.0 to disable (always apply CFG on all steps).
const double cfgSkipThreshold = 0.3;

// Sampling parameters (Repetition Aware Sampling)
const int samplingTopK = 10;
const double repetitionPenalty = 1.2;

// LLM decode length limits
const int maxTokenTextRatio = 20;
const int minTokenTextRatio = 2;

// Special token IDs (match Python: SPEECH_TOKEN_SIZE + N)
const int sosToken = 6561; // speechTokenSize + 0
const int eosToken = 6562; // speechTokenSize + 1
const int taskIdToken = 6563; // speechTokenSize + 2

// Silent/breath tokens to filter
const Set<int> silentTokens = {
  1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323
};

// Reference audio prompt text mapping
const Map<String, String> refPromptMap = {
  'ref_03s': '안녕하세요 오늘 날씨가 정말 좋네요.',
  'ref_06s': '안녕하세요 저는 오늘 이렇게 만나서 정말 반갑습니다.',
  'ref_15s':
      '안녕하세요 저는 오늘 이렇게 만나서 정말 반갑습니다. 오랜만에 뵙네요. 정말 좋은 하루 되세요.',
  'reference': '안녕하세요 오늘 날씨가 정말 좋네요.',
};

// Default TTS text
const String defaultTtsText = '안녕하세요, 반갑습니다.';

// Default Android model directory
const String defaultModelDir = '/storage/emulated/0/CosyVoice';

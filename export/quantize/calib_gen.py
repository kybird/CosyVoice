"""Generate calibration data for llm_initial.onnx quantization.

Captures actual inputs_embeds and attention_mask from the ONNX pipeline
using diverse Korean texts, then saves as calib_data.npz.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # CosyVoice root
sys.path.insert(0, str(BASE_DIR))
from paths import MODEL_DIR, ONNX_DIR, TTSTEXTVIEWER_DIR, REF_WAV_SUBDIR
from cosyvoice.tokenizer.tokenizer_lite import CosyVoice3TokenizerLite

# Korean calibration texts (diverse phonemes, lengths)
KOREAN_TEXTS = [
    "안녕하세요, 반갑습니다.",
    "오늘 날씨가 정말 좋네요.",
    "저는 서울에 살고 있습니다.",
    "내일 회의가 몇 시에 시작하나요?",
    "감사합니다, 좋은 하루 되세요.",
    "이 음식은 정말 맛있네요.",
    "주말에 친구들을 만날 예정입니다.",
    "커피 한 잔 주시겠어요?",
    "영화가 정말 재미있었어요.",
    "다음 주에 출장을 가야 합니다.",
    "아이들이 학교에서 돌아왔습니다.",
    "이 책은 아주 흥미롭습니다.",
    "운동을 열심히 하고 있습니다.",
    "생일 축하합니다!",
    "비가 오니까 우산을 챙기세요.",
    "새로운 프로젝트를 시작했습니다.",
    "건강이 가장 중요합니다.",
    "여행 가고 싶네요.",
    "밥 먹었어요?",
    "컴퓨터가 갑자기 꺼졌어요.",
    "회사까지 얼마나 걸려요?",
    "한국어를 공부하고 있습니다.",
    "내년에 결혼할 예정입니다.",
    "치과 예약을 해야겠어요.",
    "택시 타고 가는 게 빠를 거예요.",
    "고양이가 귀여워요.",
    "피자 시켜먹을까요?",
    "추운데 따뜻하게 입으세요.",
    "노래를 잘 부르시네요.",
    "다음 정류장에서 내리세요.",
    "핸드폰 배터리가 없어요.",
    "회의실을 예약해 주세요.",
    "운전 조심하세요.",
    "맛있는 점심 먹었어요?",
    "좋은 생각이에요!",
    "빨리 와주세요.",
    "시간이 참 빨리 가네요.",
    "처음 뵙겠습니다.",
    "편의점이 어디에 있어요?",
    "죄송합니다, 늦어서요.",
    "행복한 하루 보내세요.",
    "이번 달은 바쁘겠네요.",
    "영어를 배우고 싶어요.",
    "강아지 산책시키러 갈 거예요.",
    "냉장고에 먹을 게 없어요.",
    "지하철이 곧 올 거예요.",
    "새 구두를 샀어요.",
    "숙제 다 했어요?",
    "봄이 왔네요!",
    "배가 고파요.",
]

SOS_ID = 6561  # speech token offset for sos
TASK_ID = 6562  # speech token offset for task_id
SPEECH_TOKEN_SIZE = 6561


def load_wav_16k(path):
    """Load WAV and resample to 16kHz."""
    speech, sr = sf.read(path, dtype='float32')
    if speech.ndim > 1:
        speech = speech.mean(axis=1)
    if sr != 16000:
        gcd = np.gcd(sr, 16000)
        speech = resample_poly(speech, 16000 // gcd, sr // gcd)
    # Trim
    energy = np.abs(speech)
    above = np.where(energy > 0.01)[0]
    if len(above) > 0:
        pad = int(50 * 16000 / 1000)
        first = max(0, above[0] - pad)
        last = min(len(speech) - 1, above[-1] + pad)
        speech = speech[first:last + 1]
    return speech


def main():
    model_dir = str(MODEL_DIR)
    onnx_dir = str(ONNX_DIR)
    output_path = str(Path(__file__).parent / "calib_data.npz")

    # Load tokenizer
    tok = CosyVoice3TokenizerLite(token_path=str(Path(model_dir) / "CosyVoice-BlankEN"), skip_special_tokens=True)

    # Load speech tokens from ref audio (reuse for all samples)
    ref_path = str(TTSTEXTVIEWER_DIR / REF_WAV_SUBDIR / "ref_03s.wav")
    ref_16k = load_wav_16k(ref_path)

    # Extract mel and speech tokens
    mel_sess = ort.InferenceSession(
        str(Path(onnx_dir) / "mel_16k_128bin.onnx"),
        providers=["CPUExecutionProvider"],
    )
    mel = mel_sess.run(None, {"waveform": ref_16k.reshape(1, -1).astype(np.float32)})[0]

    sp_sess = ort.InferenceSession(
        str(Path(model_dir) / "speech_tokenizer_v3.onnx"),
        providers=["CPUExecutionProvider"],
    )
    speech_tokens = sp_sess.run(None, {
        sp_sess.get_inputs()[0].name: mel,
        sp_sess.get_inputs()[1].name: np.array([mel.shape[2]], dtype=np.int32),
    })[0].flatten().tolist()

    # Load llm_embed for generating embeddings
    embed_sess = ort.InferenceSession(
        str(Path(onnx_dir) / "llm_embed.onnx"),
        providers=["CPUExecutionProvider"],
    )
    dummy_ids = np.array([0], dtype=np.int64)
    dummy_hidden = np.zeros((1, 1, 896), dtype=np.float32)

    all_embeds = []
    all_masks = []
    target_seq_len = 122  # pad/truncate to this length

    prompt_text = "You are a helpful assistant.<|endofprompt|>안녕하세요 오늘 날씨가 정말 좋네요."

    for text in KOREAN_TEXTS:
        # Tokenize
        prompt_tokens = tok.encode(prompt_text, allowed_special="all")
        tts_tokens = tok.encode(text, allowed_special="all")
        all_text_tokens = prompt_tokens + tts_tokens

        # Get text embeddings
        text_ids = np.array(all_text_tokens, dtype=np.int64)
        text_emb = embed_sess.run(
            ["text_emb"],
            {"token_ids": text_ids, "speech_ids": dummy_ids, "hidden_state": dummy_hidden},
        )[0]  # (N, 896)
        text_emb = text_emb[np.newaxis, :, :]  # (1, N, 896)

        # SOS embedding
        sos_ids = np.array([SOS_ID], dtype=np.int64)
        sos_emb = embed_sess.run(
            ["speech_emb"],
            {"token_ids": dummy_ids, "speech_ids": sos_ids, "hidden_state": dummy_hidden},
        )[0][np.newaxis, :, :]  # (1, 1, 896)

        # Task ID embedding
        task_ids = np.array([TASK_ID], dtype=np.int64)
        task_emb = embed_sess.run(
            ["speech_emb"],
            {"token_ids": dummy_ids, "speech_ids": task_ids, "hidden_state": dummy_hidden},
        )[0][np.newaxis, :, :]  # (1, 1, 896)

        # Speech token embeddings
        sp_ids = np.array(speech_tokens, dtype=np.int64)
        sp_emb = embed_sess.run(
            ["speech_emb"],
            {"token_ids": dummy_ids, "speech_ids": sp_ids, "hidden_state": dummy_hidden},
        )[0][np.newaxis, :, :]  # (1, N, 896)

        # Concatenate: [sos, text_emb, task_id, speech_tokens]
        lm_input = np.concatenate([sos_emb, text_emb, task_emb, sp_emb], axis=1)
        seq_len = lm_input.shape[1]

        # Pad or truncate to target_seq_len
        if seq_len < target_seq_len:
            pad_len = target_seq_len - seq_len
            # Pad with small random noise instead of zeros to avoid NaN in calibration
            padding = np.random.normal(0, 0.01, size=(1, pad_len, 896)).astype(np.float32)
            lm_input = np.concatenate([lm_input, padding], axis=1)
            mask = np.concatenate([np.ones(seq_len, dtype=np.int64), np.ones(pad_len, dtype=np.int64)])
        elif seq_len > target_seq_len:
            lm_input = lm_input[:, :target_seq_len, :]
            mask = np.ones(target_seq_len, dtype=np.int64)
        else:
            mask = np.ones(target_seq_len, dtype=np.int64)

        all_embeds.append(lm_input.squeeze(0))  # (122, 896)
        all_masks.append(mask)  # (122,)

    embeds_arr = np.stack(all_embeds).astype(np.float32)  # (50, 122, 896)
    masks_arr = np.stack(all_masks).astype(np.int64)  # (50, 122)

    np.savez(output_path, inputs_embeds=embeds_arr, attention_mask=masks_arr)
    print(f"Calibration data saved: {output_path}")
    print(f"  inputs_embeds shape: {embeds_arr.shape}")
    print(f"  attention_mask shape: {masks_arr.shape}")
    print(f"  N={len(embeds_arr)} samples")


if __name__ == "__main__":
    main()

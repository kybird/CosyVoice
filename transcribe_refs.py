import whisper
import os, glob, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import TTSTEXTVIEWER_DIR, REF_WAV_SUBDIR

model = whisper.load_model('base')

ref_dir = str(TTSTEXTVIEWER_DIR / REF_WAV_SUBDIR)
files = sorted(glob.glob(os.path.join(ref_dir, 'ref_*s.wav')))

for f in files:
    name = os.path.basename(f)
    result = model.transcribe(f, language='ko')
    text = result["text"].strip()
    print(f'{name}: {text}')

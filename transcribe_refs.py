import whisper
import os, glob

model = whisper.load_model('base')

ref_dir = r'C:\Project\TTSTextReader\TTSTextViewer\openvoice'
files = sorted(glob.glob(os.path.join(ref_dir, 'ref_*s.wav')))

for f in files:
    name = os.path.basename(f)
    result = model.transcribe(f, language='ko')
    text = result["text"].strip()
    print(f'{name}: {text}')

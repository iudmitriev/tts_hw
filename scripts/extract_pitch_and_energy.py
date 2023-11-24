import numpy as np
from pathlib import Path
from tqdm import tqdm
import pyworld
import librosa
from scipy import interpolate

ROOT_PATH = Path(__file__).absolute().resolve().parent

def extract_pitch(data_dir = None):
    if data_dir is None:
        data_dir = ROOT_PATH / "data" / "datasets" / "ljspeech"

    wav_dir = data_dir / "wavs"
    mel_dir = data_dir / "mels"
    pitch_dir = data_dir / "pitch"
    pitch_dir.mkdir(exist_ok=True, parents=True)

    for i, wav_path in tqdm(enumerate(wav_dir.iterdir())):
        audio, sr = librosa.load(wav_path, dtype=np.float64)

        mel_name = f"ljspeech-mel-{(i+1):05d}.npy"
        mel = np.load(mel_dir / mel_name)

        frame_period = (audio.shape[0] / sr * 1000) / mel.shape[0]
        _f0, t = pyworld.dio(audio, sr, frame_period=frame_period)
        f0 = pyworld.stonemask(audio, _f0, t, sr)[:mel.shape[0]].astype(np.float32)
        
        x = np.arange(f0.shape[0])[f0 != 0]
        y = f0[f0 != 0]
        below, above = f0[f0 != 0][0], f0[f0 != 0][-1]
        transform = interpolate.interp1d(x, y, bounds_error=False, fill_value = (below, above))
        f0 = transform(np.arange(f0.shape[0]))
        pitch_name = f"ljspeech-pitch-{(i+1):05d}.npy"
        np.save(pitch_dir / pitch_name, f0)


def extract_energy(data_dir = None):
    if data_dir is None:
        data_dir = ROOT_PATH / "data" / "datasets" / "ljspeech"
        
    energy_dir = data_dir / "energy"
    mel_dir = data_dir / "mels"
    energy_dir.mkdir(exist_ok=True, parents=True)

    for mel_path in mel_dir.iterdir():
        mel = np.load(mel_path)
        energy = np.linalg.norm(mel, axis=-1)
        energy_name = mel_path.name.replace('mel', 'energy')
        np.save(energy_dir / energy_name, energy)


if __name__ == '__main__':
    print('Extracting energy...')
    extract_energy()
    print('Finished extracting energy!')
    print('Extracting pitch...')
    extract_pitch()
    print('Finished extracting pitch!')

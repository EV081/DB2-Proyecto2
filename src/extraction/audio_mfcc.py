import os
import librosa
import numpy as np
import soundfile as sf

VALID_EXTENSIONS = ('.wav', '.mp3', '.flac', '.m4a', '.ogg')


def _load_audio(audio_path: str, target_sr: int) -> tuple[np.ndarray, int]:
    y, sr = sf.read(audio_path, dtype='float32', always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr, res_type='soxr_qq')
        sr = target_sr
    return y, sr


def extract_mfcc_features(
    audio_path: str,
    n_mfcc: int = 13,
    window_ms: int = 200,
    hop_ms: int = 100,
    target_sr: int = 16000,
) -> np.ndarray:
    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in VALID_EXTENSIONS:
        raise ValueError(f"Formato no soportado: '{ext}'. Use: {VALID_EXTENSIONS}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio no encontrado: {audio_path}")

    y, sr = _load_audio(audio_path, target_sr)
    n_fft = int(sr * window_ms / 1000)
    hop_length = int(sr * hop_ms / 1000)

    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        center=False,
    )
    return mfcc.T.astype(np.float32)


__all__ = [
    "extract_mfcc_features",
]

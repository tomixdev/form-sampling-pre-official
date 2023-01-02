import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display


def draw_waveform(y, sr):
    plt.figure(figsize=(15, 3))
    librosa.display.waveshow(y, sr, alpha=0.8)


def draw_spectrogram(y, sr):
    harmonics = [1]
    S = np.abs(librosa.stft(y))
    fft_freqs = librosa.fft_frequencies(sr=sr)
    S_harm = librosa.interp_harmonics(S, freqs=fft_freqs, harmonics=harmonics, axis=0)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, _sh in enumerate(S_harm):
        img = librosa.display.specshow(librosa.amplitude_to_db(_sh,
                                                               ref=S.max()),
                                       sr=sr, y_axis='log', x_axis='time')
    fig.colorbar(img, ax=ax, format="%+2.f dB")

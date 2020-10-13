import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def make_spectrogram(pdg, df):
    results = np.empty(df.shape[1])
    for column in df.columns:
        freqs, times, spectrogram = signal.spectrogram(df[column].values)
        if pdg:
            print(freqs)
            print(times)
            print(spectrogram)
            plt.figure(figsize=(5, 4))
            plt.imshow(spectrogram, aspect='auto', cmap='hot_r', origin='lower')
            plt.title('Spectrogram of column {}'.format(column))
            plt.ylabel('Frequency band')
            plt.xlabel('Time window')
            plt.tight_layout()
            plt.show(block=False)


def make_welch_psd(pdg, df):
    fs = 4.0
    window='hann'
    results = np.empty(df.shape[1])
    for column in df.columns:
        freqs, psd = signal.welch(df[column].values)
        plt.figure(figsize=(5, 4))
        plt.semilogx(freqs, psd)
        plt.title('PSD: power spectral density of column {}'.format(column))
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        plt.tight_layout()
        plt.show(block=False)
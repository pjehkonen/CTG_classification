import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Create np array with each row presenting a windowing vector length v_len
def make_windows(v_len):
    win_name = ['blackman', 'hamming','hann','flattop', 'blackmanharris', 'tukey']
    windows = []

    for window in win_name[:-1]:
        windows.append(signal.get_window(window, v_len))

    windows.append(signal.windows.tukey(v_len, alpha=0.1))

    return np.array(windows), win_name


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
    v_len = df.shape[0]

    windows, win_names = make_windows(v_len)

    print(windows)
    print(win_names)
    results = np.empty(df.shape[1])
    for column in df.columns:
        freqs, psd = signal.welch(df[column].values, fs=fs, )
        plt.figure(figsize=(5, 4))
        plt.semilogx(freqs, psd)
        plt.title('PSD: power spectral density of column {}'.format(column))
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        plt.tight_layout()
        plt.show(block=False)
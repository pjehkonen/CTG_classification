import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy import signal

def plot_stuff(window, title, amplitude, sample):

    plt.plot(window)
    plt.title(title)
    plt.ylabel(amplitude)
    plt.xlabel(str(sample))
    plt.figure()
    A = fft(window, sample) / (len(window) / 2.0)
    freq = np.linspace(-0.5, 0.5, len(A))
    response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    plt.plot(freq, response)
    plt.axis([-0.5, 0.5, -120, 0])
    plt.title("Frequency response of the {} window".format(title))
    plt.ylabel("Normalized magnitude [dB]")
    plt.xlabel("Normalized frequency [cycles per sample]")

def make_demo(df):
    n_samples = df.shape[0]
    b_win = ['blackman']
    window = signal.get_window(b_win[0], n_samples)
    plot_stuff(b_win[0], b_win[0], "amplitude", n_samples)
    print("hui")
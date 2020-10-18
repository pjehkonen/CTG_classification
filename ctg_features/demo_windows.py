import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy import signal

def plot_stuff(window, title, amplitude, sample):

    plt.plot(window)
    plt.title(title)
    plt.ylabel(amplitude)
    plt.xlabel(str(sample))
    plt.show(block=False)

    plt.figure()
    A = fft(window, sample) / (len(window) / 2.0)
    freq = np.linspace(-0.5, 0.5, len(A))
    response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    plt.plot(freq, response)
    plt.axis([-0.5, 0.5, -120, 0])
    plt.title("Frequency response of the {} window".format(title))
    plt.ylabel("Normalized magnitude [dB]")
    plt.xlabel("Normalized frequency [cycles per sample]")
    plt.show(block=False)

def make_demo(df):
    n_samples = df.shape[0]
    b_win = ['blackman', 'hamming','hann','flattop', 'blackmanharris']
    for win in b_win:
        window = signal.get_window(win, n_samples)
        #plot_stuff(window, win, "amplitude", n_samples)

    k_betas = [0.0, 5.0, 6.0, 8.6]
    for beta in k_betas:
        window = signal.kaiser(n_samples, beta)
        #plot_stuff(window, 'kaiser beta {}'.format(beta), "amplitude", n_samples)

    deviations = [1., 2., 3., 4., 5., 6., 7.]
    for deviation in deviations:
        window = signal.gaussian(n_samples, std=deviation)
        #plot_stuff(window, 'gaussian sigma {}'.format(deviation), "amplitude", n_samples)

    for alpha in np.arange(0.1,1.0,0.1):
        window = signal.tukey(n_samples, alpha=alpha)
        #plot_stuff(window, 'tuckey alpha {}'.format(alpha), "amplitude", n_samples)

    for attenuation in np.arange(50, 130, 10):
        window = signal.chebwin(n_samples, at=attenuation)
        plot_stuff(window, 'Dolph-Cheby at={}'.format(attenuation), "amplitude",n_samples)


    print("hui")
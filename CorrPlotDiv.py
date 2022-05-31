import sklearn
import numpy as np
import pandas as pd
from pydub import AudioSegment
import soundfile as sf
from scipy import signal
from scipy.io import wavfile
from matplotlib import pyplot as plt
from numpy import sin, linspace, pi
from pylab import plot, show, title, xlabel, ylabel, subplot
from numpy import arange
import seaborn as sns
import librosa


def plot_correlation(src_name, tgt_name):

    #SOURCE_FILE, sampling_rate = sf.read(src_name)
    #TARGET_FILE, sampling_rate2 = sf.read("../../egs/bss-example/mnmf/Outputs/" + tgt_name)
    SOURCE_FILE, sampling_rate = librosa.load("Resources/"+src_name, sr=8000)  # Downsample 44.1kHz to 8kHz
    TARGET_FILE, sampling_rate2 = librosa.load("Resources/Outputs/" + tgt_name, sr=8000)  # Downsample 44.1kHz to 8kHz
    y = pd.Series(SOURCE_FILE)
    x = pd.Series(TARGET_FILE)
    correlation = y.corr(x)
    print(correlation)

    # plt.suptitle('String 4 Source Correlation')
    # plt.scatter(x, y)
    # plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='red')
    # plt.xlabel('x axis')
    # plt.ylabel('y axis')
    # plt.title("correlation: " + str(correlation))
    # # plt.xlim([4, 6])
    # # plt.ylim([-4e8, 8e8])
    # plt.show()
    X = mix_sources([SOURCE_FILE, TARGET_FILE])
    fig = plt.figure()
    plt.subplot(3, 1, 1)
    for s in [SOURCE_FILE]:
        plt.plot(s)
    plt.title("Time Domain Envelope for Source File")
    plt.tight_layout()
    plt.subplot(3, 1, 2)
    for s in [TARGET_FILE]:
        plt.plot(s, color="orangered")
    plt.title("Time Domain Envelope for Target File")
    plt.subplot(3, 1, 3)
    plt.suptitle('Correlation for ' + src_name[:-4] + ' from ' + tgt_name[:-4])
    plt.scatter(x, y)
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='red')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title("correlation: " + str(correlation))
    plt.tight_layout()
    plt.savefig('Resources/Final_Corr_Plots/Time/'+src_name[:-4] + ' from ' + tgt_name[:-4]+'.png')
    plt.close()
    #plt.show()
    plt.subplot(2, 1, 1)
    plt.suptitle("Frequency Domain Comparision of " + src_name[:-4] + ' and ' + tgt_name[:-4])
    fft_spectrum = np.fft.rfft(SOURCE_FILE)
    freq = np.fft.rfftfreq(SOURCE_FILE.size, d=1 / sampling_rate)
    fft_spectrum_abs = np.abs(fft_spectrum)
    plt.plot(freq, fft_spectrum_abs)
    plt.xlim([0, 5000])
    plt.xlabel("frequency, Hz")
    plt.ylabel("Amplitude, units")
    plt.title("Source File")
    plt.subplot(2, 1, 2)
    fft_spectrum = np.fft.rfft(TARGET_FILE)
    freq = np.fft.rfftfreq(TARGET_FILE.size, d=1. / sampling_rate2)
    fft_spectrum_abs = np.abs(fft_spectrum)
    plt.plot(freq, fft_spectrum_abs, color='orangered', )
    plt.xlim([0, 5000])
    plt.title("Target File")
    plt.xlabel("frequency, Hz")
    plt.ylabel("Amplitude, units")
    plt.tight_layout()
    plt.savefig('Resources/Final_Corr_Plots/Frequency/' + src_name[:-4] + ' from ' + tgt_name[:-4] + '.png')
    #plt.show()
    plt.close()

def mix_sources(mixtures, apply_noise=False):
    for i in range(len(mixtures)):

        max_val = np.max(mixtures[i])

        if max_val > 1 or np.min(mixtures[i]) < 1:
            mixtures[i] = mixtures[i] / (max_val / 2) - 0.5

    X = np.c_[[mix for mix in mixtures]]

    if apply_noise:
        X += 0.02 * np.random.normal(size=X.shape)

    return X

if __name__ == '__main__':
    src = "Bss_SomeoneLikeYou_piano.wav"
    tgt = "split_Bss_SomeoneLikeYou_Mix1.wav0.wav"
    plot_correlation(src, tgt)
    src = "Bss_SomeoneLikeYou_Voice.wav"
    tgt = "split_Bss_SomeoneLikeYou_Mix1.wav1.wav"
    plot_correlation(src, tgt)
    src = "Bss_numb_Vocal.wav"
    tgt = "split_Bss_numb_mix1.wav0.wav"
    plot_correlation(src, tgt)
    src = "Bss_numb_piano.wav"
    tgt = "split_Bss_numb_mix1.wav1.wav"
    plot_correlation(src, tgt)
    src = "StrTru1stringString.wav"
    tgt = "StrTrusplit2.wav"
    plot_correlation(src, tgt)
    src = "StrTru1stringTrumpet.wav"
    tgt = "StrTrusplit1.wav"
    plot_correlation(src, tgt)
    src = "StrTruLapBassBass.wav"
    tgt = "split2sig1.wav"
    plot_correlation(src, tgt)
    src = "StrTruLapBassplucks.wav"
    tgt = "split2sig2.wav"
    plot_correlation(src, tgt)
    src = "StrTru1stringString.wav"
    tgt = "split2sig4.wav"
    plot_correlation(src, tgt)
    src = "StrTru1stringTrumpet.wav"
    tgt = "split2sig3.wav"
    plot_correlation(src, tgt)
    src = "source2.wav"
    tgt = "out2.wav"
    plot_correlation(src, tgt)
    src = "source1.wav"
    tgt = "out1.wav"
    plot_correlation(src, tgt)

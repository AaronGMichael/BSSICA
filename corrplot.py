import sklearn
import numpy as np
import pandas as pd
from pydub import AudioSegment
import soundfile as sf
from scipy import signal
from scipy.io import wavfile
from two_channelip import mix_sources
from matplotlib import pyplot as plt
from numpy import sin, linspace, pi
from pylab import plot, show, title, xlabel, ylabel, subplot
from numpy import arange
import seaborn as sns


def corrplot():
    #sampling_rate, SOURCE_FILE = wavfile.read('Resources/source2.wav')
    # SOURCE_FILE = AudioSegment.from_file(file='Resources/source2.wav', format='wav')
    #sampling_rate2, TARGET_FILE = wavfile.read('Resources/Outputs/out2.wav')
    # TARGET_FILE = AudioSegment.from_file(file='Resources/Outputs/out2.wav', format='wav')
    src = "Bss_numb_Vocal.wav"
    SOURCE_FILE, sampling_rate = sf.read("Resources/" + src)
    TARGET_FILE, sampling_rate2 = sf.read("Resources/Outputs/split_Bss_numb_mix1.wav0.wav")
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
    plt.subplot(4, 1, 1)
    plt.suptitle('Correlation for ' + src)
    plt.scatter(x, y)
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='red')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title("correlation: " + str(correlation))
    plt.subplot(4, 1, 2)
    for s in X:
        plt.plot(s, alpha=0.8)
    plt.title("Time Domain Envelope")
    plt.subplot(4, 1, 3)
    fft_spectrum = np.fft.rfft(SOURCE_FILE)
    freq = np.fft.rfftfreq(SOURCE_FILE.size, d=1 / sampling_rate)
    fft_spectrum_abs = np.abs(fft_spectrum)
    plt.plot(freq, fft_spectrum_abs, color='orangered')
    plt.xlim([0, 5000])
    plt.xlabel("frequency, Hz")
    plt.ylabel("Amplitude, units")
    plt.subplot(4, 1, 4)
    fft_spectrum = np.fft.rfft(TARGET_FILE)
    freq = np.fft.rfftfreq(TARGET_FILE.size, d=1. / sampling_rate2)
    fft_spectrum_abs = np.abs(fft_spectrum)
    plt.plot(freq, fft_spectrum_abs, color='orangered',)
    plt.xlim([0, 5000])
    plt.xlabel("frequency, Hz")
    plt.ylabel("Amplitude, units")
    fig.tight_layout()
    plt.show()

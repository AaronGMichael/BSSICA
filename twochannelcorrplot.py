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


def dual_chnl_corrplot():
    stereo_audio_source_temp = AudioSegment.from_file("Resources/Outputs/Tempfiles/Mix_Hal_Drums_Stem.wav", format="wav")
    startTime = 10
    endTime = 18 * 1000
    stereo_audio_source = stereo_audio_source_temp[startTime:endTime]
    stereo_audio_target_temp = AudioSegment.from_file("Resources/Outputs/Tempfiles/aaron_output.wav", format="wav")
    stereo_audio_target = stereo_audio_target_temp[startTime:endTime]
    mono_audios_source = stereo_audio_source.split_to_mono()
    mono_left_source = mono_audios_source[0].export("Resources/temp/mono_left_source.wav", format="wav")
    mono_right_source = mono_audios_source[1].export("Resources/temp/mono_right_source.wav", format="wav")
    mono_audios_target = stereo_audio_target.split_to_mono()
    mono_left_target = mono_audios_target[0].export("Resources/temp/mono_left_target.wav", format="wav")
    mono_right_target = mono_audios_target[0].export("Resources/temp/mono_right_target.wav", format="wav")
    SOURCE_FILE1, sampling_rate = sf.read("Resources/temp/mono_left_source.wav")
    TARGET_FILE1, sampling_rate2 = sf.read("Resources/temp/mono_left_target.wav")
    for _ in TARGET_FILE1.len() - SOURCE_FILE1.len():
        SOURCE_FILE1.append(0)

    #SOURCE_FILE1 = SOURCE_FILE1_temp[0, len(TARGET_FILE1) - 1]
    plot_corr_with_freq(SOURCE_FILE1, TARGET_FILE1, sampling_rate, sampling_rate2, 0)
    SOURCE_FILE2, sampling_rate = sf.read("Resources/temp/mono_right_source.wav")
    TARGET_FILE2, sampling_rate2 = sf.read("Resources/temp/mono_right_target.wav")
    plot_corr_with_freq(SOURCE_FILE2, TARGET_FILE2, sampling_rate, sampling_rate2, 1)

def plot_corr_with_freq(SOURCE_FILE, TARGET_FILE, sampling_rate, sampling_rate2, a):
    y = pd.Series(SOURCE_FILE)
    x = pd.Series(TARGET_FILE)
    correlation = y.corr(x)
    print(correlation)
    X = mix_sources([SOURCE_FILE, TARGET_FILE])
    fig = plt.figure()
    plt.subplot(4, 1, 1)
    if a==0:
        plt.suptitle('Left Channel Correlation')
    if a==1:
        plt.suptitle('Right Channel Correlation')
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
    plt.xlim([300, 700])
    plt.ylim(0, 200)
    plt.xlabel("frequency, Hz")
    plt.ylabel("Amplitude, units")
    plt.subplot(4, 1, 4)
    fft_spectrum = np.fft.rfft(TARGET_FILE)
    freq = np.fft.rfftfreq(TARGET_FILE.size, d=1. / sampling_rate2)
    fft_spectrum_abs = np.abs(fft_spectrum)
    plt.plot(freq, fft_spectrum_abs, color='orangered',)
    plt.xlim([300, 700])
    plt.ylim(0, 200)
    plt.xlabel("frequency, Hz")
    plt.ylabel("Amplitude, units")
    fig.tight_layout()
    plt.show()

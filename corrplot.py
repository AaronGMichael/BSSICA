import sklearn
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile
from matplotlib import pyplot as plt
import seaborn as sns
from pydub import AudioSegment


def corrplot():
    sampling_rate, SOURCE_FILE = wavfile.read('Resources/temp/mono_left.wav')
    sampling_rate, TARGET_FILE = wavfile.read('Resources/Outputs/split2.wav')
    y = pd.Series(SOURCE_FILE)
    x = pd.Series(TARGET_FILE)
    correlation = y.corr(x)
    print(correlation)
    plt.suptitle('High freq. Part Correlation')
    plt.scatter(x, y)
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='red')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title("correlation: " + str(correlation))
    # plt.xlim([4, 6])
    # plt.ylim([-4e8, 8e8])
    plt.show()

import numpy as np

np.random.seed(0)
from scipy import signal
from scipy.io import wavfile
from matplotlib import pyplot as plt
import seaborn as sns
from pydub import AudioSegment

sns.set(rc={'figure.figsize': (11.7, 8.27)})


def g(x):
    return np.tanh(x)


def g_der(x):
    return 1 - g(x) * g(x)


def center(X):
    X = np.array(X)

    mean = X.mean(axis=1, keepdims=True)

    return X - mean


def whitening(X):
    cov = np.cov(X)
    d, E = np.linalg.eigh(cov)
    D = np.diag(d)
    D_inv = np.sqrt(np.linalg.inv(D))
    X_whiten = np.dot(E, np.dot(D_inv, np.dot(E.T, X)))
    return X_whiten


def calculate_new_w(w, X):
    w_new = (X * g(np.dot(w.T, X))).mean(axis=1) - g_der(np.dot(w.T, X)).mean() * w
    w_new /= np.sqrt((w_new ** 2).sum())
    return w_new


def ica(X, iterations, tolerance=1e-5):
    X = center(X)

    X = whitening(X)

    components_nr = X.shape[0]
    W = np.zeros((components_nr, components_nr), dtype=X.dtype)
    for i in range(components_nr):

        w = np.random.rand(components_nr)

        for j in range(iterations):

            w_new = calculate_new_w(w, X)

            if i >= 1:
                w_new -= np.dot(np.dot(w_new, W[:i].T), W[:i])

            distance = np.abs(np.abs((w * w_new).sum()) - 1)

            w = w_new

            if distance < tolerance:
                break

        W[i, :] = w
        S = np.dot(W, X)

    return S


def plot_mixture_sources_predictions(X, original_sources, S):
    fig = plt.figure()
    plt.subplot(3, 1, 1)
    for x in X:
        plt.plot(x)
    plt.title("mixtures")
    plt.subplot(3, 1, 2)
    for s in original_sources:
        plt.plot(s)
    plt.title("real sources")
    plt.subplot(3, 1, 3)
    for s in S:
        plt.plot(s)
    plt.title("predicted sources")

    fig.tight_layout()
    plt.show()


def mix_sources(mixtures, apply_noise=False):
    for i in range(len(mixtures)):

        max_val = np.max(mixtures[i])

        if max_val > 1 or np.min(mixtures[i]) < 1:
            mixtures[i] = mixtures[i] / (max_val / 2) - 0.5

    X = np.c_[[mix for mix in mixtures]]

    if apply_noise:
        X += 0.02 * np.random.normal(size=X.shape)

    return X

def splitsong2sig():
    stereo_audio1 = AudioSegment.from_file("Resources/StrTruLapBassMix1.wav", format="wav")
    stereo_audio2 = AudioSegment.from_file("Resources/StrTruLapBassMix2.wav", format="wav")
    # Calling the split_to_mono method
    # on the stereo audio file
    mono_audios1 = stereo_audio1.split_to_mono()
    mono_audios2 = stereo_audio2.split_to_mono()
    # Exporting/Saving the two mono
    # audio files present at index 0(left)
    # and index 1(right) of list returned
    # by split_to_mono method
    mono_left1 = mono_audios1[0].export("Resources/temp/mono_left1.wav", format="wav")
    mono_right1 = mono_audios1[1].export("Resources/temp/mono_right1.wav", format="wav")
    mono_left2 = mono_audios2[0].export("Resources/temp/mono_left2.wav", format="wav")
    mono_right2 = mono_audios2[1].export("Resources/temp/mono_right2.wav", format="wav")
    sampling_rate, mix1 = wavfile.read('Resources/temp/mono_left1.wav')
    sampling_rate, mix2 = wavfile.read('Resources/temp/mono_right1.wav')
    sampling_rate, mix3 = wavfile.read('Resources/temp/mono_left2.wav')
    sampling_rate, mix4 = wavfile.read('Resources/temp/mono_right2.wav')
    # sampling_rate, source1 = wavfile.read('source1.wav')
    # sampling_rate, source2 = wavfile.read('source2.wav')
    X = mix_sources([mix1, mix2, mix3, mix4])
    S = ica(X, iterations=1000)
    #plot_mixture_sources_predictions(X, [mono_left, mono_right], S)
    wavfile.write('Resources/Outputs/split2sig1.wav', sampling_rate, S[0])
    wavfile.write('Resources/Outputs/split2sig2.wav', sampling_rate, S[1])
    wavfile.write('Resources/Outputs/split2sig3.wav', sampling_rate, S[2])
    wavfile.write('Resources/Outputs/split2sig4.wav', sampling_rate, S[3])
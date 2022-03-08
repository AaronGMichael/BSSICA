# compare.py
import argparse
import scipy
from correlatefiles import correlation
from scipy.io import wavfile

def initialize():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-i ", "--source_file", help="Resources/source1.wav")
    # parser.add_argument("-o ", "--target_file", help="Resources/source1.wav")
    # args = parser.parse_args()

    sampling_rate, SOURCE_FILE = wavfile.read('Resources/StrTru1stringTrumpet.wav')
    sampling_rate, TARGET_FILE = wavfile.read('Resources/Outputs/StrTrusplit1.wav')
    # if not SOURCE_FILE or not TARGET_FILE:
    #     raise Exception("Source or Target files not specified.")
    return SOURCE_FILE, TARGET_FILE


def correlationplotgen():
    SOURCE_FILE, TARGET_FILE = initialize()
    print("Covariance of Sound files  ", correlation(SOURCE_FILE, TARGET_FILE))
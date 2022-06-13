import pyaudio
import wave
from pynput.keyboard import Key
from pynput import keyboard
import json
import numpy as np
from numba import jit
import pandas as pd
from scipy.io import wavfile


class Data:

    def __init__(self):
        self.frames = []
        self.amplitude = []
        self.pairs = []
        self.lag = 6500  # this parameter need to be manually set !!
        self.listener = None
        # recording parameters
        self.CHUNK = 1
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.RECORD_SECONDS = 60 * 1  # length of recordingffaa in seconds ############################################
        # self.json_output = "../data/data.json"
        # self.WAVE_OUTPUT_FILENAME = "../data/output_new.wav"

        self.p = None
        self.stream = None

        # helper parameters for private use
        self.clicks = []  # peaks indices
        self.letters = []  # letters typed in the same order as the clicks

    def on_press(self, key):
        print(key)
        if key != Key.space and key != Key.backspace:
            self.pairs.append((key.char, len(self.frames)))
        elif key == Key.space:
            self.pairs.append(("space", len(self.frames)))

    @staticmethod
    def on_release(key):
        if key == Key.esc:
            # Stop listener
            return False

    def record(self):
        self.p = pyaudio.PyAudio()
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  frames_per_buffer=self.CHUNK)
        print("* recording")
        for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            data = self.stream.read(self.CHUNK)
            self.frames.append(data)
        # transforms 16 bit data in integer data
        self.amplitude = np.array([np.frombuffer(i, np.int16) for i in self.frames])
        # normalizes the data
        self.amplitude = [j[0] for j in self.amplitude]
        self.amplitude = self.amplitude / np.amax(self.amplitude)
        for i in self.pairs:
            if i[1] + self.lag < len(self.amplitude):
                self.clicks.append(i[1] + self.lag)
                self.letters.append(i[0])

        self.clicks = np.array(self.clicks)
        print("* done recording")
        self.listener.stop()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def save_wav(self, file_path='../data/output.wav', mode='wb'):
        wf = wave.open(file_path, mode)
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()

    def load_wav(self, file_path):
        """
        load and append the data
        :param file_path:
        :return:
        """
        samplerate, data = wavfile.read(file_path)
        if self.amplitude is None:
            self.amplitude = []
        if len(data.shape) > 1:
            self.amplitude = list(self.amplitude) + data[0]
        else:
            self.amplitude = list(self.amplitude) + data

    def save_json(self, file_path="../data/output.json", mode='w'):
        peaks_from_clicks = get_peaks_from_click(self.amplitude, self.clicks, k=4000)
        json_data = dict()
        json_data["data"] = list(map(float, self.amplitude))
        json_data["peaks"] = list(map(int, peaks_from_clicks))
        json_data["letters"] = list(map(str, self.letters))

        with open(file_path, mode) as file:
            json.dump(json_data, file, indent=2)

    def load_json(self, file_path):
        """
        load and append the data
        :param file_path:
        :return:
        """
        with open(file_path, 'r') as file:
            json_data = json.load(file)
            if self.amplitude is None:
                self.amplitude = []
            if self.letters is None:
                self.letters = []
            if self.clicks is None:
                self.clicks = []
            offset = len(self.amplitude)
            self.amplitude = list(self.amplitude) + json_data["data"]
            self.letters = list(self.letters) + json_data["letters"]
            self.clicks = list(self.clicks) + [p + offset for p in json_data["peaks"]]

    @jit(forceobj=True)
    def get_DataFrame(self, left=300, right=700):
        peaks_from_clicks = get_peaks_from_click(self.amplitude, self.clicks, k=4000)
        samples = get_samples(np.array(self.amplitude), np.array(peaks_from_clicks), left=left, right=right)
        data_matrix = [list(samples[i]) + [self.letters[i]] for i in range(len(samples))]
        data_frame = pd.DataFrame(data_matrix)
        return data_frame


###############################################################
# Helper Functions
###############################################################
def get_peaks(liste, k=100, alpha=2, min_d=150, threshold=1e-3):
    l = np.abs(liste[:])
    peaks = []
    d = 0
    i = 0
    while i < len(l):
        M = np.mean(l[max(i - k, 0):i + k])

        if d < 0 and l[i] > threshold and l[i] > M * alpha:
            am = np.argmax(l[i:i + k])
            peaks.append(i + am)
            i = i + am
            d = min_d
        d -= 1
        i += 1
    return peaks


def get_energy(data):
    fs = 44100
    fft_size = 256
    overlap_fac = 0.5

    hop_size = np.int32(np.floor(fft_size * (1 - overlap_fac)))
    pad_end_size = fft_size  # the last segment can overlap the end of the data array by no more than one window size
    total_segments = np.int32(np.ceil(len(data) / np.float32(hop_size)))
    t_max = len(data) / np.float32(fs)

    window = np.hanning(fft_size)  # our half cosine window
    inner_pad = np.zeros(fft_size)  # the zeros which will be used to double each segment size

    proc = np.concatenate((data, np.zeros(pad_end_size)))  # the data to process
    result = np.empty(total_segments, dtype=np.float32)  # space to hold the result

    for i in range(total_segments):  # for each segment
        current_hop = hop_size * i  # figure out the current segment offset
        segment = proc[current_hop:current_hop + fft_size]  # get the current segment
        windowed = segment * window  # multiply by the half cosine function
        padded = np.append(windowed, inner_pad)  # add 0s to double the length of the data
        spectrum = np.fft.fft(padded) / fft_size  # take the Fourier Transform and scale by the number of samples
        autopower = np.abs(spectrum * np.conj(spectrum))  # find the autopower spectrum
        result[i] = sum(autopower)  # append to the results array

    print("shape= ", result.shape)
    peaks = get_peaks(result, k=20, alpha=4, min_d=80)
    return result, peaks


def get_peaks_from_click(data, clicks, k=1000):
    """
    :param data: list containing the audio data
    :param clicks: the list containing clicks indices
    :param k: the range [-k,k] to look for the peak around each click
    :return: array of indices of peaks
    """
    result = []
    for click in clicks:
        result.append(click - k + np.argmax(data[click - k:click + k]))
    return result


@jit(nopython=True)  # jit for faster python code
def get_samples(liste, peaks_list, left=300, right=700):
    """
    :param liste: the raw data as numpy array !!!
    :param peaks_list: a numpy array containing peaks indices
    :param left: how many values to take left of the peak indice
    :param right: how many values to take right of the peak indice
    :return: a list were each element represent [left-peak_indice: right+peak_indice] with padding equals 0
    """
    result = []
    for p in peaks_list:
        l = liste[max(p - left, 0):p + right]
        if len(l) < left + right:
            l = ([0.0] * (left + right - len(l))).extend(l)
        result.append(l)
    return result

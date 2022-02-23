#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from __future__ import generator_stop

import librosa
import numpy as np
import random
import scipy.signal as sig
import time

from multiprocessing import Process, Queue

from config import *

# Spectral inversion for FIR filters
def spectinvert(taps):
    l = len(taps)
    return ([0]*(l//2) + [1] + [0]*(l//2)) - taps

# Returns the dit length in secods from the WPM
def wpm2dit(wpm):
    return 1.2 / wpm

# The length of a dit. Deviation is in percent of dit length
def dit_len(wpm, deviation):
    dl = wpm2dit(wpm)
    return int(random.normalvariate(dl, dl * deviation) * FRAMERATE)

# The length of a dah
def dah_len(wpm, deviation, dahw = 3.0):
    return int(dahw * dit_len(wpm, deviation))

# The length of pause between dits and dahs inside a character
def symspace_len(wpm, deviation, symw = 1.0):
    return int(symw * dit_len(wpm, deviation))

# The length of pause between characters inside a word
def chrspace_len(wpm, deviation, chrw = 3.0):
    return int(chrw * dit_len(wpm, deviation))

# The length of a space between two words
def wordspace_len(wpm, deviation, wsw = 6.0):
    return int(wsw * dit_len(wpm, deviation))

# Generates <frames> length of white noise 
def whitenoise(frames, vol):
    return np.random.normal(0, vol, frames)

# Generates <frames> length of popping noise
def impulsenoise(frames, th):
    r = np.random.normal(0.0, 1.0, frames)
    r[r < th] = 0.0
    i = r >= th
    r[r >= th] = 1.0
    ret = sig.convolve(r, [1.0] * 10 + [-1.0] * 10, mode='same')
    return ret

# Generates a sequence that when multiplied with the signal, it will cause fading
def qsb(frames, vol, f):
    return 1.0 - np.sin(np.linspace(0, 2 * np.pi * frames / FRAMERATE * f, frames)) * vol

# Returns a random morse character
def get_next_character():
    return random.choice(MORSE_CHR[0:-1] + [' '] * 2)

# Returns: ([(1/0, duration), ...], total length)
def get_onoff_data(c, wpm, deviation):
    pairs = []
    length = 0
    if c == ' ':
        pairs.append((0.0, wordspace_len(wpm, deviation)))
        length += pairs[-1][1]
    else:
        last_symspace_len = 0
        for sym in CHARS[c]:
            pairs.append((1.0, dit_len(wpm, deviation) if sym == '.' else dah_len(wpm, deviation)))
            length += pairs[-1][1]
            pairs.append((0.0, symspace_len(wpm, deviation)))
            length += pairs[-1][1]
        length -= pairs[-1][1]
        pairs[-1] = (0.0, (chrspace_len(wpm, deviation)))
        length += pairs[-1][1]
    
    return (pairs, length)

def generate_seq(seq_length, framerate=FRAMERATE):
    # Words per minute
    wpm       = random.uniform(5,  25.0)
    # Error in timing
    deviation = random.uniform(0.0,  0.1)
    # Signal volume
    sigvol    = random.uniform(0.1,  4.0)
    # White noise volume
    wnvol     = random.uniform(0.05 * sigvol,  0.5 * sigvol)
    # QSB volume: 0=no qsb, 1: full silencing QSB
    qsbvol    = random.uniform(0.01,  0.7)
    # QSB frequency in Hertz
    qsbf      = random.uniform(0.1,  1.0)
    # Signal frequency
    sigf      = random.uniform(500.0, 800.0)
    # Signal phase
    phase     = random.uniform(0.0,  framerate / sigf)
    # Filter lower cutoff
    f1        = random.uniform(sigf - 400, sigf - 100)
    # Filter higher cutoff
    f2        = random.uniform(sigf + 100, sigf + 400)
    # Number of taps in the filter
    taps      = 63 # The number of taps of the FIR filter

    audio = np.zeros(seq_length, dtype=np.float64)
    characters = []

    padl = int(max(0, random.normalvariate(1, 0.5)) * framerate) # Padding at the beginning
    i = padl # The actual index in the samlpes
    s_pairs, s_length = get_onoff_data(' ', wpm, deviation)
    c = ' '
    while True:
        prev_c = c
        c = get_next_character()
        # Generate a character,
        # but not space at the beginning, or repeating spaces
        while prev_c == ' ' and c == ' ':
            c = get_next_character()

        # Get the audio samples for this character
        # TODO: for fuck's sake rename this
        pairs, length = get_onoff_data(c, wpm, deviation)

        # Check if it's too long to fit
        # Leave some space on the right
        # Always insert a space after the sequence
        if i + length + s_length + 1000 >= seq_length:
            for p in s_pairs:
                audio[i:i+p[1]] = p[0]
                i += p[1]
            characters.append((' ', i / float(framerate)))
            break

        # Write it into the audio data array
        for p in pairs:
            audio[i:i+p[1]] = p[0]
            i += p[1]
        characters.append((c, i / float(framerate)))

    #print(characters)
    #characters = [c for c in characters if c[0] != MORSE_CHR[0]]
    #characters = characters[:2]
    #print(characters)

    #characters = [characters[0]] if len(characters) == 1 else [characters[0], characters[1]]

    #characters = [e for e in characters if e]
    #print(characters)

    # Set up the bandpass filter
    fil_lowpass = sig.firwin(taps, f1/(framerate/2))
    fil_highpass = spectinvert(sig.firwin(taps, f2/(framerate/2)))
    fil_bandreject = fil_lowpass+fil_highpass
    fil_bandpass = spectinvert(fil_bandreject)

    # Remove clicks
    s = sig.convolve(audio, np.array(range(80, 0, -1)) / 3240.0, mode='same') * sigvol
    # Sinewave with phase shift (cw signal)
    s *= np.sin((np.arange(0, seq_length) + phase) * sigf * 2 * np.pi / framerate)
    # QSB
    s *= qsb(seq_length, qsbvol, qsbf)
    # Add white noise
    s += whitenoise(seq_length, wnvol)
    # Add impulse noise
    s += impulsenoise(seq_length, 4.2)
    # Filter signal
    s = sig.lfilter(fil_bandpass, 1.0, s)
    # AGC with fast attack and slow exponential decay
    #a = 0.02  # Attack. The closer to 0 the slower.
    #d = 0.002 # Decay. The closer to 0 the slower.
    #agc_coeff = 1.0   # Correction factor 
    #for k in range(len(s)):
    #    s[k] *= agc_coeff
    #    err = s[k]**2 - 1.0
    #    if err >= 0:
    #        # Level is too high
    #        agc_coeff -= abs(err * a)
    #    else:
    #        # Level is too low
    #        agc_coeff += abs(err * d)
    #s *= 1.56

    s /= np.sqrt(np.average(s**2))
    #print np.average(s), np.average(s**2), max(s), min(s) # Debug mean, variance, max, min
    #exit()
    # TODO: QRN
    # Scale and convert to int
    s = (s * 2**12).astype(np.int16)

    return s, characters


class DataGenerator:
    def __init__(self):
        self.processes = []
        self.queue = Queue(10)
        self.threads_count = 4

    # A generator yielding an audio array, and indices and lables for
    # building a sparsetensor describing labels for CTC functions
    def seq_generator(self, seq_length, framerate, chunk, sr, mel_count):
        def do_work():
            while True:
                audio, labels = generate_seq(seq_length, framerate)
                audio = np.reshape(audio, (seq_length // chunk, chunk))
                # audio = (audio - np.mean(audio)) / np.std(audio) # Normalization
                audio = audio.astype(np.float32)
                mel = self.get_wave_mel_features(audio, sr, mel_count)
                labels = np.asarray([MORSE_CHR.index(l[0]) for l in labels])

                while self.queue.full():
                    time.sleep(0.0100)

                self.queue.put((audio, labels, mel))

        self.start_threads(do_work)

        while True:
            audio, labels, mel = self.queue.get()

            label_indices = []
            label_values = []
            for i, value in enumerate(labels):
                label_indices.append([i])
                label_values.append(value)

            yield (audio, label_indices, label_values, [len(labels)], mel)

    def start_threads(self, do_work):
        if len(self.processes) != self.threads_count:
            for _ in range(self.threads_count):
                p = Process(target=do_work)
                p.daemon = True
                self.processes.append(p)
                p.start()

    @staticmethod
    def get_wave_mel_features(wave, sr, mel_count):
        wave = wave.reshape(SEQ_LENGTH)
        wave = librosa.util.normalize(wave)
        mel = librosa.feature.melspectrogram(wave, sr=sr, n_fft=250, n_mels=mel_count, hop_length=200, power=2)
        mel = mel.T
        mel = mel / np.max(mel)
        # mel = np.round(mel, decimals=4)
        return mel


if __name__ == "__main__":
    import os
    import sys
    import argparse
    import scipy.io.wavfile

    def save_files(dirname, seq_length, batch_size):

        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for i in range(batch_size):
            w = 20
            n = w*i//batch_size
            sys.stdout.write("\r[%s>%s] %4d/%4d  " % ("="*n, " "*(w-n), i, batch_size))
            sys.stdout.flush()
            filename = dirname + f'/{i}.wav'

            audio, characters = generate_seq(seq_length)

            scipy.io.wavfile.write(filename, FRAMERATE, audio)

            with open(dirname + f'/{i}.txt', 'w') as f:
                f.write('\n'.join(map(lambda x: x[0] + ',' + str(x[1]), characters)))

            with open(dirname + '/config.txt', 'w') as f:
                f.write('%d' % seq_length)
        print("")

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        'dirname', metavar='DIRNAME', type=str, default='generated',
        help='name of the directory into wich the output .wav files are to be saved'
    )
    parser.add_argument(
        'batchsize', metavar='BATCHSIZE', type=int,
        help='the number of examples to generate'
    )
    parser.add_argument(
        '--length', metavar='LENGTH', type=int, default=SEQ_LENGTH // FRAMERATE,
        help='the approximate length of the samples in whole seconds'
    )

    args = parser.parse_args()

    save_files(args.dirname, args.length * FRAMERATE, args.batchsize)

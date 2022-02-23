import cProfile
import librosa
import librosa.display
import numpy as np
from config import SEQ_LENGTH, FRAMERATE, CHUNK, FFT_SIZE
import generate_wav_samples as gen
from config import MORSE_CHR
from tqdm import tqdm


sample_len = SEQ_LENGTH

samples_count = 7000
sr = 8000
dict_len = len(MORSE_CHR)
max_seq_len = 5
mel_count = 1
mel_len = 161




def read_data(set_len, g):
    l = np.zeros([set_len, max_seq_len], dtype=np.int32)
    X = np.zeros([set_len, mel_len, mel_count])
    input_length = np.zeros([set_len, 1], dtype=np.int32)
    label_length = np.zeros([set_len, 1], dtype=np.int32)

    i = 0
    for wave, label_indexes, labels, c, mel in tqdm(g):
        if len(labels) > max_seq_len:
            continue

        X[i, :, :] = mel

        l[i, :len(labels)] = labels
        input_length[i, :] = mel.shape[0]

        label_length[i, :1] = c

        i += 1
        if i == set_len:
            break

    return [X, l, input_length, label_length], l


def main():
    dg = gen.DataGenerator()
    g = dg.seq_generator(SEQ_LENGTH, FRAMERATE, 1, sr, mel_count)
    X, l = read_data(samples_count, g)


def prof():
    for i in range(1000):
        audio, labels = gen.generate_seq(SEQ_LENGTH, FRAMERATE)
        audio = np.reshape(audio, (SEQ_LENGTH // 1, 1))
        # audio = (audio - np.mean(audio)) / np.std(audio) # Normalization
        audio = audio.astype(np.float32)
        mel = gen.get_wave_mel_features(audio, sr, mel_count)
        labels = np.asarray([MORSE_CHR.index(l[0]) for l in labels])


if __name__ == '__main__':
    cProfile.run('main()', sort='tottime')

    #main()
import os
import numpy as np
import h5py
import librosa
import config
from pathlib import Path
import utils

# Extracts phonemes from labelled phoneme files.
def process_lab_file(filename):
    lab_f = open(filename)
    phos = lab_f.readlines()
    lab_f.close()
    phonemes = []

    # Populate phonemes with start, end and phonote
    for pho in phos:
        start, end, phonote = pho.split()
        st = int(np.round(float(start) / 0.005804576860324892))
        en = int(np.round(float(end) / 0.005804576860324892))
        if phonote == "pau" or phonote == "br" or phonote == "sil":
            phonote = "Sil"
        phonemes.append([st, en, phonote])

    strings_p = np.zeros((phonemes[-1][1], 1))

    for i in range(len(phonemes)):
        pho = phonemes[i]
        value = config.phonemas_nus.index(pho[2])

        strings_p[pho[0] : pho[1] + 1] = value
    return strings_p


# Applies short time fourier transform to an audio file to extract features.
# Saves the short time fourier transform, extracted features, and phonemes into a .hdf5 file for future use.
def wav_to_hdf5(audio_file, directory, singer, sing=True):
    # Loads audio file. Returns 1. audio as numpy array (float time series) and 2. sampling rate.
    audio, fs = librosa.core.load(
        os.path.join(directory, audio_file), sr=config.SAMPLE_RATE
    )
    # Cast audio numpy array to 64 bits
    audio = np.float64(audio)

    # Flattens a dual-channel array if needed into a single array
    if len(audio.shape) == 2:
        vocals = np.array((audio[:, 1] + audio[:, 0]) / 2)
    else:
        vocals = np.array(audio)

    # Applies short time fourier transform to vocals. 
    voc_stft = abs(utils.stft(vocals))
    # Gets features from vocals using pyworld vocoder.
    out_feats = utils.audio_to_feats(vocals, fs)

    # Extracts phonemes for each frame as created by the fourier transform.
    strings_p = process_lab_file(os.path.join(directory, audio_file[:-4] + ".txt"))
    voc_stft, out_feats, strings_p = utils.match_time([voc_stft, out_feats, strings_p])

    # Saves extracted phonemes, features and short time fourier transform as a .hdf5 file.
    Path(config.voice_dir).mkdir(parents=True, exist_ok=True)
    hdf5_type = "_sing_"
    if not sing:
        hdf5_type = "_read_"        
    hdf5_file = h5py.File(
        config.voice_dir + "nus_" + singer + hdf5_type + audio_file[:-4] + ".hdf5",
        mode="a",
    )
    if not "phonemes" in hdf5_file:
        hdf5_file.create_dataset("phonemes", [voc_stft.shape[0]], int)
    hdf5_file["phonemes"][:, ] = strings_p[:, 0]
    hdf5_file.create_dataset("voc_stft", voc_stft.shape, np.float32)
    hdf5_file.create_dataset("feats", out_feats.shape, np.float32)
    hdf5_file["voc_stft"][:, :] = voc_stft
    hdf5_file["feats"][:, :] = out_feats
    hdf5_file.close()


def main():
    singers = next(os.walk(config.NUS_DIR))[1]

    for singer in singers:
        # Get sing and read directories for each singer
        sing_dir = config.NUS_DIR + singer + "/sing/"
        read_dir = config.NUS_DIR + singer + "/read/"

        # Processes singing samples
        sing_wav_files = [
            x
            for x in os.listdir(sing_dir)
            if x.endswith(".wav") and not x.startswith(".")
        ]
        print("Processing singer %s" % singer)
        for wav in sing_wav_files:
            print("Processing sing file %s" % wav)
            wav_to_hdf5(wav, sing_dir, singer)

        # Processes reading samples
        read_wav_files = [
            x
            for x in os.listdir(read_dir)
            if x.endswith(".wav") and not x.startswith(".")
        ]
        print("Processing reader %s" % singer)
        for wav in read_wav_files:
            print("Processing read file %s" % wav)
            wav_to_hdf5(wav, read_dir, singer, sing=False)


if __name__ == "__main__":
    main()
import soundfile as sf
import numpy as np
import pyworld as pw
import matplotlib.pyplot as plt
from conversions import (
    spectrogram_to_mel_freq_spectral_coefficent,
    spectrogram_to_mel_generalized_cepstral,
    mel_generalized_cepstral_to_spectrogram,
    mel_freq_spectral_coefficent_to_mel_generalized_cepstral,
)

import config


# Performs short time fourier transform over input audio data.
# STFT is necessary to extract frequency features (e.g. f0) from our data.
def stft(
    data, window=np.hanning(1024), hopsize=256.0, nfft=1024.0
):
    # Add 2 to account for start and end frames.
    num_frames = np.ceil(data.size / np.double(hopsize)) + 2

    # Allocate enough frames, assuming the first frame is centered on first sample
    length = (num_frames - 1) * hopsize + window.size

    # Take the mean over one dimension if data is 2D
    if len(data.shape) > 1:
        data = np.mean(data, axis=-1)
    # Add 0s to beginning of data so that first sample is in center of first window
    data = np.concatenate((np.zeros(int(window.size / 2)), data))

    # Pad data with 0s as allocated earlier in "length"
    # Ensures an exact number of frames
    data = np.concatenate((data, np.zeros(int(length - data.size))))

    # nfft refers to nonequispaced fast Fourier transform, and affects the number of frequencies retrieved
    # nfft must be an even number and preferably a power of 2 (to speed up computation)
    num_freq = nfft / 2 + 1

    # Create array to hold the STFT output.
    STFT = np.zeros([int(num_frames), int(num_freq)], dtype=complex)

    # Apply Fourier tranform on each frame, and store FT of each frame in STFT:
    for n in np.arange(num_frames):
        start_frame = n * hopsize
        end_frame = start_frame + window.size
        frameToProcess = window * data[int(start_frame) : int(end_frame)]
        STFT[int(n), :] = np.fft.rfft(frameToProcess, np.int32(nfft), norm="ortho")

    return STFT


# Fills NaN values and handles indices.
def nan_helper(y):
    return np.isinf(y), lambda z: z.nonzero()[0]


# Extracts features (f0, harm, ap) from short time fourier transform with the pyworld vocoder.
# f0 is the pitch contour of the fundamental frequency of the audio track.
# harm is the harmonic spectral envelope.
# ap is the aperiodic spectral envelope.
def audio_to_feats(vocals, fs, mode=config.comp_mode):
    feats = pw.wav2world(vocals, fs, frame_period=5.80498866)
    ap = feats[2].reshape([feats[1].shape[0], feats[1].shape[1]]).astype(np.float32)
    ap = 10.0 * np.log10(ap**2)
    harm = 10 * np.log10(feats[1].reshape([feats[2].shape[0], feats[2].shape[1]]))
    feats = pw.wav2world(vocals, fs, frame_period=5.80498866)

    f0 = feats[0]
    f0 = hertz_to_midi(f0)

    nans, x = nan_helper(f0)
    naners = np.isinf(f0)
    f0[nans] = np.interp(x(nans), x(~nans), f0[~nans])

    f0 = np.array(f0).reshape([len(f0), 1])
    guy = np.array(naners).reshape([len(f0), 1])
    f0 = np.concatenate((f0, guy), axis=-1)

    if mode == "mfsc":
        harmy = spectrogram_to_mel_freq_spectral_coefficent(harm, 60, 0.45)
        apy = spectrogram_to_mel_freq_spectral_coefficent(ap, 4, 0.45)
    elif mode == "mgc":
        harmy = spectrogram_to_mel_generalized_cepstral(harm, 60, 0.45)
        apy = spectrogram_to_mel_generalized_cepstral(ap, 4, 0.45)

    # Concats features and returns output
    out_feats = np.concatenate((harmy, apy, f0.reshape((-1, 2))), axis=1)

    return out_feats

# Converting from hertz to midi numbers
# Note: a divide by 0 error is common here since f0 might be 0.
# This invalid value is handled by the nan_helper later.
def hertz_to_midi(f0):
    # Midi number 69 represents the equal tempered frequncy 440hz.
    # m  =  12*log2(fm/440 Hz) + 69
    return 69 + 12 * np.log2(f0 / 440)

# Converting from midi numbers to hertz
def midi_to_hertz(f0):
    f0 = f0 - 69
    f0 = f0 / 12
    f0 = 2**f0
    f0 = f0 * 440
    return f0

# Converts features back to audio using the pyworld synthesizer.
def feats_to_audio(in_feats, filename, fs=config.fs, mode=config.comp_mode):
    harm = in_feats[:, :60]
    ap = in_feats[:, 60:-2]
    f0 = in_feats[:, -2:]
    f0[:, 0] = midi_to_hertz(f0[:, 0])

    f0 = f0[:, 0] * (1 - f0[:, 1])

    if mode == "mfsc":
        harm = mel_freq_spectral_coefficent_to_mel_generalized_cepstral(harm)
        ap = mel_freq_spectral_coefficent_to_mel_generalized_cepstral(ap)

    harm = mel_generalized_cepstral_to_spectrogram(harm, 1025, 0.45)
    ap = mel_generalized_cepstral_to_spectrogram(ap, 1025, 0.45)

    harm = 10 ** (harm / 10)
    ap = 10 ** (ap / 20)

    y = pw.synthesize(
        f0.astype("double"),
        harm.astype("double"),
        ap.astype("double"),
        fs,
        config.hoptime,
    )

    sound_file = config.val_dir + filename + ".wav"
    sf.write(sound_file, y, fs)
    print("Sound file saved to", sound_file)


# Generates overlapping windows for the output synthesis
def generate_overlapadd(
    input,
    time_context=config.max_phr_len,
    overlap=config.max_phr_len / 2,
    batch_size=config.batch_size,
):
    input_size = input.shape[-1]

    index = 0
    start = 0
    while (start + time_context) < input.shape[0]:
        index = index + 1
        start = start - overlap + time_context
    batch = (
        np.zeros(
            [int(np.ceil(float(index) / batch_size)), batch_size, time_context, input_size]
        )
        + 1e-10
    )

    index = 0
    start = 0

    while (start + time_context) < input.shape[0]:
        batch[int(index / batch_size), int(index % batch_size), :, :] = input[int(start) : int(start + time_context), :]
        index = index + 1  # index for each block
        start = start - overlap + time_context  # starting point for each block

    return batch, index

# Combine overlapping windows together to complete output synthesis
def overlapadd(batch, num_chunks, overlap=int(config.max_phr_len / 2)):

    input_size = batch.shape[-1]
    time_context = batch.shape[-2]
    batch_size = batch.shape[1]

    window = np.linspace(0.0, 1.0, num=overlap)
    window = np.concatenate((window, window[::-1]))
    window = np.repeat(np.expand_dims(window, axis=1), input_size, axis=1)

    output = np.zeros((int(num_chunks * (time_context - overlap) + time_context), input_size))

    index = 0
    start = 0
    while index < num_chunks:
        s = batch[int(index / batch_size), int(index % batch_size), :, :]

        if start == 0:
            output[0:time_context] = s

        else:
            output[int(start + overlap) : int(start + time_context)] = s[
                overlap:time_context
            ]
            output[start : int(start + overlap)] = (
                window[overlap:] * output[start : int(start + overlap)]
                + window[:overlap] * s[:overlap]
            )
        index = index + 1  # index for each block
        start = int(start - overlap + time_context)  # starting point for each block
    return output

# Matches the shape across the time dimension of a list of arrays.
# Assumes that the first dimension is in time, preserves the other dimensions
def match_time(feat_list):
    shapes = [f.shape for f in feat_list]
    shapes_equal = [s == shapes[0] for s in shapes]
    if not all(shapes_equal):
        min_time = np.min([s[0] for s in shapes])
        new_list = []
        for i in range(len(feat_list)):
            new_list.append(feat_list[i][:min_time])
        feat_list = new_list
    return feat_list

# Used in generation to plot the generated spectrogram.
def plot_features(feats, out_feats):
    plt.figure(1)
    ax1 = plt.subplot(211)
    plt.imshow(feats[:, :-2].T, aspect="auto", origin="lower")
    ax1.set_title("Ground Truth STFT", fontsize=10)
    ax3 = plt.subplot(212, sharex=ax1, sharey=ax1)
    ax3.set_title("Output STFT", fontsize=10)
    plt.imshow(out_feats.T, aspect="auto", origin="lower")
    plt.show()

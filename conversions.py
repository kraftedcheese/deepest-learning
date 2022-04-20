import config

import numpy as np
import pysptk

# Converts an input spectrogram into mel generalised cepstral (MGC)
def spectrogram_to_mel_generalized_cepstral(spectrogram, ndim, fw, ):
    # Convert the spectrogram to float64
    converted_spectrogram = spectrogram.astype(np.float64)
    mel_gen_cep = np.apply_along_axis(
        func1d=pysptk.mcep, 
        axis=1, 
        arr=np.atleast_2d(converted_spectrogram),
        # Params to pysptk.mcep
        order=ndim-1,
        alpha=fw,
        maxiter=0, 
        etype=1,
        eps=10**(config.noise_floor_db/10), 
        min_det=0.0, 
        itype=1,
    )
    
    # Flatten the output to have the same dimension as the spectrogram
    if converted_spectrogram.ndim == 1:
        mel_gen_cep = mel_gen_cep.flatten()
    
    # Return the same type as what was given in the spectrogram. 
    return mel_gen_cep.astype(spectrogram.dtype)

# Converts an input mel generalised cepstral (MGC) into a spectrogram 
def mel_generalized_cepstral_to_spectrogram(mel_gen_cep, spec_size, fw):
    # Convert the mgc to float64
    converted_mel_gen_cep = mel_gen_cep.astype(np.float64)

    # Fast Fourier Transformation Length 
    fftlen = 2*(spec_size - 1)
    spectrogram = np.apply_along_axis(
        func1d = pysptk.mgc2sp,
        axis = 1, 
        arr = np.atleast_2d(converted_mel_gen_cep), 
        # Params to pysptk.mgc2sp
        alpha=fw, 
        gamma=0.0,
        fftlen=fftlen,
    )

    spectrogram = 20*np.real(spectrogram)/np.log(10)

    # Flatten the output to have the same dimension as the input
    if converted_mel_gen_cep.ndim == 1:
        spectrogram = spectrogram.flatten()

    # Return the same type as what was given in the mgc. 
    spectrogram = spectrogram.astype(mel_gen_cep.dtype)
    return spectrogram

# Converts an input mel generalised cepstral (MGC) into a mel freq spectral coefficent 
def mel_generalized_cepstral_to_mel_freq_spectral_coefficent(mel_gen_cep):
    # Ensure that the mel gen cep is at least a 2d matrix.
    transformed_mel_gen_cep = np.atleast_2d(mel_gen_cep)
    transformed_ndim = transformed_mel_gen_cep.shape[1]

    concat_mel_gen_cep = np.concatenate(
        [
            transformed_mel_gen_cep[:, :], 
            transformed_mel_gen_cep[:, -2:0:-1]
        ], 
        axis=-1
    )

    # Rescaling
    concat_mel_gen_cep[:, 0] *= 2
    concat_mel_gen_cep[:, transformed_ndim-1] *= 2
    
    # Fast Fourier Transform 
    mel_freq_spectral_coefficent = np.real(np.fft.fft(concat_mel_gen_cep))[:, :transformed_ndim]
    mel_freq_spectral_coefficent = 10*mel_freq_spectral_coefficent/np.log(10)

    # Flatten the output to have the same dimension as the input
    if mel_gen_cep.ndim == 1:
        mel_freq_spectral_coefficent = mel_freq_spectral_coefficent.flatten()

    return mel_freq_spectral_coefficent

# Converts a spectrogram to a mel freq spectral coefficent 
def spectrogram_to_mel_freq_spectral_coefficent(sp, ndim, fw):
    mgc = spectrogram_to_mel_generalized_cepstral(sp, ndim, fw)
    return mel_generalized_cepstral_to_mel_freq_spectral_coefficent(mgc)

# Converts a a mel freq spectral coefficent to a mel generliased cepstral (mgc)
def mel_freq_spectral_coefficent_to_mel_generalized_cepstral(mel_freq_spectral_coefficent):
    transformed_mel_freq_cep_coeff = np.atleast_2d(mel_freq_spectral_coefficent)
    ndim = transformed_mel_freq_cep_coeff.shape[1]

    transformed_mel_freq_cep_coeff /= (10*np.log(10))
    concat_mel_freq_cep_coeff = np.concatenate(
        [
            transformed_mel_freq_cep_coeff[:, :],
            transformed_mel_freq_cep_coeff[:, -2:0:-1]
        ],
        axis=-1
    )

    # Fast Fourier Transform 
    mel_gen_cep = np.real(np.fft.ifft(concat_mel_freq_cep_coeff))
    mel_gen_cep[:, 0] /= 2
    mel_gen_cep[:, ndim-1] /= 2
    mel_gen_cep = mel_gen_cep[:, :ndim]

    # Flatten the output to have the same dimension as the input
    if mel_freq_spectral_coefficent.ndim == 1:
        mel_gen_cep = mel_gen_cep.flatten()
    
    return mel_gen_cep
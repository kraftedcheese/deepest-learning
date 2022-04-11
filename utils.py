import sys

import soundfile as sf
import numpy as np
from scipy.stats import norm
import pyworld as pw
import matplotlib.pyplot as plt
from reduce import sp_to_mfsc, mfsc_to_sp, ap_to_wbap,wbap_to_ap, get_warped_freqs, sp_to_mgc, mgc_to_sp, mgc_to_mfsc, mfsc_to_mgc
from vocoder import extract_sp_world, extract_ap_world, gen_wave_world
# from acoufe import pitch
import librosa
from tqdm import tqdm

import config



def shuffle_two(a,b):
    c = np.c_[a.reshape(len(a), -1), b.reshape(len(b), -1)]
    np.random.shuffle(c)
    a2 = c[:, :a.size//len(a)].reshape(a.shape)
    b2 = c[:, a.size//len(a):].reshape(b.shape)
    return a2, b2


def stft(data, window=np.hanning(1024),
         hopsize=256.0, nfft=1024.0, fs=44100.0):
    """
    X, F, N = stft(data,window=sinebell(2048),hopsize=1024.0,
                   nfft=2048.0,fs=44100)
                   
    Computes the short time Fourier transform (STFT) of data.
    
    Inputs:
        data                  :
            one-dimensional time-series to be analyzed
        window=sinebell(2048) :
            analysis window
        hopsize=1024.0        :
            hopsize for the analysis
        nfft=2048.0           :
            number of points for the Fourier computation
            (the user has to provide an even number)
        fs=44100.0            :
            sampling rate of the signal
        
    Outputs:
        X                     :
            STFT of data
        F                     :
            values of frequencies at each Fourier bins
        N                     :
            central time at the middle of each analysis
            window
    """
    return librosa.stft(data,hop_length=hopsize, n_fft=nfft, window=window)

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isinf(y), lambda z: z.nonzero()[0]

def file_to_stft(input_file, mode =0):
    audio,fs=sf.read(input_file)
    if mode == 0 :
        mixture = (audio[:,0]+audio[:,1])*0.7
        mix_stft=abs(stft(mixture))
    
        return mix_stft
    elif mode ==1:
        mixture = audio
        mix_stft=abs(stft(mixture))
    
        return mix_stft
    elif mode ==2:
        mixture = audio[:,0]
        mix_stft=abs(stft(mixture))
        return mix_stft
    elif mode ==3:
        mixture = audio

        mix_stft=stft(mixture)
        return abs(mix_stft), np.angle(mix_stft)







def input_to_feats(input_file, mode=0):
    audio,fs=sf.read(input_file)
    if mode == 0 or mode ==2:
        vocals=np.array(audio[:,1])
    elif mode ==1:
        vocals = audio

    feats = stft_to_feats(vocals,fs)


    # harm_in=mgc_to_sp(harmy, 1025, 0.45)
    # ap_in=mgc_to_sp(apy, 1025, 0.45)


    return feats

def stft_to_feats(vocals, fs, mode=config.comp_mode):
    """
    feats = f0, sp, ap
    f0: Pitch contour
    sp: Harmonic spectral envelope
    ap: Aperiodic spectral envelope (relative to the harmonic spectral envelope)
    """
    # 
    feats=pw.wav2world(vocals,fs,frame_period=5.80498866)

    ap = feats[2].reshape([feats[1].shape[0],feats[1].shape[1]]).astype(np.float32)
    ap = 10.*np.log10(ap**2)
    harm=10*np.log10(feats[1].reshape([feats[2].shape[0],feats[2].shape[1]]))
    feats=pw.wav2world(vocals,fs,frame_period=5.80498866)

    f0 = feats[0]
    """
    Converting from frequency to midi numbers
    Midi number 69 represents the equal tempered frequncy 440hz.
    m  =  12*log2(fm/440 Hz) + 69
    """
    midi_number=69+12*np.log2(f0/440)
   
    nans, x= nan_helper(midi_number)
    naners=np.isinf(midi_number)    
    midi_number[nans]= np.interp(x(nans), x(~nans), midi_number[~nans])

    midi_number=np.array(midi_number).reshape([len(midi_number),1])
    guy=np.array(naners).reshape([len(midi_number),1])
    midi_number=np.concatenate((midi_number,guy),axis=-1)

    if mode == 'mfsc':
        harmy=sp_to_mfsc(harm,60,0.45)
        apy=sp_to_mfsc(ap,4,0.45)
    elif mode == 'mgc':
        harmy=sp_to_mgc(harm,60,0.45)
        apy=sp_to_mgc(ap,4,0.45)

    out_feats=np.concatenate((harmy,apy,midi_number.reshape((-1,2))),axis=1) 

    return out_feats

def write_ori_ikala(input_file, filename):
    audio,fs = sf.read(input_file)
    mixture = (audio[:,0]+audio[:,1])*0.7
    vocals = np.array(audio[:,1])
    backing = np.array(audio[:,0])
    sf.write(config.val_dir+filename+'_mixture.wav',mixture,fs)
    sf.write(config.val_dir+filename+'_ori_vocals.wav',vocals,fs)
    sf.write(config.val_dir+filename+'_backing.wav',backing,fs)

def write_ori_med(input_file, filename):
    audio,fs = sf.read(input_file)
    mixture = np.array(audio[:,0])
    vocals = np.array(audio[:,1])
    sf.write(config.val_dir+filename+'_mixture.wav',mixture,fs)
    sf.write(config.val_dir+filename+'_ori_vocals.wav',vocals,fs)


def f0_to_hertz(f0):
    # if f0 == 0:
    #     return 0
    # else:
    f0 = f0-69
    f0 = f0/12
    f0 = 2**f0
    f0 = f0*440
    return f0


def hertz_to_new_base(f0):
    # if f0 == 0:
    #     return 0
    # else:
    return 1200*np.log2(f0/10)

def new_base_to_hertz(f0):
    return 2**(f0*10)/1200




def feats_to_audio(in_feats,filename, fs=config.fs,  mode=config.comp_mode):
    harm = in_feats[:,:60]
    ap = in_feats[:,60:-2]
    f0 = in_feats[:,-2:]
    # f0[:,0] = f0[:,0]-69
    # f0[:,0] = f0[:,0]/12
    # f0[:,0] = 2**f0[:,0]
    # f0[:,0] = f0[:,0]*440
    f0[:,0] = f0_to_hertz(f0[:,0])

    f0 = f0[:,0]*(1-f0[:,1])


    if mode == 'mfsc':
        harm = mfsc_to_mgc(harm)
        ap = mfsc_to_mgc(ap)


    harm = mgc_to_sp(harm, 1025, 0.45)
    ap = mgc_to_sp(ap, 1025, 0.45)

    harm = 10**(harm/10)
    ap = 10**(ap/20)

    y=pw.synthesize(f0.astype('double'),harm.astype('double'),ap.astype('double'),fs,config.hoptime)
    sf.write(config.val_dir+filename+'.wav',y,fs)

def feats_to_audio_test(in_feats,filename, fs=config.fs,  mode=config.comp_mode):
    harm = in_feats[:,:60]
    ap = in_feats[:,60:-2]
    f0 = in_feats[:,-2:]
    f0[:,0] = f0[:,0]-69
    f0[:,0] = f0[:,0]/12
    f0[:,0] = 2**f0[:,0]
    f0[:,0] = f0[:,0]*440


    f0 = f0[:,0]*(1-f0[:,1])


    if mode == 'mfsc':
        harm = mfsc_to_mgc(harm)
        ap = mfsc_to_mgc(ap)


    harm = mgc_to_sp(harm, 1025, 0.45)
    ap = mgc_to_sp(ap, 1025, 0.45)

    harm = 10**(harm/10)
    ap = 10**(ap/20)

    y=pw.synthesize(f0.astype('double'),harm.astype('double'),ap.astype('double'),fs,config.hoptime)
    sf.write('./medley_resynth_test/'+filename+'.wav',y,fs)
    # return harm, ap, f0

def test(ori, re):
    plt.subplot(211)
    plt.imshow(ori.T,origin='lower',aspect='auto')
    plt.subplot(212)
    plt.imshow(re.T,origin='lower',aspect='auto')
    plt.show()


def generate_overlapadd(allmix,time_context=config.max_phr_len, overlap=config.max_phr_len/2,batch_size=config.batch_size):
    input_size = allmix.shape[-1]

    i=0
    start=0  
    while (start + time_context) < allmix.shape[0]:
        i = i + 1
        start = start - overlap + time_context 
    fbatch = np.zeros([int(np.ceil(float(i)/batch_size)),batch_size,time_context,input_size])+1e-10
    
    
    i=0
    start=0  

    while (start + time_context) < allmix.shape[0]:
        fbatch[int(i/batch_size),int(i%batch_size),:,:]=allmix[int(start):int(start+time_context),:]
        i = i + 1 #index for each block
        start = start - overlap + time_context #starting point for each block
    
    return fbatch,i

def overlapadd(fbatch,nchunks,overlap=int(config.max_phr_len/2)):

    input_size=fbatch.shape[-1]
    time_context=fbatch.shape[-2]
    batch_size=fbatch.shape[1]


    window = np.linspace(0., 1.0, num=overlap)
    window = np.concatenate((window,window[::-1]))
    window = np.repeat(np.expand_dims(window, axis=1),input_size,axis=1)
    

    sep = np.zeros((int(nchunks*(time_context-overlap)+time_context),input_size))

    
    i=0
    start=0 
    while i < nchunks:
        s = fbatch[int(i/batch_size),int(i%batch_size),:,:]

        if start==0:
            sep[0:time_context] = s

        else:
            sep[int(start+overlap):int(start+time_context)] = s[overlap:time_context]
            sep[start:int(start+overlap)] = window[overlap:]*sep[start:int(start+overlap)] + window[:overlap]*s[:overlap]
        i = i + 1 #index for each block
        start = int(start - overlap + time_context) #starting point for each block
    return sep  


def normalize(inputs, feat, mode=config.norm_mode_in):
    if mode == "max_min":
        maximus = np.load(config.stat_dir+feat+'_maximus.npy')
        minimus = np.load(config.stat_dir+feat+'_minimus.npy')
        outputs = (inputs-minimus)/(maximus-minimus)

    elif mode == "mean":
        means = np.load(config.stat_dir+feat+'_means.npy')
        stds = np.load(config.stat_dir+feat+'_stds.npy')
        outputs = (inputs-means)/stds
    elif mode == "clip":
        outputs = np.clip(inputs, 0.0,1.0)

    return outputs


def list_to_file(in_list,filename):
    filer=open(filename,'w')
    for jj in in_list:
        filer.write(str(jj)+'\n')
    filer.close()

def denormalize(inputs, feat, mode=config.norm_mode_in):
    if mode == "max_min":
        maximus = np.load(config.stat_dir+feat+'_maximus.npy')
        minimus = np.load(config.stat_dir+feat+'_minimus.npy')
        # import pdb;pdb.set_trace()
        outputs = (inputs*(maximus-minimus))+minimus

    elif mode == "mean":
        means = np.load(config.stat_dir+feat+'_means.npy')
        stds = np.load(config.stat_dir+feat+'_stds.npy')
        outputs = (inputs*stds)+means
    return outputs

def query_yes_no(question, default="yes"):
    """
    Copied from https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def match_time(feat_list):
    """ 
    Matches the shape across the time dimension of a list of arrays.
    Assumes that the first dimension is in time, preserves the other dimensions
    """
    shapes = [f.shape for f in feat_list]
    shapes_equal = [s == shapes[0] for s in shapes]
    if not all(shapes_equal):
        min_time = np.min([s[0] for s in shapes])
        new_list = []
        for i in range(len(feat_list)):
            new_list.append(feat_list[i][:min_time])
        feat_list = new_list
    return feat_list

def main():
    out_feats = input_to_feats(config.NUS_DIR+'/ADIZ/read/01.wav', mode=1)
    feats_to_audio(out_feats, 'test')

    import pdb;pdb.set_trace()




if __name__ == '__main__':
    main()
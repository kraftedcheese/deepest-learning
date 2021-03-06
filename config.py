# CONFIG VARIABLES
SAMPLE_RATE = 44100
norm_mode_in = "max_min"
max_phr_len = 128
phonemas_nus = [
    "t",
    "y",
    "l",
    "k",
    "aa",
    "jh",
    "ae",
    "ng",
    "ah",
    "hh",
    "z",
    "ey",
    "f",
    "uw",
    "iy",
    "ay",
    "b",
    "s",
    "d",
    "Sil",
    "p",
    "n",
    "sh",
    "ao",
    "g",
    "ch",
    "ih",
    "eh",
    "aw",
    "sp",
    "oy",
    "th",
    "w",
    "ow",
    "v",
    "uh",
    "m",
    "er",
    "zh",
    "r",
    "dh",
    "ax",
]
num_phos = len(phonemas_nus)
singers = [
    "ADIZ",
    "JLEE",
    "JTAN",
    "KENN",
    "MCUR",
    "MPOL",
    "MPUR",
    "NJAT",
    "PMAR",
    "SAMF",
    "VKOW",
    "ZHIY",
]
aug_prob = 0.5

fs = 44100  
filters = 64

# Hyperparameters
num_epochs = 950
batches_per_epoch_train = 100
batch_size = 30
samples_per_file = 6
conv_filters = 128
comp_mode = "mfsc"
hoptime = 5.80498866

#Steps
save_every = 50
validate_every = 50
n_critic = 15

# Directories
NUS_DIR = "./datasets/nus-smc-corpus_48/"
voice_dir = "./ss_synthesis/voice/"
stat_dir = "./stats/"
val_dir = "./val_dir_synth/"
model_save_dir = "model_save_dir"
log_dir = "log_dir"
graph_dir = "graph_dir"

stats_file_name = "stats.hdf5"

# Conversion Parameters
noise_floor_db = -120.0

# Training Loss Pandas Keys
EPOCH_KEY = "epoch"
LOSS_FAKE_KEY = 'loss_fake'
LOSS_REAL_KEY = 'loss_real'
DISCRIMINATOR_LOSS_KEY ="discriminator_loss"
W_D_KEY = 'wasserstein_D' 
G_LOSS_KEY = "g_loss"

# Validation Loss Pandas Keys
VAL_LOSS_KEY = "val_loss"


# hdf5 keys 
feats_maximus_key = "feats_maximus"
feats_minimus_key = "feats_minimus"
phonemes_key = "phonemes"
feats_key = 'feats'
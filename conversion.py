
import config
import numpy as np
import matplotlib.pyplot as plt
from model.wgan_model import WGANModel
import os

voc_list = [x for x in os.listdir(config.voice_dir) if 
x.endswith('.hdf5') and x.startswith('nus') and 
not x == 'nus_MCUR_sing_04.hdf5' and 
not x == 'nus_ADIZ_read_01.hdf5' and 
not x == 'nus_JLEE_sing_05.hdf5' and 
not x == 'nus_JTAN_read_07.hdf5']

model = WGANModel(voc_list, reload_model=150)

model.test_file_hdf5("nus_ADIZ_sing_01.hdf5", "ADIZ")





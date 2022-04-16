from model.wgan_model import WGANModel
import config
import os
import sys
import argparse

def main(args):
    voc_list = [x for x in os.listdir(config.voice_dir) if 
    x.endswith('.hdf5') and x.startswith('nus') and 
    not x == 'nus_MCUR_sing_04.hdf5' and 
    not x == 'nus_ADIZ_read_01.hdf5' and 
    not x == 'nus_JLEE_sing_05.hdf5' and 
    not x == 'nus_JTAN_read_07.hdf5']

    model = WGANModel(voc_list, args.reload_model)
    model.train()

    # start evaluating on test data
    # else:
    #     model.evaluate(test_loader, args.load_D, args.load_G)
    #     for i in range(50):
    #        model.generate_latent_walk(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reload_model', type=int, default='0')
    args = parser.parse_args()
    main(args)



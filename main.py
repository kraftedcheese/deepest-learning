from model.wgan_model import WGANModel
import config
import os
import argparse

def main(args):
    #TODO: adjust this
    voc_list = [x for x in os.listdir(config.voice_dir) if 
    x.endswith('.hdf5') and x.startswith('nus') and 
    not x == 'nus_MCUR_sing_04.hdf5' and 
    not x == 'nus_ADIZ_read_01.hdf5' and 
    not x == 'nus_JLEE_sing_05.hdf5' and 
    not x == 'nus_JTAN_read_07.hdf5']

    model = WGANModel(voc_list, args.reload_model)
    
    if args.eval == 'y':
        if args.reload_model == 0:
            print("No checkpoint loaded! To evaluate, run main.py --eval y --reload_model \{checkpoint_number\}")
            return
        model.test_file_hdf5(args.source, args.target)
    else:
        model.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # For both training and evaluation, pass in --reload_model {checkpoint} to load.
    # The checkpoint file MUST be in model_save_dir.
    parser.add_argument('--reload_model', type=int, default='0')

    # Flag for evaluation, pass in --eval {y}
    parser.add_argument('--eval', type=str, default='n')
    parser.add_argument('--source', type=str, default='nus_ADIZ_sing_01.hdf5')
    parser.add_argument('--target', type=str, default='ADIZ')
    args = parser.parse_args()
    main(args)



from model.wgan_model import WGANModel
import config
import os

def main():
    voc_list = [x for x in os.listdir(config.voice_dir) if 
    x.endswith('.hdf5') and x.startswith('nus') and 
    not x == 'nus_MCUR_sing_04.hdf5' and 
    not x == 'nus_ADIZ_read_01.hdf5' and 
    not x == 'nus_JLEE_sing_05.hdf5' and 
    not x == 'nus_JTAN_read_07.hdf5']

    model = WGANModel(voc_list)
    model.train()

    # start evaluating on test data
    # else:
    #     model.evaluate(test_loader, args.load_D, args.load_G)
    #     for i in range(50):
    #        model.generate_latent_walk(i)


if __name__ == '__main__':
    main()



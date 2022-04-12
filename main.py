from parse_args_utils import parse_args
from model.wgan_model import WGANModel
import config
import os

def main(args):
    voc_list = [x for x in os.listdir(config.voice_dir) if 
    x.endswith('.hdf5') and x.startswith('nus') and 
    not x == 'nus_MCUR_sing_04.hdf5' and 
    not x == 'nus_ADIZ_read_01.hdf5' and 
    not x == 'nus_JLEE_sing_05.hdf5' and 
    not x == 'nus_JTAN_read_07.hdf5']

    # train_loader, test_loader = get_data_loader(args)
    model = WGANModel(args, voc_list)
    # Load datasets to train and test loaders
    #feature_extraction = FeatureExtractionTest(train_loader, test_loader, args.cuda, args.batch_size)

    # Start model training
    if args.is_train == 'True':
        model.train()

    # start evaluating on test data
    # else:
    #     model.evaluate(test_loader, args.load_D, args.load_G)
    #     for i in range(50):
    #        model.generate_latent_walk(i)


if __name__ == '__main__':
    args = parse_args()
    main(args)



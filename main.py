import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from LR import LR
from dnn_demo1 import DNN
from ResNet import ResNet50
from utils import show_all_variables
from utils import check_folder

import tensorflow as tf
import argparse

"""parsing and configuration"""


def parse_args():
    desc = "Tensorflow implementation of model collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--type', type=str, default='LR',
                        help='The type of Model', required=True)
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')

    return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --epoch
    assert args.epoch >= 1, 'number of epochs must be larger than or equal to one'

    # --batch_size
    assert args.batch_size % 2 == 0, 'batch size must be 2**n'

    return args


"""main"""


def main():
    print("beginning......")
    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    print("result_dir is: " + arg.result_dir)

    # open session
    models = [LR, DNN, ResNet50]
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # declare instance for GAN

        model_obj = None
        for model in models:
            if args.type == model.model_name:
                model_obj = model(sess,
                                  epoch=args.epoch,
				  all_cnt=921600,
                                  batch_size=args.batch_size,
                                  dataset_name=args.dataset,
                                  checkpoint_dir=args.checkpoint_dir,
                                  result_dir=args.result_dir,
                                  log_dir=args.log_dir)
        if model_obj is None:
            raise Exception("[!] There is no option for " + args.type)

        # build graph
        model_obj.build_model()

        # show network architecture
        show_all_variables()

        # launch the graph in a session
        model_obj.train()
        print(" [*] Training finished!")

        # visualize learned generator
        model_obj.visualize_results(args.epoch - 1, on_train=True)
        print(" [*] Testing finished!")


if __name__ == '__main__':
    main()

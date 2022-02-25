import torch
import os
from tqdm import tqdm
import random
import numpy as np
import cv2
import argparse
import networks
from networks.auxiliary import InpaintModel
from networks.config import Config
from networks.nearestx2 import InpaintGeneratorLarge

# parser = argparse.ArgumentParser(description='test script')
# parser.add_argument('--image', default='./examples/places/images', type=str)
# parser.add_argument('--mask', default='./examples/places/masks', type=str)
# parser.add_argument('--output', default='./examples/results', type=str)
# parser.add_argument('--nogpu', action='store_true')
# parser.add_argument('--opt', default='convnet', type=str)
# parser.add_argument('--load', default='./files/model_256.pth', type=str)
# args = parser.parse_args()


def main(mode=None):
    config = load_config(mode)


    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)


    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)

    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")

    cv2.setNumThreads(0)

    # torch.manual_seed(config.SEED)
    # torch.cuda.manual_seed_all(config.SEED)
    # np.random.seed(config.SEED)
    # random.seed(config.SEED)
    if config.MODE == 2:
      model = InpaintModel(config,base_g = False)
    else:
      if not config.HIGH_RES and config.ORIGIN:
        model = InpaintModel(config,base_g = None)
        # print("a")
      elif config.HIGH_RES and config.ORIGIN:
        model = InpaintModel(config,base_g = True)
        # print("b")
      else: 
        model = InpaintModel(config,base_g = False)
      # print("c")
    model.load()

    if config.MODE == 1:
        config.print()
        print('\nstart training...\n')
        model.train()

    # model test
    elif config.MODE == 2:
        print('\nstart testing...\n')
        model.test()

    # eval mode
    else:
        print('\nstart eval...\n')
        model.eval()


def load_config(mode=None):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints', help='model checkpoints path (default: ./checkpoints)')
    parser.add_argument('--name', type=str)
    parser.add_argument('--mode', type=int)
 
    # parser.add_argument('--model', type=int, choices=[1, 2, 3, 4], help='1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model')
    # args = parser.parse_args()
    # mode = args.mode
    # test mode
    if mode == 2:
        parser.add_argument('--input', type=str, help='path to the input images directory or an input image')
        parser.add_argument('--mask', type=str, help='path to the masks directory or a mask file')
        # parser.add_argument('--edge', type=str, help='path to the edges directory or an edge file')
        parser.add_argument('--output', type=str, help='path to the output directory')
    else:
        parser.add_argument('--dis', type=int , default=0)
        parser.add_argument('--high_res', type=int , default=0)
        parser.add_argument('--origin', type=int , default=0)

    args = parser.parse_args()
    mode = args.mode
    # args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')

    # create checkpoints path if does't exist
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile('./config.yml.example', config_path)

    # load config file
    config = Config(config_path)
    if not mode == 2:
      config.HIGH_RES = args.high_res 
      config.ORIGIN = args.origin
    # train mode
    if mode == 1:
        config.MODE = 1
        if args.name:
            config.NAME = args.name
        config.DIS_CHANGE = args.dis
        # if args.model:
        #     config.MODEL = args.model

    # test mode
    elif mode == 2:
        config.MODE = 2
        # config.MODEL = args.model if args.model is not None else 3
        config.INPUT_SIZE = 256

        if args.input is not None:
            config.TEST_FLIST = args.input

        if args.mask is not None:
            config.TEST_MASK_FLIST = args.mask

        # if args.edge is not None:
        #     config.TEST_EDGE_FLIST = args.edge

        if args.output is not None:
            config.RESULTS = args.output

    # eval mode
    elif mode == 3:
        config.MODE = 3
        # config.MODEL = args.model if args.model is not None else 3

    return config


if __name__ == "__main__":
    main()








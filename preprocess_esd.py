import argparse

import yaml

from preprocessor.preprocessor_esd import Preprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--config", type=str,required=False, default="./config/cl_esd/preprocess.yaml",help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    preprocessor = Preprocessor(config)
    preprocessor.build_from_path()

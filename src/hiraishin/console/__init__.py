import argparse
import importlib


def generate_config():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    # TODO: add datamodule
    parser.add_argument('--output_dir', type=str, default='./config')
    args = parser.parse_args()

    if args.model is not None:
        model_cls = getattr(importlib.import_module('hiraishin.models'), f'{args.model}Model')
        model_cls.generate_config(args.output_dir)

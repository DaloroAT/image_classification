import argparse
from argparse import Namespace, ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt

from network import Classifier, resnet18_num_classes


def main(args: Namespace) -> None:

    net = resnet18_num_classes(pretrained=False, num_classes=200, p_drop=0.5, type_net='custom')
    net.cpu()
    classifier = Classifier(net)
    classifier.load(args.path_weights.resolve())

    fig_vis = classifier.classify_and_gradcam_by_path(args.path_pic.resolve(), target_class=args.target_class)

    if args.path_save != Path(''):
        fig_vis.savefig(args.path_save.resolve())


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument('--path_pic', dest='path_pic', type=Path)
    parser.add_argument('--path_weights', dest='path_weights', type=Path)

    parser.add_argument('--path_save', dest='path_save', type=Path, default=Path(''))
    parser.add_argument('--class', dest='target_class', type=int, default=None)

    return parser


if __name__ == '__main__':
    arg_parser = get_parser()
    main(args=arg_parser.parse_args())

from datetime import datetime
import argparse
from argparse import Namespace, ArgumentParser

from pathlib import Path
import torch
from datasets import TinyImagenetDataset
import pandas as pd

from utils import prepare_dataframes, fix_seed, Stopper, parse_to_dict
from network import resnet18_num_classes, Classifier
from trainer import Trainer


def main(args: Namespace):
    args.freeze = parse_to_dict(args.freeze)
    args.aug_degree = parse_to_dict(args.aug_degree)

    results_path = args.log_dir / str(datetime.now())
    results_path.mkdir(exist_ok=True, parents=True)

    write_args(results_path, vars(args))

    fix_seed(args.seed)

    (train_frame, test_frame), labels_num2txt = prepare_dataframes(args.data_root)

    train_set = TinyImagenetDataset(train_frame)
    test_set = TinyImagenetDataset(test_frame)

    model = resnet18_num_classes(pretrained=True,
                                 num_classes=200,
                                 p_drop=args.prob_drop,
                                 type_net=args.arch)

    classifier = Classifier(net=model)

    stopper = Stopper(args.n_wrongs, args.delta_wrongs)

    trainer = Trainer(classifier=classifier,
                      train_set=train_set,
                      test_set=test_set,
                      results_path=results_path,
                      device=args.device,
                      batch_size=args.batch_size,
                      num_workers=args.num_workers,
                      num_visual=args.num_visual,
                      aug_degree=args.aug_degree,
                      lr=args.lr,
                      lr_min=args.lr_min,
                      stopper=stopper,
                      labels_num2txt=labels_num2txt,
                      freeze=args.freeze,
                      weight_decay=args.weight_decay,
                      label_smooth=args.label_smooth,
                      period_cosine=args.period_cosine)

    trainer.train(num_epoch=args.num_epoch)
    
    
def write_args(path: Path, args_dict: dict) -> None:
    frame_args = pd.DataFrame.from_dict(args_dict, orient="index")
    frame_args.to_csv(str(path / 'args.txt'), header=False, sep='=', mode='w+')


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_root', dest='data_root', type=Path)
    parser.add_argument('--log_dir', dest='log_dir', type=Path)

    parser.add_argument('--lr', dest='lr', type=float, default=5e-2)
    parser.add_argument('--lr_min', dest='lr_min', type=float, default=5e-4)
    parser.add_argument('--num_epoch', dest='num_epoch', type=int, default=150)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=300)
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=4)
    parser.add_argument('--device', dest='device', type=torch.device, default='cuda')
    parser.add_argument('--random_seed', dest='seed', type=int, default=30,
                        help='Fix seed on random of python, numpy and torch')
    parser.add_argument('--aug_degree', dest='aug_degree', metavar="KEY=VALUE", nargs='+', default=['0=0'],
                        help='Augmentation degree for training on arbitrary epochs')

    parser.add_argument('--freeze', dest='freeze', metavar="KEY=VALUE", nargs='+', default=['0=1', '3=float("inf")'],
                        help='Freeze k last layers of arbitrary epochs. '
                             'In order to unfreeze all layers use float("inf") or any number that greater than number '
                             'of layers')

    parser.add_argument('--arch', dest='arch', type=str, default='custom',
                        help='Arch of resnet18. Use the "classic" for classical architecture, or a "custom" in which '
                             'some strides are set equal to 1')
    parser.add_argument('--prob_drop', dest='prob_drop', type=float, default=0.2)
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-5)
    parser.add_argument('--label_smooth', dest='label_smooth', type=float, default=0.1)
    parser.add_argument('--period_cosine', dest='period_cosine', type=int, default=1)

    parser.add_argument('--n_wrongs', dest='n_wrongs', type=int, default=30)
    parser.add_argument('--delta_wrongs', dest='delta_wrongs', type=float, default=0.001)
    parser.add_argument('--num_visual', dest='num_visual', type=int, default=15)

    return parser


if __name__ == '__main__':
    arg_parser = get_parser()
    main(args=arg_parser.parse_args())

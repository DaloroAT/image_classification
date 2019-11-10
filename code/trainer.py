from pathlib import Path
from typing import Tuple, List, Optional, Sequence, Union, Dict, Any, Callable
import random
from math import ceil

import torch
import os
from PIL import Image
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from PIL import Image
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import TinyImagenetDataset
from network import Classifier
from utils import Stopper, AvgMoving
from loss_probs import cross_entropy_with_probs


class Trainer:
    _classifier: Classifier
    _train_set: TinyImagenetDataset
    _test_set: TinyImagenetDataset
    _results_path: Path
    _device: torch.device
    _batch_size: int
    _num_workers: int
    _num_visual: int
    _aug_degree: Dict
    _lr: float
    _lr_min: float
    _stopper: Stopper
    _labels_num2txt: Dict
    _freeze: Dict
    _weight_decay: float
    _label_smooth: float
    _period_cosine: int

    _net_path: Path
    _tensorboard_path: Path
    _writer: SummaryWriter
    _optimizer: Optimizer
    _scheduler: CosineAnnealingWarmRestarts
    _loss_func_smoothed: Callable
    _loss_func_one_hot: nn.Module
    _curr_epoch: int
    _vis_per_batch: int

    def __init__(self,
                 classifier: Classifier,
                 train_set: TinyImagenetDataset,
                 test_set: TinyImagenetDataset,
                 results_path: Path,
                 device: torch.device,
                 batch_size: int,
                 num_workers: int,
                 num_visual: int,
                 aug_degree: Dict,
                 lr: float,
                 lr_min: float,
                 stopper: Stopper,
                 labels_num2txt: Dict,
                 freeze: Dict,
                 weight_decay: float,
                 label_smooth: float,
                 period_cosine: int):

        self._classifier = classifier
        self._train_set = train_set
        self._test_set = test_set
        self._results_path = results_path
        self._device = device
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._num_visual = num_visual
        self._aug_degree = aug_degree
        self._lr = lr
        self._lr_min = lr_min
        self._stopper = stopper
        self._labels_num2txt = labels_num2txt
        self._freeze = freeze
        self._weight_decay = weight_decay
        self._label_smooth = label_smooth
        self._period_cosine = period_cosine

        self._classifier.to(self._device)

        self._net_path, self._tensorboard_path = self._results_path / 'net', self._results_path / 'tensorboard'

        for folder in [self._net_path, self._tensorboard_path]:
            folder.mkdir(exist_ok=True, parents=True)

        self._writer = SummaryWriter(log_dir=str(self._tensorboard_path))

        self._loss_func_smoothed = cross_entropy_with_probs
        self._loss_func_one_hot = nn.CrossEntropyLoss()

        self._optimizer = torch.optim.SGD(self._classifier.parameters(),
                                          lr=self._lr,
                                          momentum=0.9,
                                          weight_decay=self._weight_decay)

        self._scheduler = CosineAnnealingWarmRestarts(self._optimizer, T_0=self._period_cosine,
                                                      eta_min=self._lr_min, T_mult=2)

        self._curr_epoch = 0
        self._vis_per_batch = self._calc_vis_per_batch()

    def _calc_vis_per_batch(self) -> int:
        num_batch = ceil(len(self._test_set) / self._batch_size)
        return ceil(self._num_visual / num_batch)

    def train(self, num_epoch: int) -> Tuple[float, float, int]:
        accuracy_max = 0
        accuracy_test = 0
        loss_test = 0

        for epoch in range(num_epoch):

            self._curr_epoch = epoch

            self._set_freezing()

            accuracy_train, loss_train = self._train_epoch()

            accuracy_test, loss_test = self.test()

            self._writer.flush()

            self._stopper.update(accuracy_test)

            meta = {'epoch': self._curr_epoch,
                    'accuracy_test': accuracy_test,
                    'accuracy_train': accuracy_train,
                    'loss_test': loss_test,
                    'loss_train': loss_train,
                    'lr': self._scheduler.get_lr()[0]}

            self._classifier.save(self._net_path / f'net_epoch_{self._curr_epoch}.pth', meta)

            if accuracy_test > accuracy_max:
                accuracy_max = accuracy_test

                self._classifier.save(self._net_path / 'best.pth', meta)

            if self._stopper.is_need_stop():
                break

        self._writer.close()
        return accuracy_test, loss_test, self._curr_epoch

    def _train_epoch(self) -> Tuple[float, float]:
        self._classifier.train()
        self._set_aug_train()
        train_loader = DataLoader(self._train_set,
                                  batch_size=self._batch_size,
                                  num_workers=self._num_workers,
                                  shuffle=True)

        num_batches = len(train_loader)
        correct = 0

        train_tqdm = tqdm(train_loader, desc=f'train_{self._curr_epoch}')
        loss_avg = AvgMoving()
        for curr_batch, (images, labels, _) in enumerate(train_tqdm):
            self._optimizer.zero_grad()

            images = images.to(self._device)
            labels = labels.to(self._device)

            output = self._classifier(images)

            if self._label_smooth > 0:
                smoothed_labels = smooth_one_hot(labels=labels, num_classes=200, smoothing=self._label_smooth)
                loss = self._loss_func_smoothed(output, smoothed_labels)
            else:
                loss = self._loss_func_one_hot(output, labels)

            loss.backward()

            self._scheduler.step(self._curr_epoch + (curr_batch + 1) / num_batches)
            self._optimizer.step()

            loss = loss.item()
            loss_avg.add(loss)

            pred = output.argmax(dim=1)
            correct += torch.eq(pred, labels).sum().item()

            train_tqdm.set_postfix({'Avg train loss': round(loss_avg.avg, 4)})

        accuracy = correct / len(train_loader.dataset)
        self._add_writer_metrics(loss_avg.avg, accuracy, 'train')

        return accuracy, loss_avg.avg

    def test(self) -> Tuple[float, float]:
        self._classifier.eval()
        with torch.no_grad():
            test_loader = DataLoader(self._test_set,
                                     batch_size=self._batch_size,
                                     num_workers=self._num_workers,
                                     shuffle=True)
            correct = 0
            test_tqdm = tqdm(test_loader, desc=f'test_{self._curr_epoch}', leave=False)
            loss_avg = AvgMoving()

            labels_pred_list = []

            worst_pred_list = []
            best_pred_list = []
            some_pred_list = []

            for images, labels, paths in test_tqdm:
                images = images.to(self._device)
                labels = labels.to(self._device)

                output = self._classifier(images)
                output = softmax(output, dim=1)

                pred = output.argmax(dim=1)
                correct += torch.eq(pred, labels).sum().item()

                loss_avg.add(self._loss_func_one_hot(output, labels).item())

                labels = labels.detach().cpu().numpy()
                output = output.detach().cpu().numpy()
                pred = pred.detach().cpu().numpy()
                prob = output[np.arange(np.size(labels)), pred]

                # prepare data for cunfusion matrix and hists
                labels_pred = np.hstack((labels[:, np.newaxis], pred[:, np.newaxis]))
                labels_pred_list.append(np.copy(labels_pred))

                # prepare data for visualiztion
                prob_in_labels = output[np.arange(np.size(labels)), labels]

                worst_pred, best_pred, some_pred = self._prepare_data_for_vis(np.copy(pred),
                                                                              np.copy(prob),
                                                                              np.copy(labels),
                                                                              np.copy(prob_in_labels),
                                                                              paths)

                worst_pred_list.append(worst_pred)
                best_pred_list.append(best_pred)
                some_pred_list.append(some_pred)

            accuracy = correct / len(test_loader.dataset)

            self._add_writer_metrics(loss_avg.avg, accuracy, 'test')
            self._visual_confusion_and_hists(labels_pred_list)
            self._visual_gt_and_pred(worst_pred_list, 'Worst_pred_pic')
            self._visual_gt_and_pred(best_pred_list, 'Best_pred_pic')
            self._visual_gt_and_pred(some_pred_list, 'Some_pred_pic')

        self._classifier.train()
        return accuracy, loss_avg.avg

    def _freeze_except_k_last(self, k_last: int) -> None:
        num_layers = 0
        for _ in self._classifier._net.children():
            num_layers += 1

        for i, layer in enumerate(self._classifier._net.children()):
            if i + k_last < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    def _unfreeze_all(self) -> None:
        for param in self._classifier.parameters():
            param.requires_grad = True

    def _set_freezing(self) -> None:
        curr_epoch = str(self._curr_epoch)
        if curr_epoch in tuple(self._freeze.keys()):
            self._unfreeze_all()
            self._freeze_except_k_last(self._freeze[curr_epoch])

            self._optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self._classifier.parameters()),
                                              lr=self._lr,
                                              momentum=0.9,
                                              weight_decay=self._weight_decay)
            self._scheduler = CosineAnnealingWarmRestarts(self._optimizer, T_0=self._period_cosine,
                                                          eta_min=self._lr_min, T_mult=2)

    def _set_aug_train(self) -> None:
        curr_epoch = str(self._curr_epoch)
        if curr_epoch in tuple(self._aug_degree.keys()):
            self._train_set.set_transforms(self._aug_degree[curr_epoch])

    def _add_writer_metrics(self, loss: float, accuracy: float, mode: str) -> None:
        self._writer.add_scalars('loss', {f'loss_{mode}': loss},
                                 self._curr_epoch)
        self._writer.add_scalars('accuracy', {f'accuracy_{mode}': accuracy},
                                 self._curr_epoch)

# VISUALIZATON

    def _prepare_data_for_vis(self, pred: np.ndarray, prob: np.ndarray, labels: np.ndarray, prob_in_labels: np.ndarray,
                              paths: Path) -> List[Tuple[List[Path], np.ndarray, np.ndarray]]:
        prob_idx = np.argsort(prob_in_labels)
        prob_idx_list = list()
        prob_idx_list.append(prob_idx[0: self._vis_per_batch])  # worst preds
        prob_idx_list.append(prob_idx[-1: -self._vis_per_batch - 1: -1])  # best preds
        prob_idx_list.append(np.random.choice(prob_idx, self._vis_per_batch))  # some preds

        data_list = []
        for idx in prob_idx_list:
            pred_and_prob = np.hstack((pred[idx, np.newaxis],
                                       prob[idx, np.newaxis]))

            labels_and_prob_in_labels = np.hstack((labels[idx, np.newaxis],
                                                   prob_in_labels[idx, np.newaxis]))

            data = ([paths[k] for k in idx],
                    pred_and_prob,
                    labels_and_prob_in_labels)
            data_list.append(data)

        return data_list

    def _visual_gt_and_pred(self, data_list: List[Tuple[List[Path], np.ndarray, np.ndarray]], txt: str) -> None:
        num_batch = ceil(len(self._test_set) / self._batch_size)
        num_elem = num_batch * self._vis_per_batch

        images_array = np.zeros((num_elem, 64, 64, 3), int)
        pred_and_prob_array = np.zeros((num_elem, 2))
        labels_and_prob_in_labels_array = np.zeros((num_elem, 2))
        filenames = []
        k_im = 0
        for i, (paths, pred_and_prob, labels_and_prob_in_labels) in enumerate(data_list):
            filenames += paths
            for path in paths:
                images_array[k_im, :] = np.array(Image.open(path).convert('RGB'))
                k_im += 1

            pred_and_prob_array[i * self._vis_per_batch: (i + 1) * self._vis_per_batch, :] = \
                pred_and_prob

            labels_and_prob_in_labels_array[i * self._vis_per_batch: (i + 1) * self._vis_per_batch, :] = \
                labels_and_prob_in_labels

        idx = np.random.choice(range(num_elem), self._num_visual, replace=False)
        images_array = images_array[idx, :]
        pred_and_prob_array = pred_and_prob_array[idx, :]
        labels_and_prob_in_labels_array = labels_and_prob_in_labels_array[idx, :]

        filenames = [Path(filenames[idx_curr]).stem for idx_curr in idx]

        height_fig = self._num_visual
        width_fig = 3
        height_cell = 0.95 * (height_fig / self._num_visual) / height_fig
        width_im_cell = 1 / width_fig
        left_im_cell = 0 / width_fig
        width_txt_cell = 1.8 / width_fig
        left_txt_cell = 1.1 / width_fig
        bottom_cell = [x / height_fig for x in range(height_fig)]

        fig = plt.figure(figsize=(width_fig, height_fig), tight_layout=False)

        for k in range(self._num_visual):
            fig.add_axes((left_im_cell, bottom_cell[k], width_im_cell, height_cell))
            plt.axis('off')
            plt.imshow(images_array[k, :], aspect='auto')

            fig.add_axes((left_txt_cell, bottom_cell[k], width_txt_cell, height_cell))
            str_pic = f'{filenames[k]} \n' \
                f'gt: {round(labels_and_prob_in_labels_array[k, 1], 2)}\n' \
                f'({int(labels_and_prob_in_labels_array[k, 0])}) ' \
                f'{self._labels_num2txt[labels_and_prob_in_labels_array[k, 0]]}\n' \
                f'pred: {round(pred_and_prob_array[k, 1], 2)}\n' \
                f'({int(pred_and_prob_array[k, 0])}) ' \
                f'{self._labels_num2txt[pred_and_prob_array[k, 0]]}\n' \

            plt.text(0, 0.5, str_pic, verticalalignment='center')
            plt.axis('off')

        self._writer.add_figure(txt, fig, self._curr_epoch)

        plt.close(fig)

    def _visual_confusion_and_hists(self, labels_pred_list: List[np.ndarray]) -> None:
        labels = np.zeros(len(self._test_set))
        pred = np.zeros(len(self._test_set))

        k_batch = 0

        for labels_pred in labels_pred_list:
            diap = min((self._batch_size, np.shape(labels_pred)[0]))
            labels[k_batch * self._batch_size: k_batch * self._batch_size + diap] = labels_pred[:, 0]
            pred[k_batch * self._batch_size: k_batch * self._batch_size + diap] = labels_pred[:, 1]

            k_batch += 1

        confusion_matrix_array = confusion_matrix(y_pred=pred, y_true=labels).astype(float)

        confusion_matrix_array /= 50

        fig = plt.figure(figsize=(12, 12))
        conf_map = plt.imshow(confusion_matrix_array, cmap="gist_heat", interpolation="nearest")
        plt.colorbar(mappable=conf_map)
        self._writer.add_figure('Confusion_matrix', fig, self._curr_epoch)

        plt.close(fig)

        self._visual_hists(confusion_matrix_array)

    def _visual_hists(self, confusion_matrix_array: np.ndarray) -> None:
        num_col = 20
        correct = np.diag(confusion_matrix_array) * 100

        idx_correct = np.argsort(correct)

        idx_best = idx_correct[-1: -num_col - 1: -1]
        idx_worst = idx_correct[0: num_col]

        self._visual_hist(np.copy(correct[idx_best]), idx_best,
                          ylabel='Correct predicts, %',
                          title='Best predicts',
                          tag='Best_predicts_hist')

        self._visual_hist(np.copy(correct[idx_worst]), idx_worst,
                          ylabel='Correct predicts, %',
                          title='Worst predicts',
                          tag='Worst_predicts_hist')

    def _visual_hist(self, data: np.ndarray, labels: np.ndarray, ylabel: str, title: str, tag: str) -> None:
        num_col = np.size(data)
        len_txt = 20
        xlim = (0, num_col + 1)
        ylim = (0, max(np.max(data) * 1.1, 1e-5))

        fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
        ax.set(ylabel=ylabel, ylim=ylim,
               xlim=xlim)

        labels_on_graph = []
        for k in range(np.size(labels)):
            str_tick = self._labels_num2txt[labels[k]]
            len_str_tick = len(str_tick)
            if len_str_tick < len_txt:
                str_tick = str_tick + ' ' * (len_str_tick - len_txt)
            elif len_str_tick > len_txt:
                str_tick = str_tick[0: len_txt]

            labels_on_graph.append(str_tick)

        for i in range(num_col):
            val = data[i]
            ax.text(i + 1, val + ylim[1] * 0.01, np.round(val).astype(int), horizontalalignment='center')
            ax.vlines(x=i + 1, ymin=0, ymax=val, color='firebrick', alpha=0.7, linewidth=20)

        plt.xticks(range(1, num_col + 1), labels_on_graph, rotation=-90)
        plt.title(title)
        fig.tight_layout()

        self._writer.add_figure(tag, fig, self._curr_epoch)

        plt.close(fig)


def smooth_one_hot(labels: torch.Tensor, num_classes: int, smoothing=0.0) -> torch.Tensor:
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((labels.size(0), num_classes))
    with torch.no_grad():
        smoothed_labels = torch.empty(size=label_shape, device=labels.device)
        smoothed_labels.fill_(smoothing / (num_classes - 1))
        smoothed_labels.scatter_(1, labels.unsqueeze(1), confidence)
    return smoothed_labels




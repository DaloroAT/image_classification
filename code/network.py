from typing import Any, List, Tuple
from math import ceil

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt

from resnet import resnet18
from datasets import basic_transforms


def resnet18_num_classes(pretrained: bool, num_classes: int, p_drop: float, type_net: str) -> nn.Module:
    if type_net == 'custom':
        model = resnet18(pretrained=pretrained)
    elif type_net == 'classic':
        model = models.resnet18(pretrained=pretrained)
    else:
        raise ValueError('Unsupported type net')

    inp_features = model.fc.in_features

    model.fc = nn.Sequential(nn.Dropout(p=p_drop), nn.Linear(inp_features, num_classes))

    return model


class Classifier(nn.Module):
    _net: nn.Module

    def __init__(self, net: nn.Module):
        super(Classifier, self).__init__()
        self._net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._net(x)
        return x

    def save(self, path: Path, meta: Any) -> None:
        checkpoint = {
            'state_dict': self._net.state_dict(),
            'meta': meta
        }
        torch.save(checkpoint, path)

    def load(self, path: Path) -> Any:
        device = next(self._net.parameters()).device
        checkpoint = torch.load(path, map_location=device)
        self._net.load_state_dict(checkpoint['state_dict'])
        meta = checkpoint['meta']
        return meta

    def classify_and_gradcam_by_path(self, path: Path, target_class: int = None) -> None:
        device = next(self._net.parameters()).device
        prev_train_mode = self._net.training

        transform = basic_transforms()
        self._net.eval()
        image = Image.open(path).convert('RGB')
        image_tensor = transform(image).to(device).unsqueeze(0)
        self._net.zero_grad()
        prediction = self._net(image_tensor, prepare_gradcam=True)
        pred_class = prediction.argmax(dim=1).item()

        if target_class is None:
            class_for_backward = pred_class
        else:
            class_for_backward = target_class

        prediction[:, class_for_backward].backward()

        gradients, outputs = self._net.get_data_for_gradcam()

        heatmap = _prepare_gradcam_map((64, 64), gradients, outputs)

        fig_vis = _visualize_classifying_and_gradcam(image, heatmap, target_class, pred_class)

        self._net.train(prev_train_mode)

        return fig_vis


def _visualize_classifying_and_gradcam(image: Image, heatmap: np.ndarray, target_class: int, pred_class: int):
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    fig.add_subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(heatmap)
    plt.axis('off')
    if target_class is None:
        plt.figtext(0.5, 0.9, f'Predicted class: {pred_class}', ha='center')
    else:
        plt.figtext(0.5, 0.9, f'Target class: {target_class}, Predicted class: {pred_class}', ha='center')
    plt.show()
    return fig


def _prepare_gradcam_map(size: Tuple[int, int], gradients: torch.Tensor, outputs: torch.Tensor) -> np.ndarray:
    mean_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(mean_gradients.size()[0]):
        outputs[:, i, :, :] *= mean_gradients[i]

    heatmap = torch.mean(outputs, dim=1).squeeze()
    heatmap = torch.nn.functional.relu_(heatmap)
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.cpu().numpy()

    num_row, num_col = np.shape(heatmap)

    func_interp = interpolate.interp2d(np.linspace(1, num_row, num_row),
                                       np.linspace(1, num_col, num_col),
                                       heatmap, kind='cubic')

    heatmap = func_interp(np.linspace(1, num_row, size[0]), np.linspace(1, num_col, size[1]))

    cm = plt.get_cmap('jet')
    heatmap = cm(heatmap)
    heatmap[:, :, 3] = 0.3
    return heatmap

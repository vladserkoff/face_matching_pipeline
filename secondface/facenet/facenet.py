# pylint: disable = R0914
"""
Facenet implementation adopted from
https://github.com/furkanu/deeplearning.ai-pytorch/tree/master/4-%20Convolutional%20Neural%20Networks/Week%204/Face%20Recognition%20(Done)
"""

import zipfile
from typing import Dict, List

import numpy as np
import torch

from secondface.facenet.inception import Inception

WEIGHTS = [
    'conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3', 'inception_3a_1x1_conv',
    'inception_3a_1x1_bn', 'inception_3a_pool_conv', 'inception_3a_pool_bn',
    'inception_3a_5x5_conv1', 'inception_3a_5x5_conv2', 'inception_3a_5x5_bn1',
    'inception_3a_5x5_bn2', 'inception_3a_3x3_conv1', 'inception_3a_3x3_conv2',
    'inception_3a_3x3_bn1', 'inception_3a_3x3_bn2', 'inception_3b_3x3_conv1',
    'inception_3b_3x3_conv2', 'inception_3b_3x3_bn1', 'inception_3b_3x3_bn2',
    'inception_3b_5x5_conv1', 'inception_3b_5x5_conv2', 'inception_3b_5x5_bn1',
    'inception_3b_5x5_bn2', 'inception_3b_pool_conv', 'inception_3b_pool_bn',
    'inception_3b_1x1_conv', 'inception_3b_1x1_bn', 'inception_3c_3x3_conv1',
    'inception_3c_3x3_conv2', 'inception_3c_3x3_bn1', 'inception_3c_3x3_bn2',
    'inception_3c_5x5_conv1', 'inception_3c_5x5_conv2', 'inception_3c_5x5_bn1',
    'inception_3c_5x5_bn2', 'inception_4a_3x3_conv1', 'inception_4a_3x3_conv2',
    'inception_4a_3x3_bn1', 'inception_4a_3x3_bn2', 'inception_4a_5x5_conv1',
    'inception_4a_5x5_conv2', 'inception_4a_5x5_bn1', 'inception_4a_5x5_bn2',
    'inception_4a_pool_conv', 'inception_4a_pool_bn', 'inception_4a_1x1_conv',
    'inception_4a_1x1_bn', 'inception_4e_3x3_conv1', 'inception_4e_3x3_conv2',
    'inception_4e_3x3_bn1', 'inception_4e_3x3_bn2', 'inception_4e_5x5_conv1',
    'inception_4e_5x5_conv2', 'inception_4e_5x5_bn1', 'inception_4e_5x5_bn2',
    'inception_5a_3x3_conv1', 'inception_5a_3x3_conv2', 'inception_5a_3x3_bn1',
    'inception_5a_3x3_bn2', 'inception_5a_pool_conv', 'inception_5a_pool_bn',
    'inception_5a_1x1_conv', 'inception_5a_1x1_bn', 'inception_5b_3x3_conv1',
    'inception_5b_3x3_conv2', 'inception_5b_3x3_bn1', 'inception_5b_3x3_bn2',
    'inception_5b_pool_conv', 'inception_5b_pool_bn', 'inception_5b_1x1_conv',
    'inception_5b_1x1_bn', 'dense_layer'
]

CONV_SHAPE = {
    'conv1': [64, 3, 7, 7],
    'conv2': [64, 64, 1, 1],
    'conv3': [192, 64, 3, 3],
    'inception_3a_1x1_conv': [64, 192, 1, 1],
    'inception_3a_pool_conv': [32, 192, 1, 1],
    'inception_3a_5x5_conv1': [16, 192, 1, 1],
    'inception_3a_5x5_conv2': [32, 16, 5, 5],
    'inception_3a_3x3_conv1': [96, 192, 1, 1],
    'inception_3a_3x3_conv2': [128, 96, 3, 3],
    'inception_3b_3x3_conv1': [96, 256, 1, 1],
    'inception_3b_3x3_conv2': [128, 96, 3, 3],
    'inception_3b_5x5_conv1': [32, 256, 1, 1],
    'inception_3b_5x5_conv2': [64, 32, 5, 5],
    'inception_3b_pool_conv': [64, 256, 1, 1],
    'inception_3b_1x1_conv': [64, 256, 1, 1],
    'inception_3c_3x3_conv1': [128, 320, 1, 1],
    'inception_3c_3x3_conv2': [256, 128, 3, 3],
    'inception_3c_5x5_conv1': [32, 320, 1, 1],
    'inception_3c_5x5_conv2': [64, 32, 5, 5],
    'inception_4a_3x3_conv1': [96, 640, 1, 1],
    'inception_4a_3x3_conv2': [192, 96, 3, 3],
    'inception_4a_5x5_conv1': [32, 640, 1, 1],
    'inception_4a_5x5_conv2': [64, 32, 5, 5],
    'inception_4a_pool_conv': [128, 640, 1, 1],
    'inception_4a_1x1_conv': [256, 640, 1, 1],
    'inception_4e_3x3_conv1': [160, 640, 1, 1],
    'inception_4e_3x3_conv2': [256, 160, 3, 3],
    'inception_4e_5x5_conv1': [64, 640, 1, 1],
    'inception_4e_5x5_conv2': [128, 64, 5, 5],
    'inception_5a_3x3_conv1': [96, 1024, 1, 1],
    'inception_5a_3x3_conv2': [384, 96, 3, 3],
    'inception_5a_pool_conv': [96, 1024, 1, 1],
    'inception_5a_1x1_conv': [256, 1024, 1, 1],
    'inception_5b_3x3_conv1': [96, 736, 1, 1],
    'inception_5b_3x3_conv2': [384, 96, 3, 3],
    'inception_5b_pool_conv': [96, 736, 1, 1],
    'inception_5b_1x1_conv': [256, 736, 1, 1],
}


class FaceNet:
    """
    Face embedding model based on
    https://github.com/davidsandberg/facenet
    """

    def __init__(self, weights_path: str) -> None:
        self.weights_path = weights_path
        self.model = Inception()
        self._state_dict = {}  # type: Dict[str, torch.Tensor]

    def load_weights(self) -> None:
        """
        Load model weights from zipfile into the model object
        """
        weights = _read_weights(self.weights_path)
        state_dict_keys = self.model.state_dict().keys()
        convolutions = {
            x: x.replace('.', '_')
            for x in state_dict_keys if 'conv' in x
        }
        batch_norms = {
            x: x.replace('.', '_')
            for x in state_dict_keys if 'bn' in x
        }
        dense = {
            x: x.replace('.', '_')
            for x in state_dict_keys if 'dense' in x
        }

        _fill_state_dict(weights, convolutions, batch_norms, dense,
                         self._state_dict)
        self.model.load_state_dict(self._state_dict, strict=True)

    def __call__(self, image: np.ndarray):
        return self.model(image)


def _read_weights(weights_path: str) -> Dict[str, List]:
    """
    Read model parameters from zipped archive
    """

    def _zip_to_np(zip_file: zipfile.ZipFile, name: str) -> np.ndarray:
        return np.fromstring(zip_file.read(name), sep=',', dtype=None)

    weights_zip = zipfile.ZipFile(weights_path)

    files = weights_zip.namelist()[1:]
    paths = {}
    weights = {}

    for file in files:
        filename = file.split('/')[-1]
        paths[filename.replace('.csv', '')] = file

    for name in WEIGHTS:
        if 'conv' in name:
            conv_w = _zip_to_np(weights_zip, paths[name + '_w'])
            conv_w = np.reshape(conv_w, CONV_SHAPE[name])
            conv_b = _zip_to_np(weights_zip, paths[name + '_b'])
            weights[name] = [conv_w, conv_b]
        elif 'bn' in name:
            bn_w = _zip_to_np(weights_zip, paths[name + '_w'])
            bn_b = _zip_to_np(weights_zip, paths[name + '_b'])
            bn_m = _zip_to_np(weights_zip, paths[name + '_m'])
            bn_v = _zip_to_np(weights_zip, paths[name + '_v'])
            weights[name] = [bn_w, bn_b, bn_m, bn_v]
        elif 'dense' in name:
            dense_w = _zip_to_np(weights_zip, paths['dense_w'])
            dense_w = np.reshape(dense_w, (128, 736))
            dense_b = _zip_to_np(weights_zip, paths['dense_b'])
            weights[name] = [dense_w, dense_b]

    return weights


def _fill_state_dict(weights: Dict[str, List], convolutions: Dict[str, str],
                     batch_norms: Dict[str, str], dense: Dict[str, str],
                     state_dict: Dict[str, torch.Tensor]) -> None:
    for name in weights.keys():
        if 'conv' in name:
            _fill_conv(weights, state_dict, name, convolutions)
        elif 'bn' in name:
            _fill_bn(weights, state_dict, name, batch_norms)
        elif 'dense' in name:
            _fill_dense(weights, state_dict, name, dense)


def _fill_conv(weights: Dict[str, List], state_dict: Dict[str, torch.Tensor],
               weight_key: str, convolutions: Dict[str, str]) -> None:
    conv_weight, conv_bias = weights[weight_key]
    keys = _find_state_dict_keys(convolutions, weight_key)
    state_dict[keys[0]] = torch.from_numpy(conv_weight)
    state_dict[keys[1]] = torch.from_numpy(conv_bias)


def _fill_bn(weights: Dict[str, List], state_dict: Dict[str, torch.Tensor],
             weight_key: str, batch_norms: Dict[str, str]) -> None:
    weight, bias, mean, variance = weights[weight_key]
    keys = _find_state_dict_keys(batch_norms, weight_key)
    state_dict[keys[0]] = torch.from_numpy(weight)
    state_dict[keys[1]] = torch.from_numpy(bias)
    state_dict[keys[2]] = torch.from_numpy(mean)
    state_dict[keys[3]] = torch.from_numpy(variance)


def _fill_dense(weights: Dict[str, List], state_dict: Dict[str, torch.Tensor],
                weight_key: str, dense: Dict[str, str]) -> None:
    weight, bias = weights[weight_key]
    keys = [key for key, val in dense.items() if weight_key in val]
    state_dict[keys[0]] = torch.from_numpy(weight)
    state_dict[keys[1]] = torch.from_numpy(bias)


def _find_state_dict_keys(layer_dict: Dict[str, str], weight_key: str) -> List:
    keys = []
    for key, val in layer_dict.items():
        if (weight_key.startswith('in') and not val.startswith('bl')) or (
                not weight_key.startswith('in') and val.startswith('bl')):
            continue
        if weight_key in val:
            keys.append(key)

    return keys

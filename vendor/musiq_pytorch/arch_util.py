"""Minimal MUSIQ helpers adapted from IQA-PyTorch, with network loading disabled.
Required Notice: Copyright (c) 2022 Chaofeng Chen
License: PolyForm Noncommercial License 1.0.0; see LICENSE.
"""
from pathlib import Path
import torch


def dist_to_mos(dist_score):
    num_classes = dist_score.shape[-1]
    mos_score = dist_score * torch.arange(1, num_classes + 1).to(dist_score)
    return mos_score.sum(dim=-1, keepdim=True)


def get_url_from_name(name):
    # Compatibility only: constructing this string performs no network request.
    return 'https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/' + name


def load_pretrained_network(net, model_path, strict=True):
    path = Path(model_path)
    if str(model_path).startswith(('https://', 'http://')) or not path.is_file():
        raise ValueError('MUSIQ subset permits verified local weights only; network loading is disabled')
    state = torch.load(path, map_location='cpu', weights_only=True)
    cleaned = {key[7:] if key.startswith('module.') else key: value for key, value in state.items()}
    net.load_state_dict(cleaned, strict=strict)

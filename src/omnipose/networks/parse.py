"""Parse pretrained-model filenames for residual / style / concatenation flags."""

import os


def parse_model_string(pretrained_model):
    if isinstance(pretrained_model, list):
        model_str = os.path.split(pretrained_model[0])[-1]
    else:
        model_str = os.path.split(pretrained_model)[-1]
    if len(model_str) > 3 and model_str[:4] == 'unet':
        nclasses = max(2, int(model_str[4]))
    elif len(model_str) > 7 and model_str[:8] == 'cellpose':
        nclasses = 3
    else:
        return True, True, False
    ostrs = model_str.split('_')[2::2]
    residual_on = ostrs[0] == 'on'
    style_on = ostrs[1] == 'on'
    concatenation = ostrs[2] == 'on'
    return residual_on, style_on, concatenation

# SPDX-License-Identifier: GPL-3.0-only
# LV-Vis — Large Volume LOD Visualization System
# Copyright (c) Hsu I Chieh

import numpy as np
import itertools

def layer_2_dict(vol_size):
    output = {}
    cube = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

    for i in range(0, 8):
        for j in range(0, 8):
            for k in range(0, 8):
                m = cube[i][0] * 4 + cube[j][0] * 2 + cube[k][0]
                n = cube[i][1] * 4 + cube[j][1] * 2 + cube[k][1]
                l = cube[i][2] * 4 + cube[j][2] * 2 + cube[k][2]
                output[f'0_{i}/1_{j}/2_{k}'] = (cube[i] * 4 + cube[j] * 2 + cube[k]) * 2 * vol_size
    return output


def layer_1_dict(vol_size):
    output = {}
    cube = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

    for i in range(0, 8):
        for j in range(0, 8):
            output[f'0_{i}/1_{j}'] = (cube[i] * 2 + cube[j]) * vol_size * 2 * 2
    return output


def layer_0_dict(vol_size):
    output = {}
    cube = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    for i in range(0, 8):
        output[f'0_{i}'] = (cube[i]) * vol_size * 4 * 2
    return output

def dict_corners(dicts, vol_size, factor):
    output_dict = {}
    cube = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    for key, value in dicts.items():
        start = value
        corners = start + cube * vol_size * factor
        output_dict[key] = corners
        # print(corners)

    return output_dict


def layer_3_dict(vol_size):
    output = {}
    cube = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                     [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

    for i in range(8):  # Layer 0
        for j in range(8):  # Layer 1
            for k in range(8):  # Layer 2
                for l in range(8):  # Layer 3
                    key = f'0_{i}/1_{j}/2_{k}/3_{l}'
                    offset = (cube[i] * 8 + cube[j] * 4 + cube[k] * 2 + cube[l]) * vol_size
                    output[key] = offset
    return output


def layer_n_dict(vol_size: np.array, level: int, max_level) -> dict:
    output = {}
    cube = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                     [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

    # level=3 → (i0, i1, i2)
    for indices in itertools.product(range(8), repeat=level):
        key = "/".join([f"{l}_{idx}" for l, idx in enumerate(indices)])

        offset = np.zeros(3, dtype=int)
        for l, idx in enumerate(indices):
            offset += cube[idx] * (2 ** (level - l - 1))

        offset *= vol_size * (2 ** (max_level-level))
        output[key] = offset

    return output

if __name__ == "__main__":
        # print(layer_3_dict([100, 125, 150]))
        # print('===========')
        # print(layer_2_dict([100, 125, 150]))
        # print(layer_n_dict(np.array([100, 125, 150]), 3, 5))
        # print('===========')
        # print(layer_1_dict([100, 125, 150]))
        # print(layer_n_dict(np.array([100, 125, 150]), 2, 5))
        #
        # print('===========')
        # print(layer_0_dict([100, 125, 150]))
        # print(layer_n_dict(np.array([100, 125, 150]), 1, 5))

        print(layer_n_dict(np.array([100, 125, 150]), 0, 5))


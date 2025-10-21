# SPDX-License-Identifier: GPL-3.0-only
# LV-Vis — Large Volume LOD Visualization System
# Copyright (c) Hsu I Chieh

import os
import numpy as np


def _load_block_memmap(file_dir: str, vol_path: str):
    if not vol_path or vol_path == "no_vol":
        return None
    last = os.path.basename(vol_path.rstrip("\\/"))
    fp = os.path.join(file_dir, vol_path, f"{last}.npy")
    fp = os.path.normpath(fp)
    if not os.path.exists(fp):
        return None
    return np.load(fp, mmap_mode="r")


def _infer_out_dtype(file_dir: str, vol_paths, fallback=None):
    dt = None
    for vp in vol_paths:
        arr = _load_block_memmap(file_dir, vp)
        if arr is None:
            continue
        dt = arr.dtype if dt is None else np.result_type(dt, arr.dtype)
        del arr
    return dt if dt is not None else fallback


def extract_volume_from_next_layer(self, center, vol_global_start_point, vol_global_end_point, file_dir, layer):
    layer_dict = self.layer_dict[layer]
    layer_dict_corners = self.layer_dict_corners[layer]
    factor = self.factor[layer]
    print('[vol_extract] factor, layer', factor, layer)
    # center = np.minimum(np.maximum([0, 0, 0], center), self.vol_size)
    print("[vol_extract] global_start:", self.vol_global_start_point[layer] )
    center_correction = vol_global_start_point[layer] + np.array(center) * factor
    print(center_correction)
    vol_size = np.array(self.vol_size)
    print("[vol_extract] vol size:" , vol_size)
    # boundary = [[0,0,0], vol_size * 8]
    cube = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    # print("[vol_extract] layer dict", layer_dict)
    for key, value in layer_dict.items():
        start = value
        end = value + np.array(vol_size * (factor / 2))
        if (center_correction >= start).all() and (center_correction <= end).all():
            print("[vol_extract] center in",key, value)

            corners = layer_dict_corners[key]
            # print(corners)
            distances = []
            for corner in corners:
                distance = np.sqrt(
                    (corner[0] - center_correction[0]) ** 2 + (corner[1] - center_correction[1]) ** 2 + (
                                corner[2] - center_correction[2]) ** 2)
                distances.append(distance)
                # print(distance)
            # print(np.min(distances))
            closest_corner_id = np.argmin(distances)
            closest_corner = corners[closest_corner_id]
            used_vol = [key for key, value in layer_dict_corners.items() if
                        any((cor == closest_corner).all() for cor in value)]

            used_value = {}
            for vol in used_vol:
                used_value[vol] = layer_dict[vol]


            print("[vol_extract] next vol use: ",used_vol)
            print("[vol_extract] used_value:", used_value)
            min_start_vol = min(used_vol, key=lambda vol: tuple(layer_dict[vol]))

            print("[vol_extract] min_start_vol:",min_start_vol)

            load_vol_size = vol_size * (factor/ 2)
            print("[vol_extract] next vol size",load_vol_size)

            exam_exist = layer_dict[min_start_vol] + cube * load_vol_size
            print("[vol_extract] exam:\n", exam_exist)
            expand_used_vol = []

            for i in range(8):
                append_cnt = 0
                for key, value in used_value.items():
                    if (exam_exist[i] == value).all():
                        expand_used_vol.append(key)
                        append_cnt += 1
                        break
                if append_cnt == 0:
                    expand_used_vol.append("no_vol")

            print('[vol_extract] expand_used_vol', expand_used_vol)

            eight_dirc_form_center = np.array(
                [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
            #
            # eight_way = center_correction + load_vol_size * eight_dirc_form_center // 2

            half = (np.asarray(load_vol_size, dtype=np.int64) // 2)
            eight_way = np.asarray(center_correction, dtype=np.int64) + eight_dirc_form_center * half
            print("[vol_extract] eight_way\n", eight_way)
            # vol_start = eight_way[0]
            # vol_end = eight_way[-1]
            starts = np.stack([layer_dict[v] for v in used_vol], axis=0).astype(np.int64)  # (k,3), ZYX
            tile_sz = np.asarray(load_vol_size, dtype=np.int64)
            block_min = starts.min(axis=0)
            block_max = (starts + tile_sz).max(axis=0)
            print("[vol_extract] block_min block_max\n", block_min, block_max)
            vol_start = np.maximum(eight_way[0], block_min)
            vol_end = np.minimum(eight_way[-1],  block_max)
            print("[vol_extract] volume start end position:",vol_start, vol_end)

            vol_relative_start = (vol_start - layer_dict[min_start_vol]).astype(np.int64)
            vol_relative_end = (vol_relative_start + load_vol_size).astype(np.int64)
            print("[vol_extract] start from: ",min_start_vol)
            print("[vol_extract] volume relative start end position:", vol_relative_start, vol_relative_end)
            print("==============")

            print("[vol_extract] file dir", file_dir)

            denom = max(1, int(factor // 2))
            real_start = (vol_relative_start // denom).astype(np.int64)

            fallback_dt =  None
            out_dtype = _infer_out_dtype(file_dir, expand_used_vol, fallback=fallback_dt)
            if out_dtype is None:
                out_dtype = np.float32

            out = np.zeros(tuple(vol_size.tolist()), dtype=out_dtype)
            out0 = real_start
            out1 = real_start + vol_size

            for j, vol_path in enumerate(expand_used_vol):
                arr = _load_block_memmap(file_dir, vol_path)
                if arr is None:
                    continue
                iz = (j >> 2) & 1
                iy = (j >> 1) & 1
                ix = (j >> 0) & 1
                az, ay, ax = arr.shape
                b0 = np.array([iz * az, iy * ay, ix * ax], dtype=np.int64)
                b1 = b0 + np.array([az, ay, ax], dtype=np.int64)
                s0 = np.maximum(out0, b0)
                s1 = np.minimum(out1, b1)
                if np.any(s1 <= s0):
                    del arr
                    continue

                src0 = (s0 - b0).astype(int)
                dst0 = (s0 - out0).astype(int)
                sz = (s1 - s0).astype(int)
                src = (slice(src0[0], src0[0] + sz[0]),
                       slice(src0[1], src0[1] + sz[1]),
                       slice(src0[2], src0[2] + sz[2]))
                dst = (slice(dst0[0], dst0[0] + sz[0]),
                       slice(dst0[1], dst0[1] + sz[1]),
                       slice(dst0[2], dst0[2] + sz[2]))

                if arr.dtype == out_dtype:
                    out[dst] = arr[src]
                else:
                    out[dst] = np.asarray(arr[src], dtype=out_dtype)
                del arr

            print('[vol_extract] filled ROI directly, shape', out.shape, 'dtype', out.dtype)
            vol_global_start_point[layer + 1] = vol_start
            vol_global_end_point[layer + 1] = vol_end
            print('[vol_extract] out intensity', out.max(), out.min())
            return out, vol_global_start_point, vol_global_end_point


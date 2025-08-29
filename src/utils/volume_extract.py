from vispy.visuals.transforms import STTransform

import os
import numpy as np


def concatenate_arrays(array1, array2, axis):
    array = np.concatenate((array1, array2), axis=axis) if np.any(array1) and np.any(array2) \
        else array1 if np.any(array1) else array2 if np.any(array2) else [0]
    return array

def concatenate_8dim(idx_volume_sec):
    intermediate_volumes = [[0]] * 4
    for j in range(0, 8, 2):
        intermediate_volumes[j // 2] = concatenate_arrays(idx_volume_sec[j], idx_volume_sec[j + 1], axis=2)

    vol_0123 = concatenate_arrays(intermediate_volumes[0], intermediate_volumes[1], axis=1)
    vol_4567 = concatenate_arrays(intermediate_volumes[2], intermediate_volumes[3], axis=1)
    final_volume = concatenate_arrays(vol_0123, vol_4567, axis=0)

    return final_volume

def save_layer_volumes(self):
    # Save current volumes before moving to the next layer
    self.temp_volumes1[self.layer] = self.volume1
    self.temp_volumes2[self.layer] = self.volume2
    print("temp_volumes1",self.temp_volumes1[self.layer])



def extract_volume_from_next_layer(self, center, vol_global_start_point, file_dir, layer):
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
    boundary = [[0,0,0], vol_size * 8]
    cube = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    print("[vol_extract] layer dict", layer_dict)
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

            # print(min_start_vol)

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
            eight_way = center_correction + load_vol_size * eight_dirc_form_center // 2
            print("[vol_extract] eight_way\n", eight_way)
            vol_start = np.maximum(eight_way[0], boundary[0])
            vol_end = np.minimum(eight_way[-1],  boundary[1])
            print("[vol_extract] volume start end position:",vol_start, vol_end)

            vol_relative_start = (vol_start - layer_dict[min_start_vol]).astype(np.int64)
            vol_relative_end = (vol_relative_start + load_vol_size).astype(np.int64)
            print("[vol_extract] start from: ",min_start_vol)
            print("[vol_extract] volume relative start end position:", vol_relative_start, vol_relative_end)
            print("==============")
            volumes = []
            print("[vol_extract] file dir", file_dir)
            for volume_path in expand_used_vol:
                if volume_path != "no_vol":
                    volume_path_last = volume_path.split("/")[-1]
                    file_path = os.path.join(file_dir, f"{volume_path}/{volume_path_last}.npy")
                    if os.path.exists(file_path):
                        print(file_path)
                        # idx_volume_sec.append(np.load(file_path).astype(np.float32))
                        # volumes.append((np.load(file_path) * 127).astype(np.int8))
                        volumes.append(np.load(file_path))
                else:
                    volumes.append([0])


            cat_volume = concatenate_8dim(volumes)
            print('[vol_extract] load vols', cat_volume.shape)


            real_vol_relative_start = vol_relative_start // (factor // 2)
            # real_vol_relative_end = vol_relative_end // (factor // 2)
            # array = [[real_vol_relative_start[0], real_vol_relative_end[0]],
            #     [real_vol_relative_start[1], real_vol_relative_end[1]],
            #     [real_vol_relative_start[2], real_vol_relative_end[2]]
            # ]
            # print(array)

            outputVol = cat_volume[
                        real_vol_relative_start[0]: real_vol_relative_start[0]+vol_size[0],
                        real_vol_relative_start[1]: real_vol_relative_start[1]+vol_size[1],
                        real_vol_relative_start[2]: real_vol_relative_start[2]+vol_size[2]
                             ]
            outputVol = ((outputVol - outputVol.min()) / (
                    outputVol.max() - outputVol.min())).astype(np.float32)
            vol_global_start_point[layer+1] = vol_start
            return outputVol, vol_global_start_point


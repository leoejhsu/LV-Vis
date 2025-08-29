


def layer_1_table():
    output = []
    for i in range(0, 8, 2):
        for j in range(0, 8, 2):
            output.append(f'0_{i}/1_{j}')
            output.append(f'0_{i}/1_{j + 1}')
            output.append(f'0_{i + 1}/1_{j}')
            output.append(f'0_{i + 1}/1_{j + 1}')

    return output


def layer_2_table():
    output = []
    for i in range(0, 8, 2):
        for j in range(0, 8, 2):
            for k in range(0, 8, 2):
                output.append(f'0_{i}/1_{j}/2_{k}')
                output.append(f'0_{i}/1_{j}/2_{k + 1}')
                output.append(f'0_{i}/1_{j + 1}/2_{k}')
                output.append(f'0_{i}/1_{j + 1}/2_{k + 1}')
                output.append(f'0_{i + 1}/1_{j}/2_{k}')
                output.append(f'0_{i + 1}/1_{j}/2_{k + 1}')
                output.append(f'0_{i + 1}/1_{j + 1}/2_{k}')
                output.append(f'0_{i + 1}/1_{j + 1}/2_{k + 1}')

    return output



def fit_blocks(total_size, basic_size, block_order):

    result = []

    x_index = 0
    y_index = 0
    z_index = 0

    for block in block_order:
        offset_x = x_index * basic_size[2]
        offset_y = y_index * basic_size[1]
        offset_z = z_index * basic_size[0]
        result.append((block, offset_z, offset_y, offset_x))
        x_index += 1
        if x_index * basic_size[2] >= total_size[2]:
            x_index = 0
            y_index += 1
        if y_index * basic_size[1] >= total_size[1]:
            y_index = 0
            z_index += 1

    return result


def block_order(id_list, table1, table2, num):
    if num == 1:
        result = [i for i in table1 if i in id_list]
    elif num == 2:
        result = [i for i in table2 if i in id_list]
    else:
        result = id_list

    return result
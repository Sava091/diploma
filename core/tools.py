import numpy as np
import os


T_MIN = 200 #200 miliseconds - minimal RR
T_MAX = 2000 #2000 miliseconds - maximal RR
N_CELLS= 60 #quantity of cells
N_RANGES = 10 #quantity of ranges


def rr_reader(filename):
    fp = open(filename,'rb')
    hd = fp.read(21)
    rr = fp.read(2)
    array_rr = []
    while rr:
        r = int.from_bytes(rr, byteorder='big')
        if r > 100:
            array_rr.append(r)
        rr = fp.read(2)
    fp.close()
    return array_rr


def beat_reader(filename):
    fp = open(filename,'r')
    array_rr = []
    for _ in range(2):
        fp.readline()
    for line in fp:
        items = line.split("\t")
        if len(items) < 5:
            continue
        t = items[1]
        if t != 'X':
            r = int(items[3])
            if r > 100:
                array_rr.append(r)
    fp.close()
    return array_rr

def ms_to_cell(t):
    return int(round((t - T_MIN)/T_MAX * N_CELLS))


def fill_rr_matrix(array_rr):
    rr_matrix = np.zeros((N_CELLS, N_CELLS), dtype=int)
    for x,y in zip(array_rr[:-1], array_rr[1:]):
        i,j = ms_to_cell(x), ms_to_cell(y)
        if i < 0 or j < 0 or i >= N_CELLS or j >= N_CELLS:
            continue
        rr_matrix[i,j] += 1
    return rr_matrix


def norm_rr_matrix(rr_matrix, n_scale=N_RANGES):
    s = rr_matrix.shape
    rr_array = rr_matrix.reshape((s[0] * s[1]))
    m = rr_array.max()
    # print(m)
    norm_rr_array = np.array(list(map(lambda v: v / m * n_scale, rr_array)))
    return norm_rr_array.reshape(s)


def load_rr_matrix(fn):
    array_rr = rr_reader(fn)
    matrix_rr = fill_rr_matrix(array_rr)
    norm_matrix_rr = norm_rr_matrix(matrix_rr)
    return norm_matrix_rr


def load_beat_matrix(fn) :
    array_rr = beat_reader(fn)
    matrix_rr = fill_rr_matrix(array_rr)
    norm_matrix_rr = norm_rr_matrix(matrix_rr)
    return norm_matrix_rr


def store_matrix_list(matrix_list, path):
    np.save(os.path.join(path, 'rr_matrixes.npy'), np.array(matrix_list))


def load_matrix_list(path):
    matrix_list = np.load(os.path.join(path, 'rr_matrixes.npy'))
    return matrix_list


def get_matrix_list_count(path):
    matrix_list = load_matrix_list(path)
    return matrix_list.shape[0]
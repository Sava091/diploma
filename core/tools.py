import numpy as np
import os
import tempfile
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


R_MIN = 0.2 #200 miliseconds - minimal RR
R_MAX = 1.9 #2000 miliseconds - maximal RR
N_CELLS= 60 #quantity of cells
N_RANGES = 10 #quantity of ranges
N_CLUSTERS = 5 #quantity of clusters
MODEL_FN = 'clustering_model.npy' #
CLUSTER_COLOURS = ['green', 'yellow', 'orange', 'brown', 'red']


def save_bin(bin, fn):
    fp = open(fn,'w')
    fp.write(bin)
    fp.close()


def save_temp(content, dir, prefix, suffix):
    tf = tempfile.NamedTemporaryFile(delete=False, dir=dir, prefix=prefix, suffix=suffix)
    tf.write(content)
    tf.flush()
    return os.path.basename(tf.name)


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


def ratio_to_cell(t1, t2):
    r = max(100, t1) / max(100, t2)
    return int(round((r - R_MIN)/(R_MAX-R_MIN) * N_CELLS))


def fill_rr_matrix(array_rr):
    rr_matrix = np.zeros((N_CELLS, N_CELLS), dtype=int)
    for x,y,z in zip(array_rr[:-2], array_rr[1:-1], array_rr[2:]):
        i,j = ratio_to_cell(x,y), ratio_to_cell(y,z)
        if i < 0 or j < 0 or i >= N_CELLS or j >= N_CELLS:
            continue
        rr_matrix[i,j] += 1
    return rr_matrix


def log_rr_matrix(rr_matrix, divider = 100):
    def log_rr(rri):
        return np.log((rri + 1)/divider)
    return np.vectorize(log_rr)(rr_matrix)


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
    log_matrix_rr = log_rr_matrix(matrix_rr)
    # norm_matrix_rr = norm_rr_matrix(matrix_rr)
    return log_matrix_rr


def load_beat_matrix(fn):
    array_rr = beat_reader(fn)
    matrix_rr = fill_rr_matrix(array_rr)
    log_matrix_rr = log_rr_matrix(matrix_rr)
    # norm_matrix_rr = norm_rr_matrix(log_matrix_rr)
    return log_matrix_rr


def store_matrix_list(matrix_list, path):
    np.save(os.path.join(path, 'rr_matrixes.npy'), np.array(matrix_list))


def store_heatmap(rr_matrix, path, fn):
    np.save(os.path.join(path, fn), rr_matrix)


def store_model(model, path, fn):
    fp = open(os.path.join(path, fn), 'wb')
    pickle.dump(model, fp)


def load_model(path, fn):
    fp = open(os.path.join(path, fn), 'rb')
    model = pickle.load(fp)
    return model


def load_matrix_list(path):
    matrix_list = np.load(os.path.join(path, 'rr_matrixes.npy'))
    return matrix_list


def load_heatmap(path, fn):
    rr_matrix = np.load(os.path.join(path, fn))
    return rr_matrix


def get_matrix_list_count(path):
    matrix_list = load_matrix_list(path)
    return matrix_list.shape[0]


def snail(m):
    w, h = m.shape
    w2, h2 = w//2, h//2
    x, y = w2+1, h2+1
    s = list()
    for i in range(w2):
        if i == 0:
            s.append(m[x,y])
            continue
        for k in range(i):
            s.append(m[x+k,y])
        for l in range(-i,i):
            s.append(m[x+i,y+l])
        for k in reversed(range(-i,i)):
            s.append(m[x+k,y+i])
        for l in reversed(range(-i,i)):
            s.append(m[x-i,y+l])
        for k in range(-i,0):
            s.append(m[x+k,y-i])
    return np.array(s)


def snail_map(w=N_CELLS, h=N_CELLS):
    s = list()
    y1,x1 = 0, 0
    y2,x2 = h-1, w-1
    while x1 < x2:
        for x in range(x1,x2+1):
            s.append((x,y1))
        for y in range(y1+1,y2):
            s.append((x2,y))
        for x in range(x2,x1,-1):
            s.append((x,y2))
        for y in range(y2,y1,-1):
            s.append((x1,y))
        x1 += 1
        x2 -= 1
        y1 += 1
        y2 -= 1
    return s


def reshape(m):
    return m.ravel()


def get_x_for_pca(matrix_list):
    s_map = snail_map(N_CELLS, N_CELLS)
    # print(s_map)
    return np.array([[m[i, j] for i, j in s_map] for m in matrix_list])


def pca_transform(x, n_components =2):
    x_norm = StandardScaler().fit_transform(x)
    pca = PCA(n_components=n_components)
    pc_x = pca.fit_transform(x_norm)
    return pc_x


def clustering(pc_x, n_clusters=N_CLUSTERS):
    model = KMeans(n_clusters=n_clusters)
    model.fit(pc_x)
    return model


def predict_cluster(model, pc_x):
    return model.predict(pc_x)


def make_group_map(range1, range2):
    return np.array([(pc1, pc2) for pc1 in range(range1[0], range1[1]) for pc2 in range(range2[0], range2[1])])
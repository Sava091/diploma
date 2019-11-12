import numpy as np
import os
import tempfile
import pickle
import seaborn as sn
import pandas as pd
import codecs
from matplotlib import pyplot as plt
from PIL import Image
from io import BytesIO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


R_MIN = 0.2 #200 miliseconds - minimal RR
R_MAX = 1.9 #2000 miliseconds - maximal RR
N_CELLS= 60 #quantity of cells
N_RANGES = 10 #quantity of ranges
N_CLUSTERS = 7 #quantity of clusters
CLUSTER_FN = 'clustering_model.npy' # model
PCA_FN = 'pca_model.npy' # model
SCALER_FN = 'scaler_model.npy' # model
CLUSTER_COLOURS = ['green', 'yellow', 'orange', 'brown', 'red', 'aqua', 'purple']
METRIC_GROUP_MAX = 10 # max groups


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
    fp = codecs.open(filename,'r', 'WINDOWS-1251')
    array_rr = []
    s1 = fp.readline()
    s2 = fp.readline()
    if s1.count('N') == 0:
        return array_rr
    if s2.count('---') < 3:
        return array_rr
    for line in fp:
        items = line.split("\t")
        if len(items) < 5:
            continue
        t = items[1]
        if t in ['N', 'S', 'V']:
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


def load_beat_matrix(fn, min_beats=1000):
    array_rr = beat_reader(fn)
    if len(array_rr) < min_beats: return None
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


def pca_fit(x, n_components =2):
    scaler = StandardScaler()
    scaler.fit(x)
    x_norm = scaler.transform(x)
    pca = PCA(n_components=n_components)
    pca.fit(x_norm)
    return pca, scaler


def pca_transform(scaler, pca, x):
    x_norm = scaler.transform(x)
    pc_x = pca.transform(x_norm)
    return pc_x


def clustering(pc_x, n_clusters=N_CLUSTERS):
    model = KMeans(n_clusters=n_clusters)
    model.fit(pc_x)
    return model


def predict_cluster(model, pc_x):
    return model.predict(pc_x)


def group_metric(pc_x, groups):
    return silhouette_score(pc_x, groups)


def make_group_map(range1, range2):
    return np.array([(pc1, pc2) for pc1 in range(range1[0], range1[1]) for pc2 in range(range2[0], range2[1])])


def get_heatmap_image(norm_matrix_rr):
    # df_matrix = pd.DataFrame(norm_matrix_rr, columns=np.linspace(R_MIN, R_MAX, N_CELLS))
    scale_rr = np.arange(0.25, R_MAX, 0.25)
    # scale_rr_rev = np.arange(R_MAX, 0.25, -0.25)
    svm = sn.heatmap(norm_matrix_rr, cmap='coolwarm', center=0, xticklabels=False, yticklabels=False)  # linecolor='white', linewidths=1,
    svm.invert_yaxis()
    ax2 = svm.twiny()
    ax2.xaxis.set_ticks_position('bottom')
    ax2.set_xlim(R_MIN, R_MAX)
    ax2.set_xticks(scale_rr)
    # ax3 = svm.twiny()
    # ax3.set_ylim(R_MIN, R_MAX)
    # ax3.set_yticks(scale_rr)
    figure = svm.get_figure()
    figure_data = BytesIO()
    figure.savefig(figure_data, dpi=120)
    figure.clf()
    img = Image.open(figure_data)
    return img


def get_metric_image(metrix):
    groups = np.array([g for g, m in metrix])
    ms = np.array([m for g, m in metrix])
    df = pd.DataFrame(ms, columns=['metrics'])
    df.index = groups
    # print(df.head(10))
    svm = sn.barplot(x=groups, y=ms)
    svm.set_xlabel('N groups')
    svm.set_ylabel('metric value')
    svm.set_title('Silhouette metric')
    svm.grid(True)
    figure = svm.get_figure()
    figure_data = BytesIO()
    figure.savefig(figure_data, dpi=120)
    figure.clf()
    img = Image.open(figure_data)
    return img


def get_clustermap_image(path, norm_matrix_rr):
    pc_x = make_group_map((-30, 130), (-40, 130))
    cluster = load_model(path, CLUSTER_FN)
    pca = load_model(path, PCA_FN)
    scaler = load_model(path, SCALER_FN)
    groups = predict_cluster(cluster, pc_x)
    x_point = get_x_for_pca([norm_matrix_rr])
    pc_point = pca_transform(scaler, pca, x_point)
    colours = [CLUSTER_COLOURS[g] for g in groups]
    x,y = pc_x[:, 0], pc_x[:, 1]
    # df = pd.DataFrame(np.array((x, y, colours)).transpose(), columns=['pc1', 'pc2', 'colours'])
    # print(df.head())
    ax = plt.gca()
    ax.scatter(x=x, y=y, s=60, alpha=0.5, c=colours)
    ax.scatter(x=pc_point[:,0], y=pc_point[:,1], s=80, alpha=1, c=['black'])
    print(pc_point)
    ax.grid()
    plt.xlabel('principal component 1')
    plt.ylabel('principal component 2')
    plt.title('Cluster map')
    figure = ax.get_figure()
    figure_data = BytesIO()
    figure.savefig(figure_data, dpi=120)
    figure.clf()
    img = Image.open(figure_data)
    return img
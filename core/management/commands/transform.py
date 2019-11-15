import os
from django.core.management.base import BaseCommand
from core.tools import load_matrix_list, pca_transform, get_x_for_pca, \
    predict_cluster, CLUSTER_COLOURS, load_model, MODEL_FN, make_group_map, N_CLUSTERS, \
    get_heatmap_image, clustering, group_metric, get_metric_image, METRIC_GROUP_MAX
from django.conf import settings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_di(row, pc1, pc2):
    pc1i, pc2i = row.PC1, row.PC2
    return np.sqrt((pc1 - pc1i)**2 + (pc2 - pc2i)**2)


class Command(BaseCommand):
    help = 'transform scatter'

    def add_arguments(self, parser):
        parser.add_argument('-type', type=str, help='Transform types: map | groups | opt', default = 'map')

    def transform_groups(self, path):
        matrix_list = load_matrix_list(path)
        x_for_pca = get_x_for_pca(matrix_list)
        pc_x = pca_transform(x_for_pca)
        model = load_model(path, MODEL_FN)
        groups = predict_cluster(model, pc_x)
        pc1 = [p[0] for p in pc_x]
        pc2 = [p[1] for p in pc_x]
        d = np.array([pc1, pc2, groups]).transpose()
        # print(d)
        df = pd.DataFrame(d, columns=['PC1','PC2', 'group'])
        # print(df.head())
        for i in range(0, N_CLUSTERS):
            dfi = df[df['group'] == i]
            mi = dfi.mean()
            # print(mi)
            pc1i, pc2i = mi.loc['PC1'], mi.loc['PC2']
            dfi['DI'] = np.array([get_di(row, pc1i, pc2i) for i, row in dfi.iterrows()])
            # print(dfi.head())
            dfm = dfi[dfi['DI'] == dfi['DI'].min()]
            n_best = dfm.index[0]
            # print(n_best)
            norm_matrix_rr = matrix_list[n_best]
            img = get_heatmap_image(norm_matrix_rr)
            img.save('static/images/KMeans_group{}.png'.format(i))
        # grouped_data = [pc_x[groups == g] for g in range(0, N_CLUSTERS)]
        # print(grouped_data)
        # means = [grouped_data[i].mean() for i in range(0, N_CLUSTERS)]
        # print(means)

    def transform_map(self, path):
        # matrix_list = load_matrix_list(path)
        # x_for_pca = get_x_for_pca(matrix_list)
        # pc_x = pca_transform(x_for_pca)
        # print(pc_x)
        pc_x = make_group_map((-30, 130), (-40, 130))
        model = load_model(path, MODEL_FN)
        groups = predict_cluster(model, pc_x)
        # print(groups)
        colours = [CLUSTER_COLOURS[g] for g in groups]
        ax = plt.gca()
        ax.scatter(x=pc_x[:, 0], y=pc_x[:, 1], s=60, alpha=0.5, c=colours)
        ax.grid()
        plt.xlabel('principal component 1')
        plt.ylabel('principal component 2')
        plt.title('KMean Clustering')
        figure = ax.get_figure()
        figure.savefig('static/images/KMeans_Cluster.png', dpi=200)
        # plt.show()

    def transform_opt(self, path):
        matrix_list = load_matrix_list(path)
        x_for_pca = get_x_for_pca(matrix_list)
        pc_x = pca_transform(x_for_pca)
        num_clusters = np.arange(2, METRIC_GROUP_MAX+1)
        results = []
        for n_groups in num_clusters:
            model = clustering(pc_x, n_groups)
            groups = predict_cluster(model, pc_x)
            results.append([n_groups, group_metric(pc_x, groups)])
        # print(results)
        img = get_metric_image(results)
        img.save('static/images/KMeans_metrics.png')

    def handle(self, *args, **options):
        path = os.path.join(settings.BASE_DIR, 'samples')
        transform_type = options.get('type')
        if transform_type == 'map':
            self.transform_map(path)
        if transform_type == 'groups':
            self.transform_groups(path)
        if transform_type == 'opt':
            self.transform_opt(path)

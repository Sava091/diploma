import os
from django.core.management.base import BaseCommand
from core.tools import load_matrix_list, pca_transform, get_x_for_pca, clustering, \
    predict_cluster, CLUSTER_COLOURS, load_model, MODEL_FN, make_group_map
from django.conf import settings
import matplotlib.pyplot as plt



class Command(BaseCommand):
    help = 'transform scatter'

    def add_arguments(self, parser):
        # parser.add_argument('path', type=str, help='Name of the directory that will be scanned to import')
        pass

    def handle(self, *args, **options):
        path = os.path.join(settings.BASE_DIR, 'samples')
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
        ax.scatter(x=pc_x[:,0], y=pc_x[:,1], s=60, alpha=0.5, c=colours)
        ax.grid()
        plt.xlabel('principal component 1')
        plt.ylabel('principal component 2')
        plt.title('KMean Clustering')
        figure = ax.get_figure()
        figure.savefig('static/images/KMeans_Cluster.png', dpi=200)
        # plt.show()




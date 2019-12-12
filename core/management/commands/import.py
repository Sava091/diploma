import os
from django.core.management.base import BaseCommand
from core.tools import load_rr_matrix, store_matrix_list, load_beat_matrix, get_x_for_pca, \
    pca_fit, pca_transform, clustering, store_model, CLUSTER_FN, PCA_FN, SCALER_FN, load_matrix_list


class Command(BaseCommand):
    help = 'Import rr record to scatter'

    def add_arguments(self, parser):
        parser.add_argument('path', type=str, help='Name of the directory that will be scanned to import')
        parser.add_argument('-type', type=str, help='Beat format (.rr | .beat)', default='.beat')
        parser.add_argument('-force', type=str, help='=1 skip import', default='0')
        parser.add_argument('-vb', type=int, help='Report level', default=0)

    def full_import(self, path, ext):
        self.stdout.write("import was started at path {}".format(path))
        #заполняем matrix_list считанными скатерограммами образцов ЕКГ
        matrix_list = list()
        for fn in os.listdir(path):
            if fn.endswith(ext):
                self.stdout.write("{}".format(os.path.join(path, fn)))
                # вычисляем скатерограмму для двух форматов эекспорта ЕКГ системой ДиаКард (.rr, .beat)
                if ext == ".rr":
                    matrix_rr = load_rr_matrix(os.path.join(path, fn))
                elif ext == ".beat":
                    matrix_rr = load_beat_matrix(os.path.join(path, fn))
                else:
                    raise Exception("Unknown type")
                matrix_list.append(matrix_rr)
        # записываем matrix_list в файл
        store_matrix_list(matrix_list, path)

        # вычисялем пространство файторов методикой главных компонент
        # подготавливаем данные
        x_for_pca = get_x_for_pca(matrix_list)
        # расчитываем модель методики главных компонент
        pca, scaler = pca_fit(x_for_pca)
        # делаем преобразование скатерограмм в пространство факторов
        pc_x = pca_transform(scaler, pca, x_for_pca)

        # разбиваем на группы скатерограммы в пространстве факторов
        cluster = clustering(pc_x)

        # сохраняем расчитанную модель методики гланвых компонент
        store_model(pca, path, PCA_FN)
        # сохраняем масштабируемые преобразования для подготовки данных
        store_model(scaler, path, SCALER_FN)
        # сохраняем модель разбиения на группы
        store_model(cluster, path, CLUSTER_FN)
        self.stdout.write("import finished successfully")

    def force_import(self, path):
        self.stdout.write("force import started")
        matrix_list = load_matrix_list(path)
        x_for_pca = get_x_for_pca(matrix_list)
        pca, scaler = pca_fit(x_for_pca)
        pc_x = pca_transform(scaler, pca, x_for_pca)
        cluster = clustering(pc_x)
        store_model(pca, path, PCA_FN)
        store_model(scaler, path, SCALER_FN)
        store_model(cluster, path, CLUSTER_FN)
        self.stdout.write("force import finished successfully")

    def handle(self, *args, **options):
        path = os.path.abspath(options.get('path'))
        ext = options.get('type')
        force = options.get('force')
        if force == '1':
            self.force_import(path)
        else:
            self.full_import(path, ext)
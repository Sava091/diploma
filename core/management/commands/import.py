import os
from django.core.management.base import BaseCommand
from core.tools import load_rr_matrix, store_matrix_list, load_beat_matrix


class Command(BaseCommand):
    help = 'Import rr record to scatter'

    def add_arguments(self, parser):
        parser.add_argument('path', type=str, help='Name of the directory that will be scanned to import')
        parser.add_argument('-type', type=str, help='Beat format (.rr | .beat)', default='.beat')
        parser.add_argument('-vb', type=int, help='Report level', default=0)

    def handle(self, *args, **options):
        path = os.path.abspath(options.get('path'))
        ext = options.get('type')
        self.stdout.write("import was started at path {}".format(path))
        matrix_list = list()
        for fn in os.listdir(path):
            if fn.endswith(ext):
                # print(os.path.join(path, fn))
                if ext == ".rr" :
                    matrix_rr = load_rr_matrix(os.path.join(path,fn))
                elif ext == ".beat" :
                    matrix_rr = load_beat_matrix(os.path.join(path,fn))
                else :
                    raise Exception("Unknown type")
                matrix_list.append(matrix_rr)
        store_matrix_list(matrix_list, path)
        self.stdout.write("import finished successfully")
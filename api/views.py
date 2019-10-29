from django.http import HttpResponse, HttpResponseNotFound
from PIL import Image
from io import BytesIO
import seaborn as sn
import os
from core.tools import load_matrix_list, load_heatmap


def get_image_response(img, format="png"):
    response = HttpResponse(content_type="image/{}".format(format))
    if img is not None:
        img.save(response, 'png')
    return response


def view_heatmap_image(request, n):
    path = os.path.abspath('samples')
    matrix_list = load_matrix_list(path)
    if n < 1 or n > matrix_list.shape[0]:
        return HttpResponseNotFound()
    norm_matrix_rr = matrix_list[n-1]
    svm = sn.heatmap(norm_matrix_rr, cmap='coolwarm', center=0)  # linecolor='white', linewidths=1,
    svm.invert_yaxis()
    figure = svm.get_figure()
    figure_data = BytesIO()
    figure.savefig(figure_data, dpi=120)
    figure.clf()
    img = Image.open(figure_data)
    return get_image_response(img)


def view_patient_heatmap_image(request):
    path = os.path.abspath('upload')
    fn = request.GET.get('heatmap_fn') + '.npy'
    ext_fn = os.path.join(path, fn)
    if not os.path.isfile(ext_fn):
        return HttpResponseNotFound()
    norm_matrix_rr = load_heatmap(path, fn)
    svm = sn.heatmap(norm_matrix_rr, cmap='coolwarm', center=0)  # linecolor='white', linewidths=1,
    svm.invert_yaxis()
    figure = svm.get_figure()
    figure_data = BytesIO()
    figure.savefig(figure_data, dpi=120)
    figure.clf()
    img = Image.open(figure_data)
    return get_image_response(img)
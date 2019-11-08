from django.http import HttpResponse, HttpResponseNotFound
import os
from core.tools import load_matrix_list, load_heatmap, get_heatmap_image, get_clustermap_image


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
    img = get_heatmap_image(norm_matrix_rr)
    return get_image_response(img)


def view_patient_heatmap_image(request):
    path = os.path.abspath('upload')
    fn = request.GET.get('rr_fn') + '.npy'
    ext_fn = os.path.join(path, fn)
    if not os.path.isfile(ext_fn):
        return HttpResponseNotFound()
    norm_matrix_rr = load_heatmap(path, fn)
    img = get_heatmap_image(norm_matrix_rr)
    return get_image_response(img)


def view_patient_clustermap_image(request):
    path = os.path.abspath('upload')
    model_path = os.path.abspath('samples')
    fn = request.GET.get('rr_fn') + '.npy'
    ext_fn = os.path.join(path, fn)
    if not os.path.isfile(ext_fn):
        return HttpResponseNotFound()
    norm_matrix_rr = load_heatmap(path, fn)
    img = get_clustermap_image(model_path, norm_matrix_rr)
    return get_image_response(img)
import os
import json
from django.shortcuts import render
from django.http import HttpResponseRedirect, HttpResponse, FileResponse
from django.views.generic.base import TemplateView
from django.conf import settings
from xhtml2pdf import pisa

from core.tools import get_matrix_list_count, save_temp, load_beat_matrix, store_heatmap, get_heatmap_image,\
    get_clustermap_image


# Create your views here.
class HomeView(TemplateView):
    template_name = 'index.html'

    def get_context_data(self, **kwargs):
        context = super(HomeView, self).get_context_data(**kwargs)
        path = os.path.abspath('samples')
        n = get_matrix_list_count(path)
        context['samples'] = range(1, n+1)
        return context


def view_upload_start(request):
    info = request.GET.get('info')
    if info is None:
        info = ''
    data = request.FILES.get('inputBeats')
    if data is None:
        return render(request, 'index.html', {'info' : info})
    return upload_transfer(request)


def upload_transfer(request):
    data = request.FILES.get('inputBeats')
    if data is None:
        return HttpResponseRedirect('/?info = Beat File was not assigned')
    if data.size == 0:
        return HttpResponseRedirect('/?info = Beat File damaged')

    dir = os.path.join(settings.BASE_DIR, 'upload')
    temp_fn = save_temp(data.read(), dir, 'ecg_','.beat')
    b,e = os.path.splitext(temp_fn)
    fn_heat = b + '.heatmap'
    ext_fn = os.path.join(dir, temp_fn)
    norm_matrix_rr = load_beat_matrix(ext_fn)
    store_heatmap(norm_matrix_rr, dir, fn_heat)

    img = get_heatmap_image(norm_matrix_rr)
    img.save('static/images/{}'.format(fn_heat + '.png'))

    model_path = os.path.abspath('samples')
    img = get_clustermap_image(model_path, norm_matrix_rr)
    img.save('static/images/{}'.format(fn_heat + '.clustermap.png'))
    resp = HttpResponseRedirect('/heatmap/')
    resp.set_cookie('heatmap_file', value=fn_heat, max_age=7*24*60*60)
    return resp


def view_heatmap(request):
    beat_fn = request.COOKIES.get('heatmap_file')
    if beat_fn is None:
        return HttpResponseRedirect('/')
    annotate_level = request.POST.get('annotateLevel')
    annotate_text = request.POST.get('annotateText')
    path = os.path.abspath('upload')
    annotate_fn = os.path.join(path, beat_fn + '.annotation.json')
    if (annotate_text is not None) and (annotate_level is not None):
        annotate_data = dict(level=annotate_level, text=annotate_text)
        with open(annotate_fn, 'w') as fp:
            json.dump(annotate_data, fp, indent=4)
    has_annotation = os.path.isfile(annotate_fn)
    if has_annotation:
        with open(annotate_fn, 'r') as fp:
            annotate_data = json.load(fp)
            annotate_level = annotate_data.get('level')
            annotate_text = annotate_data.get('text')
    return render(request, 'heatmap.html', {'rr_fn': beat_fn, 'hasAnnotation': has_annotation,
                                            'annotationLevel': annotate_level, 'annotationText': annotate_text})


def view_patient_report(request):
    beat_fn = request.COOKIES.get('heatmap_file')
    if beat_fn is None:
        return HttpResponseRedirect('/')
    path = os.path.abspath('upload')
    pdf_fn = os.path.join(path, beat_fn + '.report.pdf')
    with open(pdf_fn, 'w+b') as pdf:
        r = render('report.html', {})
        pisa_status = pisa.CreatePDF(r.content, dest=pdf)
    if pisa_status.err:
        return HttpResponse(status=500)
    s = open(pdf_fn, 'rb')
    return FileResponse(s, content_type='application/pdf')


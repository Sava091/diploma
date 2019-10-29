import os
import datetime
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.views.generic.base import TemplateView
from django.conf import settings

from core.tools import get_matrix_list_count, save_temp, load_beat_matrix, store_heatmap


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
    resp = HttpResponseRedirect('/heatmap/')
    resp.set_cookie('heatmap_file', value=fn_heat, max_age=7*24*60*60)
    return resp


def view_heatmap(request):
    beat_fn = request.COOKIES.get('heatmap_file')
    if beat_fn is None:
        return HttpResponseRedirect('/')
    return render(request, 'heatmap.html', {'heatmap_fn': beat_fn})

from django.shortcuts import render
import os
from django.views.generic.base import TemplateView
from core.tools import get_matrix_list_count


# Create your views here.
class HomeView(TemplateView):
    template_name = 'index.html'

    def get_context_data(self, **kwargs):
        context = super(HomeView, self).get_context_data(**kwargs)
        path = os.path.abspath('samples')
        n = get_matrix_list_count(path)
        context['samples'] = range(1, n+1)
        return context
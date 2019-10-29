from django.urls import path
from . import views


urlpatterns = [
        path('heatmap/', views.view_heatmap),
        path('', views.view_upload_start),
    ]
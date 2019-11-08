from django.urls import path
from . import views


urlpatterns = [
        path('heatmap/', views.view_heatmap),
        path('report/', views.view_patient_report),
        path('', views.view_upload_start),
    ]
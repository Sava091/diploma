from django.urls import path
from . import views


urlpatterns = [
        path('heatmap/patient/', views.view_patient_heatmap_image),
        path('clustermap/patient/', views.view_patient_clustermap_image),
        path('heatmap/<int:n>/', views.view_heatmap_image),
]

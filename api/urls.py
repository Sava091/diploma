from django.urls import path
from . import views


urlpatterns = [
        path('heatmap/<int:n>/', views.view_heatmap_image),
    ]
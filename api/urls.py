# api/urls.py
from django.urls import path
from .views import healthz, predict, reload_labels

urlpatterns = [
    path("healthz", healthz, name="healthz"),
    path("predict", predict, name="predict"),
    path("reload", reload_labels, name="reload_labels"),
]

from django.urls import path

from . import views

app_name = "tryon"

urlpatterns = [
    path("", views.measure_view, name="measure"),
    path("verify/", views.verify_view, name="verify"),
    path("wardrobe/", views.wardrobe_view, name="wardrobe"),
    path("tryon/start/<str:garment_id>/", views.tryon_start_view, name="tryon_start"),
    path("tryon/result/<str:job_id>/", views.tryon_result_view, name="result"),
    path("tryon/status/<str:job_id>.json", views.tryon_status_json_view, name="status_json"),
]

from django.urls import path
from . import views


urlpatterns = [
    path('', views.mainView, name = 'mainView'),
    path('cluster/', views.cluster, name = 'cluster'),
    path('ingredient/', views.ingredient, name = 'ingredient'),
    path('image/', views.image, name = 'image'),
    path('changeImage/', views.changeImage, name = 'changeImage'),
    path('tech/', views.tech, name = 'tech'),
    path('barLocation/', views.barLocation, name = 'barLocation'),

]
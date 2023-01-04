from django.urls import path
from . import views


urlpatterns = [
    path('', views.mainView, name = 'mainView'),

    # clustering url, ajax
    path('cluster/', views.cluster, name = 'cluster'),
    path('cluster/ajax', views.clusterAjax, name = 'clusterajax'),

    path('ingredient/', views.ingredient, name = 'ingredient'),


    # image url, ajax
    path('image/', views.image, name = 'image'),
    path('image/ajax', views.imageAjax, name = 'imageAjax'),

    #file Download
    path('downloadFile/', views.downloadFile, name = 'downloadFile'),




    path('changeImage/', views.changeImage, name = 'changeImage'),
    path('tech/', views.tech, name = 'tech'),
    path('barLocation/', views.barLocation, name = 'barLocation'),

]
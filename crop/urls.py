
from numpy import nanmax
from crop.stats import season_crop_df
from os import name
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.home,name="home"),
    path('result',views.result,name="result"),
    path('Crop_stats',views.dist_crop_stats,name="dist_crop_stats"),
    path('Season_stats',views.dist_season_state,name="dist_season_stats"),
    path('Season_Crop',views.season_crop_stats,name='season_crop_stats'),
    path('General_Crop_stats',views.general_crop_stats,name='general_crop_stats')
]

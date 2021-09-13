from crop import Editor
from crop import stats
from django.shortcuts import render
import joblib




def home(request):
    lis=[]
    return render(request,'home.html')


def result(request):

    lm=joblib.load('.\Files\predictor.sav')

    global District,Year,Season,Crop,Area

    District=request.POST['district']
    Year=request.POST['year']
    Season=request.POST['season']
    Crop=request.POST['crop_name']
    Area=request.POST['area']

    lis=[District,Year,Season,Crop,Area]

    inp=Editor.input_type(lis)


    production=lm.predict([inp])[0]

    return render(request,'result.html',locals())


def dist_crop_stats(request):
    dist=District
    crp=Crop
    dist_detail=stats.dt_df(District,Crop)
    return render(request,'dist_crop_stats.html',locals())

def dist_season_state(request):
    dist=District
    ssn=Season
    dist_detail=stats.season_dt(dist,ssn)
    return render(request,'dist_season_stats.html',locals())

def season_crop_stats(request):
    ssn=Season
    crp=Crop
    dist_detail=stats.season_crop_df(ssn,crp)
    return render(request,'season_crop_stats.html',locals())

def general_crop_stats(request):
    crp=Crop
    dist_detail=stats.general_crop_df(crp)
    return render(request,'general_crop_stats.html',locals())
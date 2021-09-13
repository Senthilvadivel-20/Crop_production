import pandas as pd

df=pd.read_csv('.\Files\kaggle agri.csv')
df.dropna(inplace=True)

def dt_df(d,c):
    dt=df[(df['District_Name']==d)&(df['Crop']==c)]
    dt=dt.sort_values(by=['Production'],ascending=False)
    if dt.empty:
        crop_list=False
        return crop_list
    else:
        if len(dt)>=10:
            size=10
        else:
            size=len(dt)

        crop_list=[]
        for i in range(size):
            temp={
            "dist":dt['District_Name'].to_list()[i],
            'year':dt['Crop_Year'].to_list()[i],
            'season':dt['Season'].to_list()[i],
            "crop":dt['Crop'].to_list()[i],
            'area':dt['Area'].to_list()[i],
            'production':dt['Production'].to_list()[i]
            }
            crop_list.append(temp)

        return crop_list

def season_dt(d,s):
    dt=df[(df['District_Name']==d)&(df['Season']==s)]
    dt=dt.sort_values(by=['Production'],ascending=False)
    if dt.empty:
        crop_list=False
        return crop_list
    else:
        if len(dt)>=10:
            size=10
        else:
            size=len(dt)

        crop_list=[]
        for i in range(size):
            temp={
            "dist":dt['District_Name'].to_list()[i],
            'year':dt['Crop_Year'].to_list()[i],
            'season':dt['Season'].to_list()[i],
            "crop":dt['Crop'].to_list()[i],
            'area':dt['Area'].to_list()[i],
            'production':dt['Production'].to_list()[i]
            }
            crop_list.append(temp)
        return crop_list


def season_crop_df(s,c):
    dt=df[(df['Crop']==c)&(df['Season']==s)]
    dt=dt.sort_values(by=['Production'],ascending=False)
    if dt.empty:
        crop_list=False
        return crop_list
    else:
        if len(dt)>=10:
            size=10
        else:
            size=len(dt)

        crop_list=[]
        for i in range(size):
            temp={
            "dist":dt['District_Name'].to_list()[i],
            'year':dt['Crop_Year'].to_list()[i],
            'season':dt['Season'].to_list()[i],
            "crop":dt['Crop'].to_list()[i],
            'area':dt['Area'].to_list()[i],
            'production':dt['Production'].to_list()[i]
            }
            crop_list.append(temp)
        return crop_list


def general_crop_df(c):
    dt=df[(df['Crop']==c)]
    dt=dt.sort_values(by=['Production'],ascending=False)
    if dt.empty:
        crop_list=False
        return crop_list
    else:
        if len(dt)>=10:
            size=10
        else:
            size=len(dt)

        crop_list=[]
        for i in range(size):
            temp={
            "dist":dt['District_Name'].to_list()[i],
            'year':dt['Crop_Year'].to_list()[i],
            'season':dt['Season'].to_list()[i],
            "crop":dt['Crop'].to_list()[i],
            'area':dt['Area'].to_list()[i],
            'production':dt['Production'].to_list()[i]
            }
            crop_list.append(temp)
        return crop_list


# print(general_crop_df('Apple'))




import pandas as pd
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('.\Files\kaggle agri.csv')

def input_type(lis):
    df=pd.read_csv('.\Files\kaggle agri.csv')
    district_encoder=LabelEncoder()
    season_encoder=LabelEncoder()
    crop_encoder=LabelEncoder()

    df['District_Name']=district_encoder.fit_transform(df['District_Name'])
    df['Season']=season_encoder.fit_transform(df['Season'])
    df['Crop']=crop_encoder.fit_transform(df['Crop'])


    inp1=district_encoder.transform([lis[0]])
    inp2=int(lis[1])
    inp3=season_encoder.transform([lis[2]])
    inp4=crop_encoder.transform([lis[3]])
    inp5=int(lis[4])

    inputs=[inp1,inp2,inp3,inp4,inp5]

    return inputs


def get_df():
    df=pd.read_csv('.\Files\kaggle agri.csv')
    return df



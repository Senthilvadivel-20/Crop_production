import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib


file='kaggle agri.csv'

df=pd.read_csv(file)

df.dropna(inplace=True)

from sklearn.preprocessing import LabelEncoder

district_encoder=LabelEncoder()
season_encoder=LabelEncoder()
crop_encoder=LabelEncoder()

df['District_Name']=district_encoder.fit_transform(df['District_Name'])
df['Season']=season_encoder.fit_transform(df['Season'])
df['Crop']=crop_encoder.fit_transform(df['Crop'])

x=df[['District_Name','Crop_Year','Season','Crop','Area']]
y=df['Production']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state = 42)

lm=RandomForestRegressor(random_state = 42)

lm.fit(x_train,y_train)

file='predictor.sav'
joblib.dump(lm,file)
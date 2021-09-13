import pandas as pd

df=pd.read_csv('.\Files\kaggle agri.csv')
df.dropna(inplace=True)

state=False


if state:
    print('###########################################################s')
    statement=False
else:
    print("YESSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS")
    statement=True
from operator import le
import pandas as pd
info={'Gender':['Male','Female','Male','Female','Female'],
'Position':['Head','Asst.Prof','Assit.Prof','Head','Assit.Prof']}
df=pd.DataFrame(info)
print(df)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
gender_encoded=le.fit.transform(dt['Gender'])
gender_encoded=le.fit.transform(dt['Position'])
df['Encoded_Gender']=gender_encoded
df['Encoded_Postion']=encoded_position
print(df)

from urllib.request import urlretrieve
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib


url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

urlretrieve(url, "titanic.csv")

print("Downloaded successfully!")

df = pd.read_csv("titanic.csv")
df['Age']=df['Age'].fillna(df['Age'].mean())
#df=df.dropna(subset=['Cabin'])
print(df.shape)

df = df.drop(columns=['Cabin'])

df = df.drop(columns=['Ticket', 'Name', 'PassengerId'])
sns.set_style(style="darkgrid")
matplotlib.rcParams['figure.figsize'] = (12, 6)
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.facecolor'] = "#4F90E500"


from sklearn.model_selection import train_test_split
train,test=train_test_split(df,test_size=0.2,random_state=42)
train,validation=train_test_split(train,test_size=0.25,random_state=42)

targets='Survived'
'''train_split'''
x_train=train.drop(columns=[targets])

y_train=train[targets]
x_validation=validation.drop(columns=[targets])

y_validation=validation[targets]
x_test=test.drop(columns=[targets])

y_test=test[targets]



x_validation= pd.get_dummies(x_validation,columns=['Sex'],drop_first=True)
x_test= pd.get_dummies(x_test,columns=['Sex'],drop_first=True)
x_train= pd.get_dummies(x_train,columns=['Sex'],drop_first=True)
x_train=pd.get_dummies(x_train,columns=['Embarked'],drop_first=True)
x_validation=pd.get_dummies(x_validation,columns=['Embarked'],drop_first=True)
x_test=pd.get_dummies(x_test,columns=['Embarked'],drop_first=True)

print(x_train.dtypes)


print(x_train.isnull().sum())
from sklearn.linear_model import LogisticRegression

model=LogisticRegression(max_iter=1000) 
model.fit(x_train,y_train)
predictions=model.predict(x_validation)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(accuracy_score(y_validation,predictions))
print(confusion_matrix(y_validation,predictions))
print(classification_report(y_validation,predictions))

predictions=model.predict(x_test)
print(accuracy_score(y_test,predictions))   
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


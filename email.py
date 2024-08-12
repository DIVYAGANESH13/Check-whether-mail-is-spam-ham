# EMAIL SPAM DETECTION USING DATA SCIENCE
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
filepath=r"C:\Users\Jayap\OneDrive\Documents\Desktop\email real\spam.csv"
df=pd.read_csv(filepath,encoding='latin-1')
data =df.where((pd.notnull(df)),'')
data.loc[data['v1']=='spam','v1']=0
data.loc[data['v1']=='ham','v1']=1
X=data['v2']
Y=data['v1']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
feature_extraction=TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
X_train_features=feature_extraction.fit_transform(X_train)
X_test_features=feature_extraction.transform(X_test)
Y_train=Y_train.astype('int')
Y_test=Y_test.astype('int')
print(X_train_features)
model=LogisticRegression()
model.fit(X_train_features,Y_train)
prediction_on_training_data=model.predict(X_train_features)
accuracy_on_training_data=accuracy_score(Y_train,prediction_on_training_data)
print("Accuracy on training data",accuracy_on_training_data)
prediction_on_test_data=model.predict(X_test_features)
accuracy_on_test_data=accuracy_score(Y_test,prediction_on_test_data)
print("Accuracy of test data",accuracy_on_test_data)
input=[" Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's" ]
input_data_feature=feature_extraction.transform(input)
prediction=model.predict(input_data_feature)
print(prediction)
if(prediction[0]==1):
    print("ham mail")
else:
    print("spam mail")
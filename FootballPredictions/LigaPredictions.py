#Football match Predictions made based on La-Liga season 2017/18 matches
#The results can be found in the results.csv file and the data used in the SP1.csv file.
# '0' mean Home win , '1'- Away win, '2' - Draw
import pandas as pd 
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import numpy
from sklearn.metrics import accuracy_score

data = pd.read_csv("SP1.csv")#deleted unwanted columns (related to betting) initially

#converting 'D','H','A' to 2,0,1 respectively 
data['FTR'].loc[data['FTR']=='D']=2
data['FTR'].loc[data['FTR']=='H']=0
data['FTR'].loc[data['FTR']=='A']=1
data['HTR'].loc[data['HTR']=='D']=2
data['HTR'].loc[data['HTR']=='H']=0
data['HTR'].loc[data['HTR']=='A']=1

print(data.head(10))

num_matches = data.shape[0]

print(num_matches)

"""sns.countplot(x='FTR',data=data)
plt.show()"""
n_homewins = len(data[data.FTR == 0])
win_rate = (float(n_homewins) / (num_matches)) * 100
print("Win rate for home team %f /%",win_rate)

data=data.drop(['HomeTeam','AwayTeam','Date','Div'],axis=1)

X_all = data.drop(['FTR'],1)
y_all = data['FTR']

from sklearn.cross_validation import train_test_split

# Shuffle and split the dataset into training and testing set.
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 90,random_state = 2,stratify = y_all)

"""clf_A = LogisticRegression(random_state = 42)
clf_A.fit(X_train,y_train)"""#92 percent accuracy 
"""clf_B = RandomForestClassifier(n_estimators=100)
clf_B.fit(X_train,y_train)"""#94 percent accuracy
clf_C = xgb.XGBClassifier(seed = 82)#97 percent accuracy with test_size = 90
clf_C.fit(X_train,y_train)
y_pred=clf_C.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
print(clf_C.score(X_test,y_test))
df=pd.DataFrame({'FTR':y_test,'Predicted_FTR':y_pred})
df.to_csv("resultss.csv")

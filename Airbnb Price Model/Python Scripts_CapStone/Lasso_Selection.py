##universal setting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import shapely
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import seaborn as sns
from sklearn import linear_model
from sklearn.svm import SVR
# ---------data location------------------------------------------------------------------------------------------------
read_location="C:/Selina/Class/(DS_MS)/Capstone/DataSet/"
save_location="C:/Selina/Class/(DS_MS)/Capstone/Result/"
my_cmap=plt.cm.get_cmap("Pastel2")
y="price"
testing="price_less_350_more_LassoFeatureSelection"
#import data

X_train=pd.read_csv(read_location+"X_train.csv",index_col=False)
X_test=pd.read_csv(read_location+"X_test.csv",index_col=False)
y_train=pd.read_csv(read_location+"y_train.csv",index_col=False)
y_test=pd.read_csv(read_location+"y_test.csv",index_col=False)

print(X_train.head(4))
print(y_train.head(4))
print("Selina")

clf = linear_model.Lasso(alpha=1)
clf.fit(X_train,y_train)

y_train_predict = clf.predict(X_train)
mse = mean_squared_error(y_train, y_train_predict)
r2 = r2_score(y_train, y_train_predict)
ad_r2= 1 - (1 - r2 ) * ((X_train.shape[0] - 1) / (X_train.shape[0] - X_train.shape[1] - 1))
print("The model performance for training set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('R2 score is {}'.format(r2*100))
print('adjust_R2 score is {}'.format(ad_r2*100))
print("\n")

y_test_predict = clf.predict(X_test)
mse = mean_squared_error(y_test, y_test_predict)
r2 = r2_score(y_test, y_test_predict)
ad_r2= 1 - (1 - r2 ) * ((X_test.shape[0] - 1) / (X_test.shape[0] - X_test.shape[1] - 1))
print("The model performance for testing set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('R2 score is {}'.format(r2*100))
print('adjust_R2 score is {}'.format(ad_r2*100))
print("\n")

plt.figure(figsize=(16,8))
sns.regplot(y_test_predict,y_test.loc[:,y])
plt.xlabel('Predictions')
plt.ylabel('Actual')
plt.title("linear Regressor Predictions")
plt.show()


print(clf.coef_)
print(X_train.columns[clf.coef_!=0])
print(X_train.columns)
print("Selina")

#compare different models
model_list={"linear regression":LinearRegression(), "Lasso Regression":linear_model.Lasso(alpha=1), "SVM rbf": SVR(C=0.3)}
AdR=pd.DataFrame(columns=["AdjustR2_training","AdjustR2_testing","MSE_testing","coef","x_col"],index=list(model_list.keys()))
for i in model_list.keys():
    print(i)
    clf = model_list[i]
    clf.fit(X_train, y_train)
    if i!="SVM rbf":
        AdR.loc[i,"coef"]=list(clf.coef_)
        print(clf.coef_)
    AdR.loc[i,"x_col"]=str(X_train.columns)
    if i=="Lasso Regression":
        AdR.loc[i,"Lasso Coe!=0"]=str(X_train.columns[clf.coef_!=0])

    y_train_predict = clf.predict(X_train)
    mse = mean_squared_error(y_train, y_train_predict)
    r2 = r2_score(y_train, y_train_predict)
    ad_r2 = 1 - (1 - r2) * ((X_train.shape[0] - 1) / (X_train.shape[0] - X_train.shape[1] - 1))
    print("The model performance for training set")
    print("--------------------------------------")
    print('MSE is {}'.format(mse))
    print('R2 score is {}'.format(r2 * 100))
    print('adjust_R2 score is {}'.format(ad_r2 * 100))
    print("\n")
    AdR.loc[i,"AdjustR2_training"]=ad_r2*100

    y_test_predict = clf.predict(X_test)
    mse = mean_squared_error(y_test, y_test_predict)
    r2 = r2_score(y_test, y_test_predict)
    ad_r2 = 1 - (1 - r2) * ((X_test.shape[0] - 1) / (X_test.shape[0] - X_test.shape[1] - 1))
    print("The model performance for testing set")
    print("--------------------------------------")
    print('MSE is {}'.format(mse))
    print('R2 score is {}'.format(r2 * 100))
    print('adjust_R2 score is {}'.format(ad_r2 * 100))
    print("\n")
    AdR.loc[i, "AdjustR2_testing"] = ad_r2 * 100
    AdR.loc[i, "MSE_testing"] = mse

    plt.figure(figsize=(10, 10))
    sns.regplot(y_test_predict, y_test.loc[:, y])
    plt.xlabel('Predictions')
    plt.ylabel('Actual')
    plt.title(i+" Prediction")
    plt.savefig(save_location + i + "_testing_"+testing+".png")
    plt.show()


AdR.to_csv(save_location+"testing_"+testing+".csv")
print(AdR)
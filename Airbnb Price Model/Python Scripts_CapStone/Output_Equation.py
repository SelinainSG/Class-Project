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
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn import linear_model
from sklearn.svm import SVR
#--------OutPut Equation------------------------------------------------------------------------------------------------

#for ploting
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 10),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)


def OutPut(clf,X_train,y_train,X_test,y_test,classfier,save_name):
    AdR = pd.DataFrame(columns=["AdjustR2_training", "AdjustR2_testing", "MSE_testing", "coef", "x_col"],index=[classfier])

    y_train_predict = clf.predict(X_train)
    mse = mean_squared_error(y_train, y_train_predict)
    r2 = r2_score(y_train, y_train_predict)
    ad_r2 = 1 - (1 - r2) * ((X_train.shape[0] - 1) / (X_train.shape[0] - X_train.shape[1] - 1))
    AdR.loc[classfier,"AdjustR2_training"]=ad_r2*100

    y_test_predict = clf.predict(X_test)
    mse = mean_squared_error(y_test, y_test_predict)
    r2 = r2_score(y_test, y_test_predict)
    ad_r2 = 1 - (1 - r2) * ((X_test.shape[0] - 1) / (X_test.shape[0] - X_test.shape[1] - 1))
    AdR.loc[classfier, "AdjustR2_testing"] = ad_r2 * 100
    AdR.loc[classfier, "MSE_testing"] = mse

    AdR.to_csv(save_name+".csv")

    plt.figure(figsize=(10, 10))
    sns.regplot(y_test_predict, y_test.loc[:, "price"])
    plt.xlabel('Predictions')
    plt.ylabel('Actual')
    plt.title(classfier+" Prediction")
    plt.savefig(save_name+".png")
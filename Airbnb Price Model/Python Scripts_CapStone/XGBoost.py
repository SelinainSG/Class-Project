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
from Output_Equation import OutPut
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
# ---------data location------------------------------------------------------------------------------------------------
read_location="C:/Selina/Class/(DS_MS)/Capstone/DataSet/"
save_location="C:/Selina/Class/(DS_MS)/Capstone/Result/"
my_cmap=plt.cm.get_cmap("Pastel2")
y="price"
testing="CrossValidation_"
classifier="XGBT"
#import data

X_train=pd.read_csv(read_location+"X_train.csv",index_col=False)
X_test=pd.read_csv(read_location+"X_test.csv",index_col=False)
y_train=pd.read_csv(read_location+"y_train.csv",index_col=False)
y_test=pd.read_csv(read_location+"y_test.csv",index_col=False)

#--------Grid Search Setting-------------------------------------------------------------------------------------------




clf=XGBRegressor(n_estimators=200, eta=0.5, gamma=0.1, max_depth=7, reg_lambda =10)
clf.fit(X_train,y_train.values.ravel())
save_name=save_location+"XGB"
OutPut(clf,X_train,y_train,X_test,y_test,classifier,save_name)
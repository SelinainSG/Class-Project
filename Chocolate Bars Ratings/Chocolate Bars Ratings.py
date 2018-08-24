import pandas as pd
import os
import numpy as np
import re

data = pd.read_csv("flavors_of_cacao.csv")
column_names=['Brand Name', 'Specific Bar Origin','REF', 'Review Date', 'Cocoa Percent', 'Company Location', 'Rating','Bean Type', 'Bean Origin']
data=data.rename(columns=dict(zip(data.columns,column_names)))

#Data Clean--------------------------------------------------------------------------------------------------------------------------------------------------------------
# Bean Origin-------------
# drop null space
t=data["Bean Origin"].notnull()
data=data[t]
print(data.isnull().sum())
#drop space which showed ? in the csv
filter_0 = data["Bean Origin"]!='\xa0'
data=data[filter_0]
# typo
data["Bean Origin"]=data["Bean Origin"].replace("Domincan Republic","Dominican Republic")
data["Bean Origin"]=data["Bean Origin"].replace("Trinidad-Tobago","Trinidad")
data["Specific Bar Origin"]=data["Specific Bar Origin"].replace("Trinidad-Tobago","Trinidad")
data["Bean Origin"]=data["Bean Origin"].replace("Sao Tome & Principe","Sao Tome")
data["Specific Bar Origin"]=data["Specific Bar Origin"].replace("Sao Tome & Principe","Sao Tome")
#align the expression
data["Bean Origin"]=data["Bean Origin"].replace("Venezuela/ Ghana","Venezuela, Ghana")
#get out of () to simplify the data

for i in data.index:
    sub_str=str(data.loc[i,"Bean Origin"])
    sub_str=sub_str.split(" (")
    data.loc[i,"Bean Origin"]=sub_str[0].strip()

# Brand Name-------------
#get out of () to simplify the data
for i in data.index:
    sub_str = str(data.loc[i,"Brand Name"])
    sub_str = sub_str.split(" (")
    data.loc[i,"Brand Name"] = sub_str[0].strip()

#Specific Bar Origin -----------------------------------------------------------------------------
# re-organize this item , this item should be bar's name usually named after bean's country' and county
#country -> country-> special name
for i in data.index:
    if data.loc[i, "Bean Origin"] != data.loc[i, "Specific Bar Origin"]:
        sub_str=data.loc[i,"Specific Bar Origin"].split(",")
        if sub_str[0]==data.loc[i , "Bean Origin"]:
            data.loc[i,"Specific Bar Origin"]=sub_str[1]
        else:
            data.loc[i,"Specific Bar Origin"]=sub_str[0]

#SVM for missing data-----------------------------------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# decide bean type by bean origin/brand name/ specific bar origin
data_BT=data[["Brand Name","Specific Bar Origin","Bean Type","Bean Origin"]]
data_BT["Bean Type"]=data_BT["Bean Type"].replace(np.NaN,'\xa0')
print(data_BT.isnull().sum())
Seq=pd.DataFrame({"sequence":range(len(data.index)),"Index":data.index})
Seq=Seq.set_index("Index")
#creat filter  for Bean Type
filter_BT=data["Bean Type"]!='\xa0'
filter_BT_application=data["Bean Type"]=='\xa0'
#create dummy X
X_data_dummy=pd.get_dummies(data_BT[["Brand Name","Specific Bar Origin","Bean Origin"]])
#sepertate data into train and application
X_data_dummy_train = X_data_dummy[filter_BT]
X=X_data_dummy_train.values
Seq_app = Seq[filter_BT_application]
Y_data = data_BT["Bean Type"].astype(str)
Y_data_train = Y_data[filter_BT]

#SVM training function
class_le = LabelEncoder()
y = class_le.fit_transform(Y_data_train)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=200)

#RBF SVM
svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

print(classification_report(y_test,y_pred))
print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)

#prediction result by SVM
X_data_dummy_app = X_data_dummy[filter_BT_application]
X_app = X_data_dummy_app.values
y_re=svm.predict(X_app)
y_re=class_le.inverse_transform(y_re)

#insert y_re into those unknown data
ref=pd.DataFrame({"prediction": y_re,"Index": Seq_app.index,"sequence": range(len(y_re))})

for i in ref["sequence"]:
    x_index=ref.loc[i,"Index"]
    xo=ref.loc[i,"prediction"]
    if data.loc[x_index, "Bean Type"]=='\xa0':
        data.loc[x_index, "Bean Type"] = xo

# DT prototype-------------------------------------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import pydotplus
import collections
# Convert cocoa percentage into numerical
data["Cocoa Percent"]=data["Cocoa Percent"].str.strip("%").astype(float)/100
data.info()
# categorical data
X_C_CL_P = LabelEncoder()
X_C_CL=X_C_CL_P.fit_transform(data["Company Location"])
X_C_BO_P=LabelEncoder()
X_C_BO=X_C_BO_P.fit_transform(data["Bean Origin"].astype(str))
X_C_BT_P = LabelEncoder()
X_C_BT=X_C_BT_P.fit_transform(data["Bean Type"].astype(str))

X_N_CP=data["Cocoa Percent"].values
Y_C_R_P= LabelEncoder()
Y_C_R=Y_C_R_P.fit_transform(data["Rating"])
Y_C_R=Y_C_R.reshape(len(Y_C_R),1)
#creat test dataset
x_list=(X_N_CP,X_C_CL,X_C_BO,X_C_BT)
X=np.vstack(x_list).T

X_train, X_test, y_train, y_test = train_test_split(X, Y_C_R, test_size=0.2, random_state=400)

#model set up
clf = tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf=5)
clf = clf.fit(X_train,y_train)

N_list=["Coca Percent","Company Location","Bean Origin","Bean Type"]
class_name=data["Rating"].value_counts()
dot_data = tree.export_graphviz(clf,
                                feature_names=N_list,
                                out_file=None,
                                filled=True,
                                rounded=True,
                                class_names=Y_C_R_P.classes_.astype(str)
                                )
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('tree.png')
graph.write_svg('tree.svg')

# calculate metrics gini model
y_pred= clf.predict(X_test)
print("\n")
print("Results Using Gini Index: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")
print ('-'*80 + '\n')
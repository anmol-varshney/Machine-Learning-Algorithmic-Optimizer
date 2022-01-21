from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
#from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from classification_metrics import metrics
import warnings
warnings.filterwarnings("ignore")

def Logistic_regression(X_train, y_train, X_test, y_test,  data_name, algorithm_selected):
    model = LogisticRegression()
    model.fit(X_train,y_train)
    model_pred = model.predict(X_test)

    return metrics(X_test, y_test, model_pred, model,  data_name, algorithm_selected)


def XGB_Classifier(X_train, y_train, X_test, y_test,  data_name, algorithm_selected):
    model = XGBClassifier()
    model.fit(X_train,y_train)
    model_pred = model.predict(X_test)

    return metrics(X_test, y_test, model_pred, model, data_name, algorithm_selected)

def Gradient_Classifier(X_train, y_train, X_test, y_test,  data_name, algorithm_selected):
    model = GradientBoostingClassifier()
    model.fit(X_train,y_train)
    model_pred = model.predict(X_test)

    return metrics(X_test, y_test, model_pred, model,  data_name, algorithm_selected)
   
     
def Support_vector_classifier(X_train, y_train, X_test, y_test, data_name, algorithm_selected):
    model = SVC(probability=True)
    model.fit(X_train,y_train)
    model_pred = model.predict(X_test)
    return metrics(X_test, y_test, model_pred, model,  data_name, algorithm_selected)

def Random_forest_classifier(X_train, y_train, X_test, y_test, data_name, algorithm_selected):
    model = RandomForestClassifier()
    model.fit(X_train,y_train)
    model_pred = model.predict(X_test)

    return metrics(X_test, y_test, model_pred, model, data_name, algorithm_selected)
   
# def Catboost_classifier(X_train, y_train, X_test, y_test, data_name, algorithm_selected):
#     model = CatBoostClassifier()
#     model.fit(X_train,y_train)
#     model_pred = model.predict(X_test)

#     return metrics(X_test, y_test, model_pred, model, data_name, algorithm_selected)
   
def Adaboost_classifier(X_train, y_train, X_test, y_test, data_name, algorithm_selected):
    model = AdaBoostClassifier()
    model.fit(X_train,y_train)
    model_pred = model.predict(X_test)

    return metrics(X_test, y_test, model_pred, model, data_name, algorithm_selected)
     
def Decision_tree_classifier(X_train, y_train, X_test, y_test, data_name, algorithm_selected):
    model = DecisionTreeClassifier()
    model.fit(X_train,y_train)
    model_pred = model.predict(X_test)
    return metrics(X_test, y_test, model_pred, model, data_name, algorithm_selected)

def K_neighnors_classifier(X_train, y_train, X_test, y_test, data_name, algorithm_selected):
    model = KNeighborsClassifier()
    
    model.fit(X_train,y_train)
    model_pred = model.predict(X_test)
    return metrics(X_test, y_test, model_pred, model, data_name, algorithm_selected)
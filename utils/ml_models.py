import numpy as np
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression

def Logistic_regression(X_train, y_train, X_test, y_test):
    logistic = LogisticRegression()
    #tuning parameters
    parameter = {
    'penalty' : ['l1','l2','elasicnet','none'],
    'C' : np.linspace(0.5,2,12),
    'solver' : ['newton-cg','lbfgs','liblinear','sag','saga'],
    'max_iter' : np.linspace(50,200,4),
    'multi_class' : ['ovr','multinomial'],
    'l1_ratio' : np.linspace(0,1,10)}
    
    tuner = RandomizedSearchCV(logistic, parameter, random_state=1, n_iter=100, cv=5, verbose=0, n_jobs=-1)
    tuner = tuner.fit(X_train,y_train)
    model = tuner.best_estimator_
    #tuned model
    model.fit(X_train,y_train)
    model_pred = model.predict(X_test)

    return plot_confusion_matrix(estimator=tuner.best_estimator_, X=X_test, y_true=y_test, labels=model.classes_, cmap='Blues')
   

def Linear_regression(X_train, y_train, X_test, y_test):
    linear = LinearRegression()
    #tuning parameters
    param_test ={'fit_intercept': [True, False],
             'normalize': [True, False], 
             'n_jobs': [-1,0,1]
             }
    
    tuner = RandomizedSearchCV(estimator=linear, param_distributions=param_test, scoring='explained_variance',cv=5,n_iter = 100,refit=True,random_state=42,verbose=True)
    tuner = tuner.fit(X_train,y_train)
    model = tuner.best_estimator_
    #tuned model
    model.fit(X_train,y_train)
    model_pred = model.predict(X_test)

    return 'It is a linear algorithm, therefore, cannot display confusion matrix.'
     
def Support_vector_classifier(X_train, y_train, X_test, y_test):
    svc = SVC(probability=True)
    #tuning parameters
    parameter = {
    'C' : np.linspace(0.5,2,12),
    'kernel' : ['rbf'],
    'gamma' : ['scale','auto'],
    'max_iter' : np.linspace(50,200,4)}
    
    tuner = RandomizedSearchCV(svc, parameter, random_state=1,scoring='accuracy', n_iter=20, cv=5, verbose=1, n_jobs=-1)
    tuner = tuner.fit(X_train,y_train)
    model = tuner.best_estimator_
    #tuned model
    model.fit(X_train,y_train)
    model_pred = model.predict(X_test)
    return plot_confusion_matrix(estimator=tuner.best_estimator_, X=X_test, y_true=y_test, labels=model.classes_, cmap='Blues')


def Random_forest_classifier(X_train, y_train, X_test, y_test):
    forest = RandomForestClassifier()
    #tuning parameters
    parameter = {'n_estimators': [int(i) for i in np.linspace(start = 200, stop = 1600, num = 10)],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [int(i) for i in np.linspace(10, 120, num = 12)],
               'min_samples_split': [2, 5, 10, 15],
               'min_samples_leaf': [1, 2, 3, 4],
               'bootstrap': [True, False]}
    
    tuner = RandomizedSearchCV(estimator=forest, param_distributions=parameter, random_state=42, n_iter=20, cv=5, verbose=1, scoring='accuracy', n_jobs=-1)
    tuner = tuner.fit(X_train,y_train)
    model = tuner.best_estimator_
    #tuned model
    model.fit(X_train,y_train)
    model_pred = model.predict(X_test)

    return plot_confusion_matrix(estimator=tuner.best_estimator_, X=X_test, y_true=y_test, labels=model.classes_, cmap='Blues')
   
def Catboost_classifier(X_train, y_train, X_test, y_test):
    catboost = CatBoostClassifier()
    #tuning parameters
    param_list = {'iterations': np.linspace(10, 1000, 5),
              'depth': np.linspace(1, 8, 4),
              'learning_rate': np.linspace(0.01, 1.0, 10),
              'random_strength': np.linspace(1e-9, 10, 10),
              'bagging_temperature': np.linspace(0.0, 1.0, 5),
              'l2_leaf_reg': np.linspace(2, 30, 6),
              'scale_pos_weight':np.linspace(0.01, 1.0, 5)}
    
    tuner = RandomizedSearchCV(estimator = catboost,
                         param_distributions=param_list,
                         verbose=1,
                         n_iter=20,
                         scoring='accuracy',
                         cv=5, random_state=42)
    tuner = tuner.fit(X_train,y_train)
    model = tuner.best_estimator_
    #tuned model
    model.fit(X_train,y_train)
    model_pred = model.predict(X_test)

    return plot_confusion_matrix(estimator=tuner.best_estimator_, X=X_test, y_true=y_test, labels=model.classes_, cmap='Blues')
   
def Adaboost_classifier(X_train, y_train, X_test, y_test):
    adaboost = AdaBoostClassifier()
    #tuning parameters
    param_list = {'n_estimators':[10, 50, 100, 200, 500, 1000, 1500, 2000],
              'learning_rate':(0.001, 0.01,0.1, 1),
              'base_estimator__max_depth':(2,4, 6, 8, 10, 3, 5, 7, 9),
              'base_estimator__min_samples_leaf':(5,10, 15, 20)
              }
    
    tuner = RandomizedSearchCV(estimator = adaboost,
                         param_distributions=param_list,
                         verbose=1,
                         n_iter=20,
                         scoring='accuracy',
                         cv=5, random_state=42)
    tuner = tuner.fit(X_train,y_train)
    model = tuner.best_estimator_
    #tuned model
    model.fit(X_train,y_train)
    model_pred = model.predict(X_test)

    return plot_confusion_matrix(estimator=tuner.best_estimator_, X=X_test, y_true=y_test, labels=model.classes_, cmap='Blues')
   
   
     
def Decision_tree_classifier(X_train, y_train, X_test, y_test):
    tree = DecisionTreeClassifier()
    #tuning parameters
    params = {
    'max_depth': [2, 3, 4, 5, 10, 15, 20],
    'min_samples_leaf': [5, 10, 20, 30, 50, 70, 100],
    'criterion': ["gini", "entropy"]}
    
    tuner = RandomizedSearchCV(estimator=tree, param_distributions=params, random_state=42, n_iter=20, cv=5, verbose=0, scoring='accuracy', n_jobs=-1)
    tuner = tuner.fit(X_train,y_train)
    model = tuner.best_estimator_
    #tuned model
    model.fit(X_train,y_train)
    model_pred = model.predict(X_test)
    return plot_confusion_matrix(estimator=tuner.best_estimator_, X=X_test, y_true=y_test, labels=model.classes_, cmap='Blues')

def K_neighnors_classifier(X_train, y_train, X_test, y_test):
    neighbors = KNeighborsClassifier()
    #tuning parameters
    params = {'n_neighbors': [3, 4, 5, 7, 9, 11, 13, 15], 
              'weights': ['uniform', 'distance'],
              'algorithm' : ['auto', 'kd_tree', 'ball_tree', 'brute'],
              'leaf_size' : [10, 20, 30, 40, 50, 60],
              'p': [1,2]
             }

    
    tuner = RandomizedSearchCV(estimator=neighbors, param_distributions=params, random_state=42, n_iter=20, cv=5, verbose=0, scoring='accuracy', n_jobs=-1)
    tuner = tuner.fit(X_train,y_train)
    model = tuner.best_estimator_
    #tuned model
    model.fit(X_train,y_train)
    model_pred = model.predict(X_test)
    return plot_confusion_matrix(estimator=tuner.best_estimator_, X=X_test, y_true=y_test, labels=model.classes_, cmap='Blues')


def Naive_bayes_classifier(X_train, y_train, X_test, y_test):
    bayes = GaussianNB()
    #tuning parameters
    params = {'var_smoothing': np.logspace(0,-10, num=100)}

    
    tuner = RandomizedSearchCV(estimator=bayes, param_distributions=params, random_state=42, n_iter=20, cv=5, verbose=1, scoring='accuracy', n_jobs=-1)
    tuner = tuner.fit(X_train,y_train)
    model = tuner.best_estimator_
    #tuned model
    model.fit(X_train,y_train)
    model_pred = model.predict(X_test)
    return plot_confusion_matrix(estimator=tuner.best_estimator_, X=X_test, y_true=y_test, labels=model.classes_, cmap='Blues')


def K_means_classifier(X_train, y_train, X_test, y_test):
    kmeans = KMeans()
    #tuning parameters
    params = {'n_clusters': [3, 5, 7, 8, 9, 10, 11, 12, 13, 15],
              'max_iter': [50, 100, 150, 200, 250, 300, 400],
              'n_init': [4,5,6,7,8,9,10],
              'init': ['kmeans++', 'random'],
              'precompute_distances': ['auto', True, False]}

    
    tuner = RandomizedSearchCV(estimator=kmeans, param_distributions=params, random_state=42, n_iter=20, cv=5, verbose=1, scoring='accuracy', n_jobs=-1)
    tuner = tuner.fit(X_train,y_train)
    model = tuner.best_estimator_
    #tuned model
    model.fit(X_train,y_train)
    model_pred = model.predict(X_test)
    return 'Clustering algorithm is unable to display Confusion matrix'#plot_confusion_matrix(estimator=tuner.best_estimator_, X=X_test, y_true=y_test, labels=model.labels_, cmap='Blues')
    

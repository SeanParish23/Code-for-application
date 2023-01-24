# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 20:03:38 2022

@author: Shade
"""
#%% Machine Learning Algorithms Function
def project1_all(datapath):
    #%% Importing Libraries
    
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sn
    import random as rd
    import pydotplus
    import matplotlib.image as pltimg
    from sklearn.datasets import load_digits
    from sklearn.linear_model import Perceptron
    from sklearn import tree
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import svm
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import BaggingRegressor
    from math import sqrt
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score
    from warnings import filterwarnings 
    from random import choice
    from pylab import ylim, plot
    r'C:\Users\Shade\.spyder-py3\heart1.csv'
    #%% Reading in data and providing information
    df = pd.read_csv(datapath)
    
    # Creating Correlation Matrix
    correlation_matrix = df.corr()
    
    column_names = list(df.columns)
    
    # User can see correlation in relation to the last column which should be 
    # column to be predicted
    
    print('Correlation of each feature in relation to %s:\n%s'%(column_names[-1], correlation_matrix[column_names[-1]]))
    
    # algorithm_to_be_used = input('Please provide algorithm to be used for training: \n')
    
    #%% Perceptron
    
    
    #NOT WORKING YET
    
    # if algorithm_to_be_used == 'Perceptron':
    
    # create training and test sets
        
        
    features = list(map(str,input("\nPlease enter features to be used for training: ").strip().split()))
    
    print("features being used: %s\n" %features)
    
    x = df[features]
    
    y = df[df.columns[-1]]
    
    x_train, x_test, y_train, y_test =\
        train_test_split(x, y, test_size=0.3, random_state=0)
    # Create Model
    
    clf = Perceptron(tol=1e-3, random_state=0, )
    clf.fit(x_train, y_train)
    
    
    # Evaluate Model
    
    y_pred = clf.predict(x_test)
    
    if len(set(y_pred)) == 1:
        print('\nFeatures chosen did not fulfill conditions for model and results will be inconclusive\n')
    
    # Confusion Matrix
    
    cm = confusion_matrix(y_test, y_pred)
    
    print('\nConfusion Matrix:\n%s'%cm)
    
    # Accuracy score
    
    Perceptron_accuracy = accuracy_score(y_test, y_pred)
    
    print("\nPerceptron \nAccuracy (normalized): " + str(Perceptron_accuracy)+'\n')
    
    report = classification_report(y_test, y_pred)
    
    print(report)
        
    # plot predicted vs truth data
    plt.plot(y_pred,color = 'green', label = 'predicted values', marker = 'o', markerfacecolor = 'red')
    plt.plot(y_test.values,color = 'blue', label = 'test values', marker = 's', markerfacecolor = 'purple')
    plt.title('Perceptron')
    plt.xlabel('Data Points')
    plt.ylabel('Absence or Presence of Heart Disease')
    plt.legend()
    plt.show()
    
    #%% Logistic Regression
    
    
    # elif algorithm_to_be_used == 'Logistic Regression':
        # Read in data and create training and test sets
    
    
    
    # features = list(map(str,input("\nPlease enter features to be used for training: ").strip().split()))
    
    # print("features being used: %s\n" %features)
    
    x = df[features]
    
    y = df[df.columns[-1]]
    
    x_train, x_test, y_train, y_test =\
        train_test_split(x, y, test_size=0.3, random_state=0)
    
    
    # Create Model
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    
    model = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
    model.fit(x_train, y_train)
    
    # Evaluate Model
    
    x_test = scaler.transform(x_test)
    
    y_pred = model.predict(x_test)
    
    if len(set(y_pred)) == 1:
        print('\nFeatures chosen did not fulfill conditions for model and results will be inconclusive\n')
    
    # Confusion Matrix
    
    cm = confusion_matrix(y_test, y_pred)
    
    print('\nConfusion Matrix:\n%s'%cm)
    
    # Print Report of results
    
    Logistic_accuracy = accuracy_score(y_test, y_pred)
    
    print("\nLogistic Regression \nAccuracy (normalized): " + str(Logistic_accuracy)+'\n')
    
    report = classification_report(y_test, y_pred)
    
    print(report)
    
    # plot predicted vs truth data
    plt.plot(y_pred,color = 'green', label = 'predicted values', marker = 'o', markerfacecolor = 'red')
    plt.plot(y_test.values,color = 'blue', label = 'test values', marker = 's', markerfacecolor = 'purple')
    plt.title('Logistic Regresion')
    plt.xlabel('Data Points')
    plt.ylabel('Absence or Presence of Heart Disease')
    plt.legend()
    plt.show()
    
    #%% Support Vector Machine
    
    # elif algorithm_to_be_used == 'Support Vector Machine':
    
        # Read in data and create training and test sets
        
        
    # features = list(map(str,input("\nPlease enter features to be used for training: ").strip().split()))
    
    # print("features being used: %s" %features)
    
    x = df[features]
    
    y = df[df.columns[-1]]
    
    x_train, x_test, y_train, y_test =\
        train_test_split(x, y, test_size=0.3, random_state=0)
        
    # Create Model
    
    model = svm.SVC(kernel='poly', degree=2)
    model.fit(x_train, y_train)
    
    # Evaluate Model
    
    y_pred = model.predict(x_test)
    
    if len(set(y_pred)) == 1:
        print('\nFeatures chosen did not fulfill conditions for model and results will be inconclusive\n')
    
    # Confusion Matrix
    
    cm = confusion_matrix(y_test, y_pred)
    
    print('\nConfusion Matrix:\n%s'%cm)
    
    # Accuracy score
    SVM_accuracy = accuracy_score(y_test, y_pred)
    print("2nd degree polynomial Kernel\nAccuracy (normalized): " + str(SVM_accuracy) + '\n')
    

    
    report = classification_report(y_test, y_pred)
    
    print(report)
    
    # plot predicted vs truth data
    
    plt.plot(y_pred,color = 'green', label = 'predicted values', marker = 'o', markerfacecolor = 'red')
    plt.plot(y_test.values,color = 'blue', label = 'test values', marker = 's', markerfacecolor = 'purple')
    plt.title('SVM')
    plt.xlabel('Data Points')
    plt.ylabel('Absence or Presence of Heart Disease')
    plt.legend()
    plt.show()
    
    #%% Decision Tree learning
    
    # elif algorithm_to_be_used == 'Decision Tree':

    # features = list(map(str,input("\nPlease enter features to be used for training: ").strip().split()))
    
    # print("features being used: %s" %features)
    
    x = df[features]
    
    y = df[df.columns[-1]]
    
    
    x_train, x_test, y_train, y_test =\
        train_test_split(x, y, test_size=0.3, random_state=0)
    # Create Model
    
    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(x_train, y_train)
    
    
    # Decision Tree Visualization
    data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
    graph = pydotplus.graph_from_dot_data(data)
    graph.write_png('mydecisiontree.png')
    
    img=pltimg.imread('mydecisiontree.png')
    imgplot = plt.imshow(img)
    plt.show()
    
    # Evaluate Model
    y_pred = dtree.predict(x_test)
    
    if len(set(y_pred)) == 1:
        print('\nFeatures chosen did not fulfill conditions for model and results will be inconclusive\n')
    
    # Confusion Matrix
    
    cm = confusion_matrix(y_test, y_pred)
    
    print('\nConfusion Matrix:\n%s'%cm)
    
    #Accuracy score
    dtree_accuracy = accuracy_score(y_test, y_pred)
    
    print("\nDecision Tree\nAccuracy (normalized): " + str(dtree_accuracy)+'\n')
    
    report = classification_report(y_test, y_pred)
    
    print(report)
    
    # plot predicted vs truth data
    
    plt.plot(y_pred,color = 'green', label = 'predicted values', marker = 'o', markerfacecolor = 'red')
    plt.plot(y_test.values,color = 'blue', label = 'test values', marker = 's', markerfacecolor = 'purple')
    plt.title('Decision Tree')
    plt.xlabel('Data Points')
    plt.ylabel('Absence or Presence of Heart Disease')
    plt.legend()
    plt.show()
    #%% Random Forest
    
    # elif algorithm_to_be_used == 'Random Forest':
    
    # Read in data and create training and test sets
    
    
    # features = list(map(str,input("\nPlease enter features to be used for training: ").strip().split()))
    
    # print("features being used: %s" %features)
    
    x = df[features]
    
    y = df[df.columns[-1]]
    
    x_train, x_test, y_train, y_test =\
        train_test_split(x, y, test_size=0.3, random_state=0)
        
    # Create Model
    
    regressor = RandomForestClassifier(n_estimators = 100, random_state = 0)
    
    regressor.fit(x_train,y_train)
    
    #Evaluate Model
    
    y_pred = np.round(regressor.predict(x_test))
    
    if len(set(y_pred)) == 1:
        print('\nFeatures chosen did not fulfill conditions for model and results will be inconclusive\n')
    
    # Confusion Matrix
    
    cm = confusion_matrix(y_test, y_pred)
    
    print('\nConfusion Matrix:\n%s'%cm)
    
    #Accuracy score
    Regressor_accuracy = accuracy_score(y_test, y_pred)
    
    print("\nRandom Forest\nAccuracy (normalized): " + str(Regressor_accuracy)+'\n')
    
    report = classification_report(y_test, y_pred)
    
    print(report)
     
    # # Scatter plot for original data
    # plt.scatter(x_test, y_test, color = 'blue') 
     
    # plot predicted vs truth data
    
    plt.plot(y_pred,color = 'green', label = 'predicted values', marker = 'o', markerfacecolor = 'red')
    plt.plot(y_test.values,color = 'blue', label = 'test values', marker = 's', markerfacecolor = 'purple')
    plt.title('Random Forest Regression')
    plt.xlabel('Data Points')
    plt.ylabel('Absence or Presence of Heart Disease')
    plt.legend()
    plt.show()
    
    #%% K-Nearest Neighbor
    
    # elif algorithm_to_be_used == 'K-Nearest Neighbor':
    
    
    # Read in data and create training and test sets
    
    
    # features = list(map(str,input("\nPlease enter features to be used for training: ").strip().split()))
    
    # print("features being used: %s" %features)
    
    x = df[features]
    
    y = df[df.columns[-1]]
    
    x_train, x_test, y_train, y_test =\
        train_test_split(x, y, test_size=0.3, random_state=0)
        
    # Create Model
    
    # Use grid search for best neighbor value and weights
    
    parameters =  {"n_neighbors": range(1, 50), "weights": ["uniform", "distance"] }
    
    knn = GridSearchCV(KNeighborsRegressor(), parameters)
    
    knn.fit(x_train,y_train)
    
    best_neighbors = knn.best_params_["n_neighbors"]
    
    best_weights = knn.best_params_["weights"]
    
    # Use bagging to improve model further 
    
    bagged_knn = KNeighborsRegressor(n_neighbors = best_neighbors, weights = best_weights)
    
    bagging_model = BaggingRegressor(bagged_knn, n_estimators=100)
    
    bagging_model.fit(x_train,y_train)
    
    # Evaluate Model
    
    y_pred = np.round(bagging_model.predict(x_test))
    
    if len(set(y_pred)) == 1:
        print('\nFeatures chosen did not fulfill conditions for model and results will be inconclusive\n')
    
    # Confusion Matrix
    
    cm = confusion_matrix(y_test, y_pred)
    
    print('\nConfusion Matrix:\n%s'%cm)
    
    # Accuracy score
    knn_accuracy = accuracy_score(y_test, y_pred)
    
    print("\nKnn\nAccuracy (normalized): " + str(knn_accuracy)+'\n')
    
    report = classification_report(y_test, y_pred)
    
    print(report)
    
    # plot predicted vs truth data
    
    plt.plot(y_pred,color = 'green', label = 'predicted values', marker = 'o', markerfacecolor = 'red')
    plt.plot(y_test.values,color = 'blue', label = 'test values', marker = 's', markerfacecolor = 'purple')
    plt.title('Knn Regression')
    plt.xlabel('Data Points')
    plt.ylabel('Absence or Presence of Heart Disease')
    plt.legend()
    plt.show()
    
    #%% Algorithm choice not supported
    # else:
        # print('Algorithm input provided does not match supported algorithms please try again.\n')
        # return(project1(datapath))
    # user_input = input("\nPrediction completed, type 'ENTER' to try another algorithm or 'END' to terminate\n")
    # if user_input.lower() == "end":
    #     return
    # if user_input.lower() == "enter":
    #     return(project1(datapath))

#%% Ensemble function
def ensemble_3(datapath):
    # importing utility modules and machine learning modules
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sn
    import random as rd
    import pydotplus
    import matplotlib.image as pltimg
    from sklearn.datasets import load_digits
    from sklearn.linear_model import Perceptron
    from sklearn import tree
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import svm
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import BaggingRegressor
    from math import sqrt
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score
    from warnings import filterwarnings 
    from random import choice
    from pylab import ylim, plot
    from sklearn.metrics import log_loss
    from sklearn.ensemble import RandomForestClassifier
     
    # importing voting classifier
    from sklearn.ensemble import VotingClassifier
    
    # Reading in data and providing information
    df = pd.read_csv(datapath)
    
    # Creating Correlation Matrix
    correlation_matrix = df.corr()
    
    column_names = list(df.columns)
    
    print('Correlation of each feature in relation to %s:\n%s'%(column_names[-1], correlation_matrix[column_names[-1]]))
    
    features = list(map(str,input("\nPlease enter features to be used for training: ").strip().split()))
    
    print("features being used: %s\n" %features)
    
    x = df[features]
    
    y = df[df.columns[-1]]
    
    x_train, x_test, y_train, y_test =\
        train_test_split(x, y, test_size=0.3, random_state=0)
    
    # initializing all the model objects with default parameters
    model_1 = LogisticRegression()
    model_2 = Perceptron()
    model_3 = DecisionTreeClassifier()
    
    # Making the final model using voting classifier
    final_model = VotingClassifier(
    estimators=[('lr', model_1), ('pct', model_2), ('dct', model_3)], voting='hard')
 
    # training all the model on the train dataset
    final_model.fit(x_train, y_train)
     
    # predicting the output on the test dataset
    pred_final = final_model.predict(x_test)
    
    # Accuracy score
    ensemble_3_prong = accuracy_score(y_test, pred_final)
    
    print("\n3 stage ensemble \nAccuracy (normalized): " + str(ensemble_3_prong)+'\n')
    
    return
#%% Ensemble 4
def ensemble_4(datapath):
    # importing utility modules and machine learning modules
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sn
    import random as rd
    import pydotplus
    import matplotlib.image as pltimg
    from sklearn.datasets import load_digits
    from sklearn.linear_model import Perceptron
    from sklearn import tree
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import svm
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import BaggingRegressor
    from math import sqrt
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score
    from warnings import filterwarnings 
    from random import choice
    from pylab import ylim, plot
    from sklearn.metrics import log_loss
    from sklearn.ensemble import RandomForestClassifier
     
    # importing voting classifier
    from sklearn.ensemble import VotingClassifier
    
    # Reading in data and providing information
    df = pd.read_csv(datapath)
    
    # Creating Correlation Matrix
    correlation_matrix = df.corr()
    
    column_names = list(df.columns)
    
    print('Correlation of each feature in relation to %s:\n%s'%(column_names[-1], correlation_matrix[column_names[-1]]))
    
    features = list(map(str,input("\nPlease enter features to be used for training: ").strip().split()))
    
    print("features being used: %s\n" %features)
    
    x = df[features]
    
    y = df[df.columns[-1]]
    
    x_train, x_test, y_train, y_test =\
        train_test_split(x, y, test_size=0.3, random_state=0)
    
    # initializing all the model objects with default parameters
    model_1 = LogisticRegression()
    model_2 = Perceptron()
    model_3 = DecisionTreeClassifier()
    model_4 = RandomForestClassifier()
    
    
    # Making the final model using voting classifier
    final_model = VotingClassifier(
    estimators=[('lr', model_1), ('pct', model_2), ('dct', model_3),('rf', model_4)], voting='hard')
 
    # training all the model on the train dataset
    final_model.fit(x_train, y_train)
     
    # predicting the output on the test dataset
    pred_final = final_model.predict(x_test)
    
    # Accuracy score
    ensemble_4_prong = accuracy_score(y_test, pred_final)
    
    print("\n4 stage ensemble \nAccuracy (normalized): " + str(ensemble_4_prong)+'\n')
    
    print("\n Ties were not counted as Decision Trees have different results everytime leading to the possibility of a more accurate model\n")
    
    return
#%% Ensemble 5
def ensemble_5(datapath):
    # importing utility modules and machine learning modules
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sn
    import random as rd
    import pydotplus
    import matplotlib.image as pltimg
    from sklearn.datasets import load_digits
    from sklearn.linear_model import Perceptron
    from sklearn import tree
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import svm
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import BaggingRegressor
    from math import sqrt
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score
    from warnings import filterwarnings 
    from random import choice
    from pylab import ylim, plot
    from sklearn.metrics import log_loss
    from sklearn.ensemble import RandomForestClassifier
     
    # importing voting classifier
    from sklearn.ensemble import VotingClassifier
    
    # Reading in data and providing information
    df = pd.read_csv(datapath)
    
    # Creating Correlation Matrix
    correlation_matrix = df.corr()
    
    column_names = list(df.columns)
    
    print('Correlation of each feature in relation to %s:\n%s'%(column_names[-1], correlation_matrix[column_names[-1]]))
    
    features = list(map(str,input("\nPlease enter features to be used for training: ").strip().split()))
    
    print("features being used: %s\n" %features)
    
    x = df[features]
    
    y = df[df.columns[-1]]
    
    x_train, x_test, y_train, y_test =\
        train_test_split(x, y, test_size=0.3, random_state=0)
    
    # initializing all the model objects with default parameters
    model_1 = LogisticRegression()
    model_2 = Perceptron()
    model_3 = DecisionTreeClassifier()
    model_4 = RandomForestClassifier()
    model_5 = KNeighborsClassifier()
    
    # Making the final model using voting classifier
    final_model = VotingClassifier(
    estimators=[('lr', model_1), ('pct', model_2), ('dct', model_3),('rf', model_4), ('knn',model_5)], voting='hard')
 
    # training all the model on the train dataset
    final_model.fit(x_train, y_train)
     
    # predicting the output on the test dataset
    pred_final = final_model.predict(x_test)
    
    # Accuracy score
    ensemble_5_prong = accuracy_score(y_test, pred_final)
    
    print("\n5 stage ensemble \nAccuracy (normalized): " + str(ensemble_5_prong)+'\n')
    
    
    return
#%% Main

# Insert datapath in each function call below the rest should run as expected

project1_all(r'C:\Users\Shade\.spyder-py3\heart1.csv')

ensemble_3(r'C:\Users\Shade\.spyder-py3\heart1.csv')

ensemble_4(r'C:\Users\Shade\.spyder-py3\heart1.csv')

ensemble_5(r'C:\Users\Shade\.spyder-py3\heart1.csv')
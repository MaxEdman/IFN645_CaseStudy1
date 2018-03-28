def preprocess():
    # Import pandas
    import pandas as pd
    df = pd.read_csv("datasets/organics.csv")
    
    # Drops irrelevant columns.
    df = df.drop(['CUSTID', 'LCDATE', 'ORGANICS', 'AGEGRP1', 'AGEGRP2', 'NEIGHBORHOOD'], axis = 1)
    
    # Calculates the years between DOB and EDATE and adds that value to age for missing values.
    from datetime import datetime
    import numpy as np
    dateformat = '%Y-%m-%d'
    edate = pd.Timestamp(df['EDATE'][0])
    df['DOB'] = pd.to_datetime(df['DOB'], format=dateformat)    # 1
    df['DOB'] = df['DOB'].where(df['DOB'] < edate, df['DOB'] -  np.timedelta64(100, 'Y'))   # 2
    df['AGE'] = (edate - df['DOB']).astype('<m8[Y]')    # 3
    # Drops Edate and Dob since the Age col is the only one needed.
    df = df.drop(['EDATE', 'DOB'], axis = 1)
    
    # denote errorneous values in AFFL column. Should be on scale 1-30.
    mask = df['AFFL'] < 1
    df.loc[mask, 'AFFL'] = 1
    mask = df['AFFL'] > 30
    df.loc[mask, 'AFFL'] = 30

    # Fill mean values for AFFL column.
    df['AFFL'].fillna(df['AFFL'].mean(), inplace=True)
    # Convert the scale to integer. Not sure if this is necessary.
    df['AFFL'] = df['AFFL'].astype(int)
    
    # Fills mean values based on age for loyalty time. 
    means = df.groupby(['AGE'])['LTIME'].mean()
    df = df.set_index(['AGE'])
    df['LTIME'] = df['LTIME'].fillna(means)
    df = df.reset_index()
    
    # Fills all unknown values of gender with U, which is the same thing basically.
    df['GENDER'].fillna('U', inplace=True)
    
    # One-hot-encodes columns with REGION, TV_REGION and NGROUP.
    df = pd.get_dummies(df)
    
    # Returns the dataframe.
    return df


def preprocess_david():
    # Import pandas
    import pandas as pd
    df = pd.read_csv("datasets/organics.csv")
    
    # Drops irrelevant columns.
    df = df.drop(['CUSTID', 'LCDATE', 'ORGANICS', 'AGEGRP1', 'AGEGRP2', 'NEIGHBORHOOD', 'TV_REG', 'NGROUP', 'CLASS'], axis = 1)
    
    # Calculates the years between DOB and EDATE and adds that value to age for missing values.
    from datetime import datetime
    import numpy as np
    dateformat = '%Y-%m-%d'
    edate = pd.Timestamp(df['EDATE'][0])
    df['DOB'] = pd.to_datetime(df['DOB'], format=dateformat)    # 1
    df['DOB'] = df['DOB'].where(df['DOB'] < edate, df['DOB'] -  np.timedelta64(100, 'Y'))   # 2
    df['AGE'] = (edate - df['DOB']).astype('<m8[Y]')    # 3
    # Drops Edate and Dob since the Age col is the only one needed.
    df = df.drop(['EDATE', 'DOB'], axis = 1)
    
    # denote errorneous values in AFFL column. Should be on scale 1-30.
    mask = df['AFFL'] < 1
    df.loc[mask, 'AFFL'] = 1
    mask = df['AFFL'] > 30
    df.loc[mask, 'AFFL'] = 30

    # Fill mean values for AFFL column.
    df['AFFL'].fillna(df['AFFL'].mean(), inplace=True)
    # Convert the scale to integer. Not sure if this is necessary.
    df['AFFL'] = df['AFFL'].astype(int)
    
    # Fills mean values based on age for loyalty time. 
    means = df.groupby(['AGE'])['LTIME'].mean()
    df = df.set_index(['AGE'])
    df['LTIME'] = df['LTIME'].fillna(means)
    df = df.reset_index()
    
    # Fills all unknown values of gender with U, which is the same thing basically.
    df['GENDER'].fillna('U', inplace=True)
    
    # One-hot-encodes columns with REGION, TV_REGION and NGROUP.
    df = pd.get_dummies(df)
    
    # Returns the dataframe.
    return df

# Function for visualizing decision tree
def visualize_decision_tree(dm_model, feature_names, save_name):
    import pydot
    from io import StringIO
    from sklearn.tree import export_graphviz
    
    dotfile = StringIO()
    export_graphviz(dm_model, out_file=dotfile, feature_names=feature_names)
    graph = pydot.graph_from_dot_data(dotfile.getvalue())
    graph.write_png(save_name) # saved in the following file
    
    return






    
def get_neural_networks_model(): 
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier

    rs = 10
    
     # Gets the preprocessed data set for Organics.
    df = preprocess()

    y = df['ORGYN']
    X = df.drop(['ORGYN'], axis=1)
    X_mat = X.as_matrix()
    X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.3, stratify=y, random_state=rs)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train, y_train)
    X_test = scaler.transform(X_test)
    
    params = {'hidden_layer_sizes': [(3)], 'alpha': [0.0001]}

    cv = GridSearchCV(param_grid=params, estimator=MLPClassifier(random_state=rs), cv=10, n_jobs=-1)
    cv.fit(X_train, y_train)

    y_pred = cv.predict(X_test)
    print("Neural Network Model Statistics:")
    print("Train accuracy:", cv.score(X_train, y_train))
    print("Test accuracy:", cv.score(X_test, y_test))

    y_pred = cv.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return cv.best_estimator_
    
    
def get_decision_tree():  
    # Gets the preprocessed data set for Organics.
    df = preprocess()
    
    # Import necssary packages
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.preprocessing import StandardScaler

    # Sets target column to ORGYN
    target_dataset = df['ORGYN']
    # Removes ORGYN from the dataset in order to avoid false predictor.
    dataset = df.drop(['ORGYN'], axis=1)

    # Sets random state to 10. This will be kept consistently throughout the case study.
    random_state = 10
    # Sets the test size to be 30% of the total data set.
    test_size = 0.3

    # Transform the dataset into a matrix.
    dataset_matrix = dataset.as_matrix()

    # Splits the data into train and test sets.
    dataset_train, dataset_test, target_dataset_train, target_dataset_test = train_test_split(dataset_matrix,
                                                                                              target_dataset,
                                                                                              test_size=test_size,
                                                                                              stratify=target_dataset,
                                                                                              random_state=random_state
                                                                                             )
    
    # GridSearchCV parameters
    params = {'criterion': ['gini', 'entropy'],
              'max_depth': range(2, 5),
              'min_samples_leaf': range(1, 2)}

    cross_validation_optimal_model = GridSearchCV(param_grid=params,
                                          estimator=DecisionTreeClassifier(random_state=random_state),
                                          cv=10)
    cross_validation_optimal_model.fit(dataset_train, target_dataset_train)

    train_accuracy_optimal_cv = cross_validation_optimal_model.score(dataset_train, target_dataset_train)
    test_accuracy_optimal_cv = cross_validation_optimal_model.score(dataset_test, target_dataset_test)

    # test the best model
    target_prediction = cross_validation_optimal_model.predict(dataset_test)
    
        # Prints train and test accuracy.
    print("CV Tuned Decision Tree Statistics:")
    print("Train Accuracy:", cross_validation_optimal_model.score(dataset_train, target_dataset_train))
    print("Test Accuracy:", cross_validation_optimal_model.score(dataset_test, target_dataset_test))

    # Printing a classification report of the model.
    print("")
    print("Classification Report:")
    target_predict = cross_validation_optimal_model.predict(dataset_test)
    print(classification_report(target_dataset_test, target_predict))
    print("Number of nodes in the decision tree:", cross_validation_optimal_model.best_estimator_.tree_.node_count)
    
    return cross_validation_optimal_model.best_estimator_






def get_decision_tree_david_special():  
    # Gets the preprocessed data set for Organics.
    df = preprocess_david()
    
    df = df.drop(['BILL', 'LTIME', 'REGION_Midlands', 'REGION_Scottish', 'REGION_South East', 'REGION_South West'], axis = 1)
    print(df.info())
    
    # Import necssary packages
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.preprocessing import StandardScaler

    # Sets target column to ORGYN
    target_dataset = df['ORGYN']
    # Removes ORGYN from the dataset in order to avoid false predictor.
    dataset = df.drop(['ORGYN'], axis=1)

    # Sets random state to 10. This will be kept consistently throughout the case study.
    random_state = 10
    # Sets the test size to be 30% of the total data set.
    test_size = 0.3

    # Transform the dataset into a matrix.
    dataset_matrix = dataset.as_matrix()

    # Splits the data into train and test sets.
    dataset_train, dataset_test, target_dataset_train, target_dataset_test = train_test_split(dataset_matrix,
                                                                                              target_dataset,
                                                                                              test_size=test_size,
                                                                                              stratify=target_dataset,
                                                                                              random_state=random_state
                                                                                             )
    
    # GridSearchCV parameters
    params = {'criterion': ['gini', 'entropy'],
              'max_depth': range(2, 5),
              'min_samples_leaf': range(1, 2)}

    cross_validation_optimal_model = GridSearchCV(param_grid=params,
                                          estimator=DecisionTreeClassifier(random_state=random_state),
                                          cv=10)
    cross_validation_optimal_model.fit(dataset_train, target_dataset_train)

    train_accuracy_optimal_cv = cross_validation_optimal_model.score(dataset_train, target_dataset_train)
    test_accuracy_optimal_cv = cross_validation_optimal_model.score(dataset_test, target_dataset_test)

    # test the best model
    target_prediction = cross_validation_optimal_model.predict(dataset_test)
    
        # Prints train and test accuracy.
    print("David's Special Decision Tree Statistics:")
    print("Train Accuracy:", cross_validation_optimal_model.score(dataset_train, target_dataset_train))
    print("Test Accuracy:", cross_validation_optimal_model.score(dataset_test, target_dataset_test))

    # Printing a classification report of the model.
    print("")
    print("Classification Report:")
    target_predict = cross_validation_optimal_model.predict(dataset_test)
    print(classification_report(target_dataset_test, target_predict))
    print("Number of nodes in the decision tree:", cross_validation_optimal_model.best_estimator_.tree_.node_count)
    
    return cross_validation_optimal_model.best_estimator_


def get_logistic_regression_model(): 
    
    
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.model_selection import GridSearchCV
    from casestudy_tools import preprocess
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    
    # preprocessing step
    df = preprocess()

    # random state
    rs = 10
    
    # train test split
    y = df['ORGYN']
    X = df.drop(['ORGYN'], axis=1)
    X_mat = X.as_matrix()
    X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.3, stratify=y, random_state=rs)


    # initialise a standard scaler object
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train, y_train)
    X_test = scaler.transform(X_test)
    
    df_log = df.copy()
        
    #Create X, y and train test data partitions
    # create X, y and train test data partitions
    y_log = df_log['ORGYN']
    X_log = df_log.drop(['ORGYN'], axis=1)
    X_mat_log = X_log.as_matrix()
    X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_mat_log, y_log, test_size=0.3, stratify=y_log, 
                                                                        random_state=rs)

    # standardise them again
    scaler_log = StandardScaler()
    X_train_log = scaler_log.fit_transform(X_train_log, y_train_log)
    X_test_log = scaler_log.transform(X_test_log)
    
    print("Using RFECV")
    #Q3 Feature Transformation
    from sklearn.feature_selection import RFECV
    rfe = RFECV(estimator = LogisticRegression(random_state=rs), cv=10)
    rfe.fit(X_train, y_train) # run the RFECV
    
    print("RFECV")
    
    #test the performance
    X_train_sel = rfe.transform(X_train)
    X_test_sel = rfe.transform(X_test)
    
    from sklearn.tree import DecisionTreeClassifier
    from casestudy_tools import get_decision_tree
    #from casestudy_tools import analyse_feature_importance

    params = {'criterion': ['gini', 'entropy'],
              'max_depth': range(2, 5),
              'min_samples_leaf': range(1,2)}

    cv = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(random_state=rs), cv=10)
    cv.fit(X_train_log, y_train_log)

    
    from sklearn.feature_selection import SelectFromModel

    # use the trained best decision tree from GridSearchCV to select features
    # supply the prefit=True parameter to stop SelectFromModel to re-train the model
    selectmodel = SelectFromModel(cv.best_estimator_, prefit=True)
    X_train_sel_model = selectmodel.transform(X_train_log)
    X_test_sel_model = selectmodel.transform(X_test_log)
    
    # Grid search cv for RFE SELECTION MODEL (BEST MODEL)
    params = {'C': [pow(10, x) for x in range(-6, 4)]}

    cv = GridSearchCV(param_grid=params, estimator=LogisticRegression(random_state=rs), cv=10, n_jobs=-1)
    cv.fit(X_train_sel_model, y_train_log)
    
    print("Logistic Regression Model Statistics:")
    print("Train accuracy:", cv.score(X_train_sel_model, y_train_log))
    print("Test accuracy:", cv.score(X_test_sel_model, y_test_log))

    # test the best model
    y_pred = cv.predict(X_test_sel_model)
    print("Classification Report:")
    print(classification_report(y_test_log, y_pred))

    # print parameters of the best model
    print(cv.best_params_)


    return cv.best_estimator_



def visualise_all_models():
    
        
    df = preprocess()
    # Removes ORGYN from the dataset in order to avoid false predictor.
    dataset = df.drop(['ORGYN'], axis=1)
    
    # Visualises the models
    visualize_decision_tree(get_decision_tree(), dataset.columns, "Decision Tree Model - Task 2.png")
    visualize_decision_tree(get_logistic_regression_model(), dataset.columns, "Logistic Regression Model - Task 3.png")
    visualize_decision_tree(get_neural_networks_model(), dataset.columns, "Neural Network Model - Task 4.png")
    
def analyse_feature_importance(dm_model, feature_names, n_to_display=20):
    import numpy as np
    # grab feature importances from the model
    importances = dm_model.feature_importances_
    
    # sort them out in descending order
    indices = np.argsort(importances)
    indices = np.flip(indices, axis=0)

    # limit to 20 features, you can leave this out to print out everything
    indices = indices[:n_to_display]

    for i in indices:
        print(feature_names[i], ':', importances[i])
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
    
    print("Train accuracy:", cv.score(X_train, y_train))
    print("Test accuracy:", cv.score(X_test, y_test))

    y_pred = cv.predict(X_test)
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
    print("Default Decision Tree Statistics:")
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
    print("Default Decision Tree Statistics:")
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
    
    return null


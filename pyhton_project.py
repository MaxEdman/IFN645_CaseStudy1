# Gets the preprocessed data set for Organics.
import casestudy_tools as tools
df = tools.preprocess()
df.info()

# Building a decision tree using the default settings.
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

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

# Training a decision tree model based on deafault settings.
decisiontree_model_def = DecisionTreeClassifier(random_state=random_state)
decisiontree_model_def.fit(dataset_train, target_dataset_train)

# Prints train and test accuracy.
print("Default Decision Tree Statistics:")
print("Train Accuracy:", decisiontree_model_def.score(dataset_train, target_dataset_train))
print("Test Accuracy:", decisiontree_model_def.score(dataset_test, target_dataset_test))

# Printing a classification report of the model.
print("")
print("Classification Report:")
target_predict = decisiontree_model_def.predict(dataset_test)
print(classification_report(target_dataset_test, target_predict))
print("Number of nodes in the decision tree:", decisiontree_model_def.tree_.node_count)



# Evaluating the feature importance of the default_decision tree
import numpy as np

# Gets feature importance and relates to the column names of the model
feature_importances = decisiontree_model_def.feature_importances_
feature_names = dataset.columns

# Sorts the features
feature_indices = np.flip(np.argsort(feature_importances), axis=0)

# Prints the features
for i in feature_indices:
    print(feature_names[i], ':', feature_importances[i])
    
# Training a decision tree model based on deafault settings.
decisiontree_model_optimal = DecisionTreeClassifier(max_depth=5, random_state=random_state)
decisiontree_model_optimal.fit(dataset_train, target_dataset_train)

print(decisiontree_model_optimal)

train_accuracy = decisiontree_model_optimal.score(dataset_train, target_dataset_train)
test_accuracy = decisiontree_model_optimal.score(dataset_test, target_dataset_test)
# Prints train and test accuracy.
print("Decision Tree Statistics:")
print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
print("Accuracy Difference:", train_accuracy - test_accuracy)

# Printing a classification report of the model.
print("")
print("Classification Report:")
target_predict = decisiontree_model_optimal.predict(dataset_test)
print(classification_report(target_dataset_test, target_predict))
print("Number of nodes in the decision tree:", decisiontree_model_optimal.tree_.node_count)

# Gets feature importance and relates to the column names of the model
feature_importances = decisiontree_model_optimal.feature_importances_
feature_names = dataset.columns

# Sorts the features
feature_indices = np.flip(np.argsort(feature_importances), axis=0)

# Prints the features
for i in feature_indices:
    print(feature_names[i], ':', feature_importances[i])
    
# Visualising relationship between hyperparameters and model performance
import matplotlib.pyplot as plt
%matplotlib inline

# Sets the color to white.
params = {"ytick.color" : "w",
          "xtick.color" : "w",
          "axes.labelcolor" : "w",
          "axes.edgecolor" : "w"}
plt.rcParams.update(params)

test_score = []
train_score = []

# check the model performance for max depth from 2-20
for max_depth in range(2, 21):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(dataset_train, target_dataset_train)
    
    test_score.append(model.score(dataset_test, target_dataset_test))
    train_score.append(model.score(dataset_train, target_dataset_train))

# plot max depth hyperparameter values vs training and test accuracy score
plt.plot(range(2, 21), train_score, 'b', range(2,21), test_score, 'r')
plt.xlabel('max_depth\nBlue = training acc. Red = test acc.')
plt.ylabel('accuracy')
plt.show()

print("This shows that the optimal node depth of the decision tree is 5.")
print("Anything over a max depth of 5 is considered to overfit the model to the train data.")

from sklearn.model_selection import GridSearchCV

# grid search CV
params = {'criterion': ['gini', 'entropy'],
          'max_depth': range(2, 5),
          'min_samples_leaf': range(10, 11)}

cross_validation_optimal_model = GridSearchCV(param_grid=params,
                                      estimator=DecisionTreeClassifier(random_state=random_state),
                                      cv=10)
cross_validation_optimal_model.fit(dataset_train, target_dataset_train)

train_accuracy_optimal_cv = cross_validation_optimal_model.score(dataset_train, target_dataset_train)
test_accuracy_optimal_cv = cross_validation_optimal_model.score(dataset_test, target_dataset_test)
# Prints train and test accuracy.
print("Decision Tree Statistics:")
print("Train Accuracy:", train_accuracy_optimal_cv)
print("Test Accuracy:", test_accuracy_optimal_cv)
print("Accuracy Difference:", train_accuracy_optimal_cv - test_accuracy_optimal_cv)


# test the best model
target_prediction = cross_validation_optimal_model.predict(dataset_test)
print(classification_report(target_dataset_test, target_prediction))

# print parameters of the best model
print(cross_validation_optimal_model.best_params_)

print("Number of nodes in the decision tree:", cross_validation_optimal_model.best_estimator_.tree_.node_count)

# Gets feature importance and relates to the column names of the model
feature_importances_cv = cross_validation_optimal_model.best_estimator_.feature_importances_
feature_names_cv = dataset.columns

# Sorts the features
feature_indices_cv = np.flip(np.argsort(feature_importances_cv), axis=0)

# Prints the features
for i in feature_indices_cv:
    print(feature_names_cv[i], ':', feature_importances_cv[i])
    
    
    
    
    


    
    
    
    
    
    
    
    
    
###################################### TASK 3 ######################################
import pandas as pd
import numpy as np
import casestudy_tools as tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from casestudy_tools import preprocess
from sklearn.preprocessing import StandardScaler
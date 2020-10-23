from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist

def CTG_DTC(X_train, X_text, y_train, y_test):

    param_dist = {"max_depth": [3, None],
                  "max_features": randint(1, 9),
                  "min_samples_leaf": randint(1, 9),
                  "criterion": ["gini", "entropy"]}

    # Instantiate a Decision Tree classifier: tree
    tree = DecisionTreeClassifier()

    # Instantiate the RandomizedSearchCV object: tree_cv
    tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

    # Fit it to the data
    tree_cv.fit(X_train,y_train)

    # Print the tuned parameters and score
    print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
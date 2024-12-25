model_names = ['SVM', 'LogisticRegression', 'MultinomialNB',
               'DecisionTreeClassifier', 'KNeighborsClassifier', 'RandomForestClassifier']

hyperparameters = {
    'SVM': {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': [0.001, 0.01, 0.1],
        'degree': [2, 3, 4, 5],
        'max_iter': [100, 200, 300,-1],
    },
    'LogisticRegression': {
        'C': [1.0, 0.1, 10],
        'penalty': ['l2', 'l1', 'elasticnet', 'None'],
        'l1_ratio': [0.1, 0.5, 0.9],
        'solver': ['lbfgs', 'liblinear', 'saga', 'newton-cg'],
        'max_iter': [100, 200, 300, -1],
    },
    'MultinomialNB': {
        'alpha': [0.1, 0.7, 1.0, 2.0],
        'fit_prior': [True, False],
    },
    'DecisionTreeClassifier': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [10, 50, 100],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 5, 10],
    },
    'KNeighborsClassifier': {
        'n_neighbors': [5, 3, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['minkowski','euclidean', 'manhattan','cosine'],
    },
    'RandomForestClassifier': {
        'n_estimators': [50, 100, 150],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [-1, 10, 50, 100],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 5, 10],
    }
}


param_types = {
        "C": float,
        'kernel': str,
        "gamma": float,
        "degree": int,
        'penalty': str,
        'solver': str,
        'max_iter': int,
        'alpha': float,
        'fit_prior': bool,
        'criterion': str,
        "max_depth": int,
        'min_samples_split': int,
        'n_neighbors': int,
        'weights': str,
        'metric': str,
        "n_estimators": int,
        'min_samples_leaf': int,
        'l1_ratio': float,
    }

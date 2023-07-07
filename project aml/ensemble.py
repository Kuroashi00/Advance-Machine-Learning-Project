import numpy as np
import pandas as pd

from ml_from_scratch.tree import DecisionTreeClassifier
from ml_from_scratch.ensemble import BaggingClassifier
from ml_from_scratch.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

def accuracy_score(y_true, y_pred):
    """Accuracy classification score.
    
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
    
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
    
    Returns
    -------
    score : float
        The accuracy score
        # correct prediction / # total data
    
    Examples
    --------
    >>> from sklearn.metrics import accuracy_score
    >>> y_pred = [0, 2, 1, 3]
    >>> y_true = [0, 1, 2, 3]
    >>> accuracy_score(y_true, y_pred)
    0.5
    """
    # Compute accuracy score
    n_true = np.sum(y_true == y_pred)
    n_total = len(y_true)

    return n_true/n_total

def load_data(filename):
    return pd.read_csv(filename)

def split_input_output(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    return X, y

def split_train_test(X, y, test_size, random_state=42):
    return train_test_split(X, y,
                            test_size = test_size,
                            stratify = y,
                            random_state = random_state)


if __name__ == "__main__":
    # PREPARE THE DATA
    # -----------------
    # Load data
    filename = "data/result.csv"
    data = load_data(filename)
    
    # Filter data
    #data = data[["age", "balance", "duration", "day", "y"]]
    #data["y"] = np.where(data["y"] == "no", -1, 1)

    data = data[['ID', 'MONTHS_BALANCE', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
       'DAYS_BIRTH', 'DAYS_EMPLOYED', 'FLAG_WORK_PHONE', 'FLAG_PHONE',
       'FLAG_EMAIL', 'CNT_FAM_MEMBERS', 'CODE_GENDER_F', 'CODE_GENDER_M',
       'FLAG_OWN_CAR_N', 'FLAG_OWN_CAR_Y', 'CODE_GENDER_F', 'CODE_GENDER_M',
       'FLAG_OWN_REALTY_N', 'FLAG_OWN_REALTY_Y',
       'NAME_INCOME_TYPE_Commercial associate', 'NAME_INCOME_TYPE_Pensioner',
       'NAME_INCOME_TYPE_State servant', 'NAME_INCOME_TYPE_Student',
       'NAME_INCOME_TYPE_Working', 'NAME_EDUCATION_TYPE_Academic degree',
       'NAME_EDUCATION_TYPE_Higher education',
       'NAME_EDUCATION_TYPE_Incomplete higher',
       'NAME_EDUCATION_TYPE_Lower secondary',
       'NAME_EDUCATION_TYPE_Secondary / secondary special',
       'NAME_FAMILY_STATUS_Civil marriage', 'NAME_FAMILY_STATUS_Married',
       'NAME_FAMILY_STATUS_Separated',
       'NAME_FAMILY_STATUS_Single / not married', 'NAME_FAMILY_STATUS_Widow',
       'NAME_HOUSING_TYPE_Co-op apartment',
       'NAME_HOUSING_TYPE_House / apartment',
       'NAME_HOUSING_TYPE_Municipal apartment',
       'NAME_HOUSING_TYPE_Office apartment',
       'NAME_HOUSING_TYPE_Rented apartment', 'NAME_HOUSING_TYPE_With parents', 'bad_flag']]
    
    data["bad_flag"] = np.where(data["bad_flag"] == 0, -1, 1)

    # Split data
    X, y = split_input_output(data, "bad_flag")
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2)
    X_train, X_valid, y_train, y_valid = split_train_test(X_train, y_train, test_size=0.2)
    

    # TREE MODELING
    # -------------
    # Create Decision Tree Classifier
    clf_tree = DecisionTreeClassifier()
    clf_tree.fit(X_train, y_train)

    # Predict the tree
    y_pred_train_tree = clf_tree.predict(X_train)
    y_pred_valid_tree = clf_tree.predict(X_valid)

    acc_train_tree = accuracy_score(y_train, y_pred_train_tree)
    acc_valid_tree = accuracy_score(y_valid, y_pred_valid_tree)

    print("Decision Tree")
    print("-------------")
    print(f"acc. train  : {acc_train_tree*100:.2f}%")
    print(f"acc. valid  : {acc_valid_tree*100:.2f}%")
    print("")


    # BAGGING MODELING
    # -------------
    # Create Bagging Classifier
    clf_bagging = BaggingClassifier(estimator = DecisionTreeClassifier(),
                                    n_estimators = 1,
                                    random_state = 42)
    clf_bagging.fit(X_train, y_train)

    # Predict the tree
    y_pred_train_bagging = clf_bagging.predict(X_train)
    y_pred_valid_bagging = clf_bagging.predict(X_valid)

    acc_train_bagging = accuracy_score(y_train, y_pred_train_bagging)
    acc_valid_bagging = accuracy_score(y_valid, y_pred_valid_bagging)

    print("Bagging Tree")
    print("-------------")
    print(f"acc. train  : {acc_train_bagging*100:.2f}%")
    print(f"acc. valid  : {acc_valid_bagging*100:.2f}%")
    print("")


    # RANDOM FOREST MODELING
    # -------------
    # Create Random Forest Classifier
    clf_rf = RandomForestClassifier(n_estimators = 1,
                                    random_state = 42)
    clf_rf.fit(X_train, y_train)

    # Predict the tree
    y_pred_train_rf = clf_rf.predict(X_train)
    y_pred_valid_rf = clf_rf.predict(X_valid)

    acc_train_rf = accuracy_score(y_train, y_pred_train_rf)
    acc_valid_rf = accuracy_score(y_valid, y_pred_valid_rf)

    print("Random Forest Tree")
    print("-------------")
    print(f"acc. train  : {acc_train_rf*100:.2f}%")
    print(f"acc. valid  : {acc_valid_rf*100:.2f}%")
    print("")

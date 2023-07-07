import copy
import numpy as np



MAX_INT = np.iinfo(np.int32).max


def _generate_random_seed(random_state):
    """
    Generate the random seed

    Parameters
    -----------
    random_state : int
        The condition of random_state {`None` or `int`}

    Returns
    -------
    seed : int
        The random seed
    """
    if random_state is None:
        seed = np.random.randint(0, MAX_INT)
    else:
        seed = random_state

    return seed

def _generate_ensemble_estimators(base_estimator, n_estimators):
    """
    Generate the Ensemble estimators

    Parameters
    ----------
    estimator : object
        The base estimator of Ensemble model

    n_estimators : int
        The number of base estimator to generate

    Returns
    -------
    estimators : {array-like} of (n_estimators,)
        The list of base estimators
    """
    estimators = [copy.deepcopy(base_estimator) for i in range(n_estimators)]

    return estimators

def _generate_sample_indices(seed, n_estimators, n_population, n_samples, bootstrap=True):
    """
    Generate the Bootstrapped samples indices

    Parameters
    ----------
    seed : int
        The random seed

    n_estimators : int
        The number of bootstrapped samples were to generate

    n_population : int
        The number of maximum samples available

    n_samples : int
        The number of samples to generate in each bootstrapped samples

    bootstrap : bool, default=True
        The bootstrap condition
        If `True`, you do the sampling WITH REPLACEMENT
        Else, you do the sampling WITHOUT REPLACEMENT

    Returns
    -------
    sample_indices : {array-like} of shape (n_estimators, n_samples)
        The bootstrapped sample indices
    """
    # Get the seed
    np.random.seed(seed)

    # Get the bagging indices
    sample_indices = np.random.choice(n_population,
                                      size = (n_estimators, n_samples),
                                      replace = bootstrap)
    
    return sample_indices

def _generate_feature_indices(seed, n_estimators, n_population, n_features, bootstrap=False):
    """
    Generate the Bootstrapped samples indices

    Parameters
    ----------
    seed : int
        The random seed

    n_estimators : int
        The number of bootstrapped samples were to generate

    n_samples : int
        The number of samples to generate in each bootstrapped samples

    bootstrap : bool, default=False
        The bootstrap condition
        If `True`, you do the sampling WITH REPLACEMENT
        Else, you do the sampling WITHOUT REPLACEMENT

    Returns
    -------
    feature_indices : {array-like} of shape (n_estimators, n_feature)
        The bootstrapped sample indices
    """
    # Get the seed
    np.random.seed(seed)

    # Get the bagging indices
    feature_indices = np.empty((n_estimators, n_features), dtype="int")
    for i in range(n_estimators):
        feature_indices[i] = np.random.choice(n_population, 
                                              n_features, 
                                              replace=bootstrap)
        feature_indices[i].sort()

    return feature_indices

def _predict_ensemble(estimators, feature_indices, X):
    """
    Predict all the ensemble results for X

    Parameters
    ----------
    estimators : {array-like} of shape (n_estimators,)
        The ensemble models

    feature_indices : {array-like} of shape (n_estimators, n_features)
        The feature used on every ensemble models

    X : {array-like} of shape (n_samples, n_features)
        The training input samples.

    Returns
    -------
    y_pred : {array-like} of shape (n_samples, n_estimators)
        The predicted results of all ensemble model
    """
    # Prepare the data
    X = np.array(X).copy()
    n_samples = X.shape[0]

    # Prepare the ensemble model
    n_estimators = len(estimators)

    # Create the output
    y_preds = np.empty((n_estimators, n_samples))

    # Fill the output with the given ensemble model
    for i, estimator in enumerate(estimators):
        # Extract the estimators
        X_ = X[:, feature_indices[i]]

        # Get the predictions
        y_preds[i] = estimator.predict(X_)

    return y_preds

def _predict_aggregate(y_ensemble, aggregate_func):
    """
    Function to obtain the majority vote for each data using 

    Parameters
    ----------
    y_ensemble : {array-like} of shape (n_estimators, n_samples)
        The predicted ensembled results

    aggregate_func : object
        The function used to aggregate the output

    Returns
    y_pred : {array-like} of shape (n_samples,)
        The aggregate results
    """
    # Extract the predicted data
    n_estimators, n_samples = y_ensemble.shape

    # Find the majority vote for each samples
    y_pred = np.empty(n_samples)
    for i in range(n_samples):
        # Extract the ensemble results on
        y_samples = y_ensemble[:, i]

        # Predict the aggregate
        y_pred[i] = aggregate_func(y_samples)

    return y_pred


class BaseEnsemble:
    """
    Base class for Ensemble Model

    estimator = estimator yang kita pakai e.g. DT, KNN
    n_estimators =  banyak bootstrap sample yang ingin dibuat
    max_features = membuat model di setiap bootstrap sample
    random_state = untuk memastikan agar code kita bisa direproduce dengan baik

    """
    def __init__(
        self,
        estimator,
        n_estimators,
        max_features=None,
        random_state=None
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        """
        Build a Ensemble of estimators from the training set (X, y)

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The training input samples.

        y : {array-like} of shape (n_samples,)
            The target values.
            - Class labels in classification
            - Real number in regression

        Returns
        -------
        self : object
            Fitted estimator
        """
        # Convert data
        X = np.array(X).copy()
        y = np.array(y).copy()

        # Extract number of samples & features
        self.n_samples, self.n_features = X.shape

        # Generate the Ensemble estimators
        # menunjukan banyaknya estimator pada objek tersebut
        self.estimators_ = _generate_ensemble_estimators(base_estimator = self.estimator,
                                                         n_estimators = self.n_estimators)
        
        # Generate the random seed, generate randomize index
        seed = _generate_random_seed(random_state = self.random_state)

        # Generate the ensemble sample indices, 
        # Karena kita menyimpan index dari sample masing-masing data
        sample_indices = _generate_sample_indices(seed = seed,
                                                  n_estimators = self.n_estimators,
                                                  n_population = self.n_samples,
                                                  n_samples = self.n_samples,
                                                  bootstrap = True)
        
        
        # Generate the ensemble features indices
        # Berfungsi untuk bisa memfilter berapa banyak fitur dan fitur apa saja yang kita akan pakai secara random
        if isinstance(self.max_features, int):
            max_features = self.max_features
        elif self.max_features == "sqrt":
            max_features = int(np.sqrt(self.n_features))
        elif self.max_features == "log2":
            max_features = int(np.log2(self.n_features))
        else:
            max_features = self.n_features

        # fungsi untuk memilih feature index
        self.feature_indices = _generate_feature_indices(seed = seed,
                                                         n_estimators = self.n_estimators,
                                                         n_population = self.n_features,
                                                         n_features = max_features,
                                                         bootstrap = False)
        
        # Iterasi pembuatan model bagging/random forest
        for b in range(self.n_estimators):
            # Ekstrak data bootstrap
            # harus memiliki list data model yang ingin dimodelkan
            # untuk memilih data di fitur mana yang ingin kita bootstrap-kan
            X_bootstrap = X[:, self.feature_indices[b]]

            # Get the bootstrapped samples
            # modelkan data bootstrap
            # data di observasi mana yang akan kita bootstrap-kan
            X_bootstrap = X_bootstrap[sample_indices[b], :]
            y_bootstrap = y[sample_indices[b]]

            
            # Fit the model from the bootstrapped sample
            estimator = self.estimators_[b]
            estimator.fit(X_bootstrap, y_bootstrap)

    def predict(self, X):
        """
        Predict the class or real-value for X
        Melakukan prediksi
        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y_pred : {array-like} of shape (n_samples,)
            The predicted classes / real-valued
        """
        # Predict the ensemble
        # revisit di setiap model-modelnya, keluarkan hasil prediksinya
        y_pred_ensemble = _predict_ensemble(estimators = self.estimators_,
                                            feature_indices = self.feature_indices,
                                            X = X)
        
        # agregasi prediksinya
        y_pred = _predict_aggregate(y_pred_ensemble,
                                    self.aggregate_function)

        return y_pred

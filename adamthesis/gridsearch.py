import logging

import numpy as np
import os
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, LeaveOneOut, train_test_split, KFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer

from adamthesis.data_loading import DatasetLoader, DatasetLoaderError, slim_down_features
from adamthesis.data_labels import old_prereqs, new_prereqs


def compare_prereq_grade_to_academic_and_twq_features():
    experiments = [
        (DatasetLoader().use_batch1_individual_data_csv().set_target("491_success").use_academic_features(),
         DatasetLoader().use_batch2_group_data_csv().set_target("491_success").use_academic_features()),
        (DatasetLoader().use_batch1_individual_data_csv().set_target(
            "491_success").create_average_prereq_grade_column().use_academic_features(),
         DatasetLoader().use_batch2_group_data_csv().set_target(
             "491_success").create_average_prereq_grade_column().use_academic_features())
    ]
    for train_loader, val_loader in experiments:
        X_train, y_train = train_loader.get_features_and_target()
        X_val, y_val = val_loader.get_features_and_target()
        X_val = slim_down_features(X_train, X_val)
        grid_search_experiment = GridSearchExperiment(X_train, y_train)
        grid_search_experiment.run(print_each_model=True, print_final_results=True)
        grid_search_experiment.use_for_validation(X_val, y_val).validate_all_models()

        logger = GridSearchExperimentFileLogger(grid_search_experiment)
        logger.log_results()


def test_confusion_matrix_plotting():
    train_loader, val_loader = (
        DatasetLoader().use_batch1_individual_data_csv().set_target("492_success").use_academic_and_twq_features(),
        DatasetLoader().use_batch2_individual_data_csv().set_target("492_success").use_academic_and_twq_features())
    X_train, y_train = train_loader.get_features_and_target()
    X_val, y_val = val_loader.get_features_and_target()
    X_val = slim_down_features(X_train, X_val)
    grid_search_experiment = GridSearchExperiment(X_train, y_train, models_to_run=[RandomForestAdapter()])
    grid_search_experiment.run(print_each_model=True, print_final_results=True)
    grid_search_experiment.use_for_validation(X_val, y_val).validate_best_model()
    grid_search_experiment.generate_confusion_matrices()
    logger = GridSearchExperimentFileLogger(grid_search_experiment)
    logger.log_results()

def run_relevant_experiments_with_combined_dataset_and_folding():

    def get_loader(setup: str):
        if setup == "individual-491":
            return DatasetLoader().use_combined_individual_data_csv().set_target("491_success")
        elif setup == "group-491":
            return DatasetLoader().use_combined_group_data_csv().set_target("491_success")
        elif setup == "individual-492":
            return DatasetLoader().use_combined_individual_data_csv().set_target("492_success")
        elif setup == "group-492":
            return DatasetLoader().use_combined_group_data_csv().set_target("492_success")

    experiments = [
        ("twq", lambda loader: loader.use_teamwork_features()),
        ("academic",lambda loader: loader.use_academic_features()),
        ("both",lambda loader: loader.use_academic_and_twq_features()),
        ("old-prereqs",lambda loader: loader.use_features(old_prereqs)),
        ("new-prereqs",lambda loader: loader.use_features(new_prereqs)),
    ]

    for target in ["individual-491", "group-491", "individual-492", "group-492"]:
        for name, experiment in experiments:
            features, target_series =  experiment(get_loader(target)).get_features_and_target()
            grid_search_experiment = GridSearchExperiment(features, target_series)
            grid_search_experiment.run_with_folding(n_folds_outer=10)
            grid_search_experiment.generate_confusion_matrices()
            grid_search_experiment.perform_statistical_tests_on_folds()

            # pickle results
            import pickle
            with open(f'logs/pickles/{target}_{name}_10folding_with_stats.pkl', 'wb') as f:
                pickle.dump(grid_search_experiment, f)

            logger = GridSearchExperimentFileLogger(grid_search_experiment, file_path='logs/10folding_with_stats.csv')
            logger.log_results()


def run_relevant_experiments_with_combined_dataset():

    def get_loader(setup: str):
        if setup == "individual-491":
            return DatasetLoader().use_combined_individual_data_csv().set_target("491_success")
        elif setup == "group-491":
            return DatasetLoader().use_combined_group_data_csv().set_target("491_success")
        elif setup == "individual-492":
            return DatasetLoader().use_combined_individual_data_csv().set_target("492_success")
        elif setup == "group-492":
            return DatasetLoader().use_combined_group_data_csv().set_target("492_success")

    from data_labels import old_prereqs

    experiments = [
        lambda loader: loader.use_teamwork_features(),
        lambda loader: loader.use_academic_features(),
        lambda loader: loader.use_academic_and_twq_features(),
        lambda loader: loader.use_features(old_prereqs)
    ]

    for target in ["individual-491", "group-491", "individual-492", "group-492"]:
        for experiment in experiments:
            features, target_series =  experiment(get_loader(target)).get_features_and_target()
            X_train, X_val, y_train, y_val= train_test_split(features, target_series, test_size=0.3, random_state=42, stratify=target_series)
            grid_search_experiment = GridSearchExperiment(X_train, y_train)
            grid_search_experiment.run(print_each_model=True, print_final_results=True)
            grid_search_experiment.use_for_validation(X_val, y_val).validate_all_models()
            grid_search_experiment.generate_confusion_matrices()

            logger = GridSearchExperimentFileLogger(grid_search_experiment)
            logger.log_results()



def run_all_experiments_with_batch2_validation():
    """ Step 1 of integrating new batch into our experiments: use batch2 only as validation"""

    def setup_targets():
        return (DatasetLoader().use_batch1_individual_data_csv().set_target("492_success"),
                DatasetLoader().use_batch1_individual_data_csv().set_target("492_success"))

    experiments = [
        # lambda loader: loader.use_all_features(),
        # lambda loader: loader.use_all_features().temp_remove_relevancy_columns(),
        # lambda loader: loader.use_all_features().temp_remove_relevancy_columns().create_not_taken_columns(),
        # lambda loader: loader.use_all_features().temp_remove_semester_columns().temp_remove_relevancy_columns(),
        # lambda
        #     loader: loader.use_all_features().temp_remove_semester_columns().temp_remove_relevancy_columns().create_not_taken_columns(),
        lambda loader: loader.use_academic_features(),
        lambda loader: loader.use_academic_features().create_not_taken_columns(),
        lambda loader: loader.use_teamwork_features(),
        lambda loader: loader.use_academic_and_twq_features(),
        lambda loader: loader.use_academic_and_twq_features().create_not_taken_columns(),
        lambda loader: loader.use_academic_and_twq_features().create_relevance_ratio_column(value_if_none_relevant=0),
        lambda
            loader: loader.use_academic_and_twq_features().create_not_taken_columns().create_average_relevant_grade_column(
            value_if_none_relevant=0).create_average_prereq_grade_column(),
        lambda
            loader: loader.use_academic_and_twq_features().create_average_relevant_grade_column(
            value_if_none_relevant=0).create_average_prereq_grade_column(),
        lambda
            loader: loader.use_academic_and_twq_features().create_average_relevant_grade_column(
            value_if_none_relevant=0).create_average_prereq_grade_column().create_relevance_ratio_column(
            value_if_none_relevant=0.0),
        lambda loader: loader.use_academic_and_twq_features().create_weighted_grade_columns()
    ]

    for experiment in experiments:
        try:
            train_loader, val_loader = setup_targets()
            X_train, y_train = experiment(train_loader).get_features_and_target()
            X_val, y_val = experiment(val_loader).get_features_and_target()

            # Ensure validation set has the same features as training set
            X_val = slim_down_features(X_train, X_val)

            grid_search_experiment = GridSearchExperiment(X_train, y_train)
            grid_search_experiment.run(print_each_model=True, print_final_results=True)
            grid_search_experiment.use_for_validation(X_val, y_val).validate_all_models()

            # Initialize and use the logger
            logger = GridSearchExperimentFileLogger(grid_search_experiment)
            logger.log_results()
        except DatasetLoaderError as e:
            logging.warning(
                f"Skipping experiment due to error: {str(e)}")  # one group did not have any relevant courses
            continue


def run_all_experiments():
    """A Meta-Experiment to run all gridsearch experiments post-refactoring for logging"""

    def individual_491_experiment():
        return DatasetLoader().use_batch1_group_data_csv().set_target("491_success")

    datasets = [
        individual_491_experiment().use_all_features(),
        individual_491_experiment().use_all_features().temp_remove_relevancy_columns(),
        individual_491_experiment().use_all_features().temp_remove_relevancy_columns().create_not_taken_columns(),
        individual_491_experiment().use_all_features().temp_remove_semester_columns().temp_remove_relevancy_columns(),
        individual_491_experiment().use_all_features().temp_remove_semester_columns().temp_remove_relevancy_columns().create_not_taken_columns(),
        individual_491_experiment().use_academic_features(),
        individual_491_experiment().use_academic_features().create_not_taken_columns(),
        individual_491_experiment().use_teamwork_features(),
        individual_491_experiment().use_academic_and_twq_features(),
        individual_491_experiment().use_academic_and_twq_features().create_not_taken_columns(),
        individual_491_experiment().use_academic_and_twq_features().create_relevance_ratio_column(),
        individual_491_experiment().use_academic_and_twq_features().create_not_taken_columns().create_average_relevant_grade_column().create_average_prereq_grade_column(),
        individual_491_experiment().use_academic_and_twq_features().create_average_relevant_grade_column().create_average_prereq_grade_column(),
        individual_491_experiment().use_academic_and_twq_features().create_average_relevant_grade_column().create_average_prereq_grade_column().create_relevance_ratio_column(),
        individual_491_experiment().use_academic_and_twq_features().create_weighted_grade_columns()
    ]

    for dataset in datasets:
        X, y = (dataset
                .get_features_and_target())
        experiment = GridSearchExperiment(X, y)
        experiment.run(print_each_model=True, print_final_results=True)

        # Initialize and use the logger
        logger = GridSearchExperimentFileLogger(experiment)
        logger.log_results()


def main():
    run_relevant_experiments_with_combined_dataset_and_folding()


class GridSearchModelAdapter(ABC):
    """Base adapter class for models to be used in grid search"""

    def __init__(self, name, model, params):
        self.name = name
        self.model = model
        self.params = params

    @abstractmethod
    def get_feature_importance(self, best_estimator, feature_names):
        """Extract feature importance from the best estimator"""
        pass

    def to_tuple(self):
        """Return a tuple of (name, model, params) for backward compatibility"""
        return self.name, self.model, self.params


class BaselineAdapter(GridSearchModelAdapter):
    """Adapter for the baseline model using DummyClassifier"""

    def __init__(self):
        name = 'baseline_classifier'
        model = DummyClassifier(strategy='uniform', random_state=42)
        params = {}
        super().__init__(name, model, params)

    def get_feature_importance(self, best_estimator, feature_names):
        # No feature importance for dummy classifier
        return {}


class RandomForestAdapter(GridSearchModelAdapter):
    """Adapter for RandomForestClassifier"""

    def __init__(self):
        name = 'random_forest'
        model = RandomForestClassifier(random_state=42, class_weight='balanced')
        params = {
            'n_estimators': [10, 50, 100],  # also tried other values such as 45,49,50,51, but 50 performs best
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10],  # 2 is the default, performs best
            # increase min_samples_leaf to reduce overfitting
            'min_samples_leaf': [5, 10, 20],
            'bootstrap': [True, False]  # False = use entire dataset for each tree
        }
        super().__init__(name, model, params)

    def get_feature_importance(self, best_estimator, feature_names):
        # For Random Forest, use feature_importances_
        importance_dict = {}
        importances = best_estimator.named_steps[self.name].feature_importances_
        for feature, importance in zip(feature_names, importances):
            importance_dict[feature] = importance
        # Sort by importance, most important first
        return {k: v for k, v in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)}


class SVCAdapter(GridSearchModelAdapter):
    """Adapter for SVC"""

    def __init__(self):
        name = 'svc'
        model = SVC(random_state=42, class_weight='balanced')
        params = {
            'C': [0.001, 0.01, 0.1],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid', ],
            'gamma': ['scale', 'auto']
        }
        super().__init__(name, model, params)

    def get_feature_importance(self, best_estimator, feature_names):
        importance_dict = {}
        if best_estimator.named_steps[self.name].kernel == 'linear':
            # For SVC with linear kernel
            importances = best_estimator.named_steps[self.name].coef_[0]
            for feature, importance in zip(feature_names, importances):
                importance_dict[feature] = abs(importance)
            # Sort by importance, most important first
            return {k: v for k, v in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)}
        else:
            print(f"No feature importance available for SVC with {best_estimator.named_steps[self.name].kernel} kernel")
            return {}


class LogisticRegressionAdapter(GridSearchModelAdapter):
    """Adapter for LogisticRegression"""

    def __init__(self):
        name = 'logistic_regression'
        model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        params = {
            'C': [0.001, 0.01],
            'penalty': ['l1', 'l2', ],  # I also tried saga and elasticnet, but they did not converge
            'solver': ['liblinear']  # Required for l1 penalty
        }
        super().__init__(name, model, params)

    def get_feature_importance(self, best_estimator, feature_names):
        # For Logistic Regression, use coefficients
        importance_dict = {}
        importances = best_estimator.named_steps[self.name].coef_[0]
        for feature, importance in zip(feature_names, importances):
            importance_dict[feature] = abs(importance)  # Taking absolute for importance
        # Sort by importance, most important first
        return {k: v for k, v in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)}


class NaiveBayesAdapter(GridSearchModelAdapter):
    """Adapter for GaussianNB"""

    def __init__(self):
        name = 'naive_bayes'
        model = GaussianNB()
        params = {
            'var_smoothing': [1e-10, 1e-09, 1e-08, 1e-07]
            # 1e-09 (default) performed best so I added 1e-10 but it did not improve
        }
        super().__init__(name, model, params)

    def get_feature_importance(self, best_estimator, feature_names):
        # For Naive Bayes, calculate feature importance based on variance
        importance_dict = {}
        var = best_estimator.named_steps[self.name].var_

        # Higher variance difference between classes = more important feature
        if var.shape[0] > 1:  # More than one class
            var_diff = np.abs(var[0] - var[1])
            for feature, diff in zip(feature_names, var_diff):
                importance_dict[feature] = diff
        else:
            # Just use the variance
            for feature, variance in zip(feature_names, var[0]):
                importance_dict[feature] = variance
        # Sort by importance, most important first
        return {k: v for k, v in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)}


class KNNAdapter(GridSearchModelAdapter):
    """Adapter for KNeighborsClassifier"""

    def __init__(self):
        name = "KNN"
        model = KNeighborsClassifier()
        params = {
            'n_neighbors': [1, 3, 5, 7, 9, 11, ],  # Odd numbers to avoid ties
            'weights': ['uniform', 'distance'],  # Uniform or distance weighting
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']  # Different algorithms for KNN
        }
        super().__init__(name, model, params)

    def get_feature_importance(self, best_estimator, feature_names):
        # No direct feature importance for KNN
        print("No feature importance available for KNN")
        return {}


class DecisionTreeAdapter(GridSearchModelAdapter):
    """Adapter for DecisionTreeClassifier"""

    def __init__(self):
        name = "DecisionTree"
        # add class_weight balancing
        model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
        params = {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'min_samples_split': [2, 4, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'splitter': ['best', 'random'],
        }
        super().__init__(name, model, params)

    def get_feature_importance(self, best_estimator, feature_names):
        # For Decision Tree, use feature_importances_
        importance_dict = {}
        importances = best_estimator.named_steps[self.name].feature_importances_
        for feature, importance in zip(feature_names, importances):
            importance_dict[feature] = importance
        # Sort by importance, most important first
        return {k: v for k, v in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)}


def get_all_models_to_run():
    """Return a list of all model adapters to run in grid search"""
    return [
        BaselineAdapter(),
        RandomForestAdapter(),
        SVCAdapter(),
        LogisticRegressionAdapter(),
        NaiveBayesAdapter(),
        KNNAdapter(),
        DecisionTreeAdapter()
    ]


class GridSearchExperiment:
    """ A grid search experiment for multiple models.

    params:
        X (pd.DataFrame): feature dataframe
        y (pd.Series): target series
        x_val (pd.DataFrame): validation feature dataframe
        y_val (pd.Series): validation target series
        models_to_run (list): list of GridSearchModelAdapter instances
    """

    def __init__(self, X, y, X_val=None, y_val=None, models_to_run=None, cv=LeaveOneOut()):
        if models_to_run is None:
            models_to_run = get_all_models_to_run()
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val
        self.models_to_run = models_to_run
        self.cv = cv
        self.results = []
        self.validation_results = []
        self.confusion_matrices = {}
        self.statistical_results = {}

    def run(self, print_each_model: bool = True, print_final_results: bool = True):
        for model_adapter in self.models_to_run:
            result = self._conduct_grid_search(model_adapter, print_model_result=print_each_model)
            if result:
                self.results.append(result)

        if print_final_results:
            self._print_all_results()
            self._print_feature_importance_for_spreadsheet()
        return self.results

    def run_with_folding(self, n_folds_outer, n_folds_inner= 3, print_each_model: bool = True, print_final_results: bool = True):
        """Rewriting the run and conduct grid search methods with K-Fold cross-validation"""
        outer_cv = StratifiedKFold(n_splits=n_folds_outer, shuffle=True, random_state=42)

        for model_adapter in self.models_to_run:
            fold_train_scores = []
            fold_val_scores = []
            fold_val_f1 = []
            fold_val_precision = []
            fold_val_recall = []
            fold_train_f1 = []
            fold_train_precision = []
            fold_train_recall = []
            # Accumulate out-of-fold predictions across all outer folds.
            # Each sample appears as a test sample exactly once, so concatenating
            # these gives a full out-of-sample confusion matrix for the whole dataset.
            all_y_true = []
            all_y_pred = []
            for train_index, test_index in outer_cv.split(self.X, self.y):
                X_train_fold, X_test_fold = self.X.iloc[train_index], self.X.iloc[test_index]
                y_train_fold, y_test_fold = self.y.iloc[train_index], self.y.iloc[test_index]

                # Inner grid search (for hyperparameter tuning)
                inner_cv = KFold(n_splits=n_folds_inner, shuffle=True, random_state=42)
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    (model_adapter.name, model_adapter.model)
                ])
                param_grid = {f'{model_adapter.name}__{param}': values for param, values in model_adapter.params.items()}

                grid_search = GridSearchCV(
                    pipeline,
                    param_grid,
                    cv=inner_cv,
                    scoring = {
                        "accuracy": "accuracy",
                        "f1": "f1",
                        "fail_recall": make_scorer(recall_score, pos_label=0)
                    },
                    refit='accuracy',
                    n_jobs=-1,
                    verbose=1,
                )
                grid_search.fit(X_train_fold, y_train_fold)

                best_estimator = grid_search.best_estimator_
                train_score = grid_search.best_score_
                # Predict on the held-out outer fold and collect for confusion matrix
                y_pred_fold = best_estimator.predict(X_test_fold)
                val_score = np.mean(y_pred_fold == y_test_fold)
                fold_train_scores.append(train_score)
                fold_val_scores.append(val_score)

                # Additional metrics on outer validation fold
                fold_val_f1.append(f1_score(y_test_fold, y_pred_fold, zero_division=0))
                fold_val_precision.append(precision_score(y_test_fold, y_pred_fold, zero_division=0))
                fold_val_recall.append(recall_score(y_test_fold, y_pred_fold, zero_division=0))

                # Training metrics (resubstitution on outer training fold)
                y_train_pred = best_estimator.predict(X_train_fold)
                fold_train_f1.append(f1_score(y_train_fold, y_train_pred, zero_division=0))
                fold_train_precision.append(precision_score(y_train_fold, y_train_pred, zero_division=0))
                fold_train_recall.append(recall_score(y_train_fold, y_train_pred, zero_division=0))
                all_y_true.extend(y_test_fold.tolist())
                all_y_pred.extend(y_pred_fold.tolist())

            avg_train_score = np.mean(fold_train_scores)
            std_score = np.std(fold_train_scores)

            # Build the aggregated out-of-fold confusion matrix.
            all_y_true_series = pd.Series(all_y_true, name=self.y.name)
            all_y_pred_series = pd.Series(all_y_pred, name="Predicted")
            fold_cm = pd.crosstab(all_y_true_series, all_y_pred_series)
            self.confusion_matrices[model_adapter.name] = {
                'validation': fold_cm  # aggregated out-of-fold (OOF) confusion matrix
            }

            result = {
                'model': model_adapter.name,
                'best_params': grid_search.best_params_,
                'best_score': avg_train_score,
                'std_score': std_score,
                'feature_importance': model_adapter.get_feature_importance(best_estimator, self.X.columns),
                'best_estimator': best_estimator,
                'fold_train_scores': fold_train_scores,
                'fold_val_scores': fold_val_scores,
                'fold_val_f1': fold_val_f1,
                'fold_val_precision': fold_val_precision,
                'fold_val_recall': fold_val_recall,
                'fold_train_f1': fold_train_f1,
                'fold_train_precision': fold_train_precision,
                'fold_train_recall': fold_train_recall,
            }

            if print_each_model:
                self._print_model_result(result)

            self.results.append(result)

            self.validation_results.append({
                'model': model_adapter.name,
                'validation_score': np.mean(fold_val_scores),
                'validation_f1': np.mean(fold_val_f1),
                'validation_precision': np.mean(fold_val_precision),
                'validation_recall': np.mean(fold_val_recall),
            })
        if print_final_results:
            self._print_all_results()
            self._print_feature_importance_for_spreadsheet()
        return self.results


    def perform_statistical_tests_on_folds(self):
        """ Perform statistical tests (e.g., paired t-test) on fold results of models against baseline"""
        if not self.results:
            raise ValueError("No results available. Run the experiment first.")
        elif not all('fold_val_scores' in result for result in self.results):
            raise ValueError("Fold validation scores not available in results. Run the experiment with folding.")

        baseline_result = next((r for r in self.results if r['model'] == 'baseline_classifier'), None)
        if not baseline_result:
            raise ValueError("Baseline model results not found.")
        baseline_scores = baseline_result['fold_val_scores']

        self.statistical_results = {}

        for result in self.results:
            model_name = result['model']
            model_scores = result['fold_val_scores']
            if model_name == 'baseline_classifier':
                continue
            t_stat, p_value = stats.ttest_rel(model_scores, baseline_scores, alternative='greater')
            sign_stat = sum(1 for m, b in zip(model_scores, baseline_scores) if m > b)
            sign_pval = stats.binomtest(sign_stat, n=len(model_scores), p=0.5, alternative='greater')  # NOTE to adam: monitor if alternative param needed
            # store results for later logging/inspection
            self.statistical_results[model_name] = {
                't_statistic': t_stat,
                't_pvalue': p_value,
                'sign_wins': sign_stat,
                'sign_pvalue': sign_pval,
                'fold_val_scores': model_scores
            }

            # Statistical tests for additional metrics
            for metric_key in ['fold_val_f1', 'fold_val_precision', 'fold_val_recall']:
                model_metric = result.get(metric_key, [])
                baseline_metric = baseline_result.get(metric_key, [])
                if model_metric and baseline_metric:
                    t_stat_m, p_value_m = stats.ttest_rel(model_metric, baseline_metric, alternative='greater')
                    self.statistical_results[model_name][f'{metric_key}_t_statistic'] = t_stat_m
                    self.statistical_results[model_name][f'{metric_key}_t_pvalue'] = p_value_m

    def generate_confusion_matrices(self):
        """ Generate and store the validation confusion matrix for the best validation model.

        For ``run_with_folding`` flows the per-model out-of-fold matrices are written
        directly inside ``run_with_folding`` as ``confusion_matrices[model_name] = {'validation': cm}``.
        This method only adds a flat ``'validation'`` matrix when a separate ``X_val/y_val``
        validation set is supplied (i.e. the ``run()`` flow).
        """
        if not self.results:
            raise ValueError("No results available. Run the experiment first.")

        if self.X_val is not None and self.y_val is not None and self.validation_results:
            best_validation_model = max(self.validation_results, key=lambda x: x['validation_score'])
            best_validation_model_name = best_validation_model['model']
            best_validation_model_adapter = next(
                (m for m in self.models_to_run if m.name == best_validation_model_name), None)
            if not best_validation_model_adapter:
                raise ValueError(f"Model adapter for {best_validation_model_name} not found.")
            best_validation_pipeline = next(
                (r.get('best_estimator') for r in self.results if r['model'] == best_validation_model_name), None)
            if not best_validation_pipeline:
                raise ValueError(f"No best estimator found for model {best_validation_model_name}.")
            y_val_pred = pd.Series(best_validation_pipeline.predict(self.X_val), index=self.y_val.index,
                                   name="Predicted")
            df_confusion_val = pd.crosstab(self.y_val, y_val_pred)
            self.confusion_matrices['validation'] = df_confusion_val

    def use_for_validation(self, X_val, y_val):
        """Set validation data for later use"""
        self.X_val = X_val
        self.y_val = y_val
        return self

    def validate_best_model(self):
        """Validate the best model on a new dataset"""
        if self.X_val is None or self.y_val is None:
            raise ValueError("Validation data not provided.")
        if not self.results:
            raise ValueError("No results available. Run the experiment first.")

        best_model = max(self.results, key=lambda x: x['best_score'])
        name = best_model['model']
        model_adapter = next((m for m in self.models_to_run if m.name == name), None)
        if not model_adapter:
            raise ValueError(f"Model adapter for {name} not found.")

        best_pipeline = best_model.get('best_estimator')
        if not best_pipeline:
            raise ValueError(f"No best estimator found for model {name}.")

        score = best_pipeline.score(self.X_val, self.y_val)
        print(f"Validation score of the best model ({name}) on new data: {score:.4f}")
        self.validation_results.append({
            'model': name,
            'validation_score': score
        })
        return score

    def validate_all_models(self):
        """Validate all models on a new dataset"""
        if self.X_val is None or self.y_val is None:
            raise ValueError("Validation data not provided.")
        if not self.results:
            raise ValueError("No results available. Run the experiment first.")

        for result in self.results:
            name = result['model']
            model_adapter = next((m for m in self.models_to_run if m.name == name), None)
            if not model_adapter:
                print(f"Model adapter for {name} not found. Skipping.")
                continue

            best_pipeline = result.get('best_estimator')
            if not best_pipeline:
                print(f"No best estimator found for model {name}. Skipping.")
                continue

            score = best_pipeline.score(self.X_val, self.y_val)
            self.validation_results.append({
                'model': name,
                'validation_score': score
            })
        return self.validation_results

    def _conduct_grid_search(self, model_adapter, print_model_result: bool = True):
        """Conduct grid search for a given model and parameters"""
        name, model, params = model_adapter.to_tuple()

        # Create a pipeline with StandardScaler and the model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            (name, model)
        ])

        # Adjust parameter names for pipeline
        param_grid = {f'{name}__{param}': values for param, values in params.items()}

        try:
            # Grid search with K-Fold cross-validation
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=self.cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1,
            )

            grid_search.fit(self.X, self.y)

            best_estimator = grid_search.best_estimator_
            feature_importance = model_adapter.get_feature_importance(best_estimator, self.X.columns)

            # Store results
            result = {
                'model': name,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'std_score': grid_search.cv_results_['std_test_score'][grid_search.best_index_],
                'feature_importance': feature_importance,
                'best_estimator': grid_search.best_estimator_,
            }

            if print_model_result:
                self._print_model_result(result)
            return result

        except Exception as e:
            print(f"Error with {name}: {str(e)}")
            return None

    def _print_all_results(self):
        if self.results:
            # Find best overall model
            best_model = max(self.results, key=lambda x: x['best_score'])
            print(f"\nBest model: {best_model['model'].upper()}")
            print(f"Score: {best_model['best_score']:.4f} +/- {best_model['std_score']:.4f}")
            print(f"Parameters: {best_model['best_params']}")

            # Print feature importance for best model
            if 'feature_importance' in best_model:
                print("\nTop 10 Important Features for Best Model:")
                for i, (feature, importance) in enumerate(list(best_model['feature_importance'].items())[:10]):
                    print(f"   {i + 1}. {feature}: {importance:.4f}")
        else:
            print("\nNo models completed successfully.")

    def _print_feature_importance_for_spreadsheet(self):
        # pretty print best model feature importance for our results table
        if self.results:
            print("\nComplete feature Importance for Best Model:")
            best_model = max(self.results, key=lambda x: x['best_score'])
            if 'feature_importance' in best_model:
                # Print all feature importances
                print("[", end='')
                for i, (feature, importance) in enumerate(best_model['feature_importance'].items()):
                    print(f"'{feature}': {importance:.4f}", end=',\n')
                print("]")
            else:
                print("N/A")

    @staticmethod
    def _print_model_result(result):
        print(f"\nModel: {result['model'].upper()}")
        print(f"Best Parameters: {result['best_params']}")
        print(f"Best CV Score: {result['best_score']:.4f} +/- {result['std_score']:.4f}")
        # Print top 10 most important features
        if 'feature_importance' in result and result['feature_importance']:
            print("\nTop 10 Important Features:")
            for i, (feature, importance) in enumerate(list(result['feature_importance'].items())[:10]):
                print(f"   {i + 1}. {feature}: {importance:.4f}")
        print("-" * 40)


class GridSearchExperimentFileLogger:
    """ A simple file logger for GridSearchExperiment results. """

    def __init__(self, experiment: GridSearchExperiment, file_path="logs/grid_search_experiment.csv" ):
        self.file_path = file_path
        self.experiment = experiment
        self._header = ("Features,Target,"
                        "Best Model, Best Training accuracy, "
                        "Best Validation Model, Best Validation Accuracy, "
                        "Feature Importance,Results, Validation Results, "
                        "Validation Confusion,"
                        "Best Params,Statistical Results,"
                        "Results F1, Results Precision, Results Recall,"
                        "Validation Results F1, Validation Results Precision, Validation Results Recall\n")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        # Initialize headers if needed
        if not self._headers_exist():
            self.write_header()

    def _headers_exist(self):
        """Check if the file exists and has headers"""
        try:
            with open(self.file_path, 'r') as f:
                first_line = f.readline().strip()
                return first_line.startswith(self._header)
        except FileNotFoundError:
            return False

    def write_header(self):
        """Write headers to the file if they don't exist"""
        # If file doesn't exist or is empty, write headers
        write_mode = 'a' if os.path.exists(self.file_path) else 'w'
        with open(self.file_path, write_mode) as f:
            if write_mode == 'w' or os.path.getsize(self.file_path) == 0:
                f.write(self._header)

    def _format_results_line(self):
        """Format the experiment results as a single line for the log file"""
        if not self.experiment.results:
            return None

        # Determine best training model(s) (handle ties)
        tol = 1e-12
        max_train_score = max(r['best_score'] for r in self.experiment.results)
        tied_train_models = [r['model'] for r in self.experiment.results if
                             abs(r['best_score'] - max_train_score) <= tol]
        best_training_model_name = " == ".join(tied_train_models)
        best_training_accuracy = f"{max_train_score:.4f}"

        # For feature importance, pick the first tied model (deterministic)
        first_best_result = next(r for r in self.experiment.results if abs(r['best_score'] - max_train_score) <= tol)

        # Format feature names
        feature_str = ",\n".join(self.experiment.X.columns)

        # Format target
        target_str = str(self.experiment.y.name)

        if 'feature_importance' in first_best_result and first_best_result['feature_importance']:
            importance_str = ",\n".join(
                [f"{feat}:{imp:.4f}" for feat, imp in first_best_result['feature_importance'].items()])
        else:
            importance_str = "N/A"

        # Find best validation model(s) and accuracy (handle ties)
        if self.experiment.validation_results:
            max_val_score = max(r['validation_score'] for r in self.experiment.validation_results)
            tied_val_models = [r['model'] for r in self.experiment.validation_results if
                               abs(r['validation_score'] - max_val_score) <= tol]
            best_val_model_name = " == ".join(tied_val_models)
            best_val_accuracy = f"{max_val_score:.4f}"
        else:
            best_val_model_name = "N/A"
            best_val_accuracy = "N/A"

        # Format all results summary
        results_summary = ",\n".join([f"{r['model']}:{r['best_score']:.4f}" for r in self.experiment.results])

        val_results_summary = ""
        # Format validation results summary
        if len(self.experiment.validation_results) > 1:
            val_results_summary = ",\n".join(
                [f"{r['model']}:{r['validation_score']:.4f}" for r in self.experiment.validation_results])
        else:
            val_results_summary = "N/A"

        if self.experiment.confusion_matrices:
            validation_cm = self._get_validation_cm_string()
        else:
            validation_cm = "N/A"

        # write best params for all models
        best_params_summary = ",\n".join(
            [f"{r['model']}:{r['best_params']}" for r in self.experiment.results])

        # Format statistical results if available
        stat_results_str = ""
        if self.experiment.statistical_results:
            for model_name, stats_dict in self.experiment.statistical_results.items():
                stat_results_str += (f"{model_name}: " +
                                      f"ttest_pvalue={stats_dict['t_pvalue']:.4f}, "+
                                      f"sign_wins={stats_dict['sign_wins']}, "+
                                      f"sign_pvalue={stats_dict['sign_pvalue'].pvalue:.4f}; "
                                      f"fold_val_scores=[{', '.join(f'{score:.4f}' for score in stats_dict['fold_val_scores'])}]")
                # Append additional metric stats if present
                for metric_key in ['fold_val_f1', 'fold_val_precision', 'fold_val_recall']:
                    if f'{metric_key}_t_pvalue' in stats_dict:
                        stat_results_str += f"; {metric_key}_ttest_pvalue={stats_dict[f'{metric_key}_t_pvalue']:.4f}"
                stat_results_str += "\n"

        else:
            stat_results_str = "N/A"

        # Format per-model F1, precision, recall summaries (training)
        results_f1 = ",\n".join(
            [f"{r['model']}:{np.mean(r.get('fold_train_f1', [])):.4f}" for r in self.experiment.results])
        results_precision = ",\n".join(
            [f"{r['model']}:{np.mean(r.get('fold_train_precision', [])):.4f}" for r in self.experiment.results])
        results_recall = ",\n".join(
            [f"{r['model']}:{np.mean(r.get('fold_train_recall', [])):.4f}" for r in self.experiment.results])

        # Format per-model F1, precision, recall summaries (validation)
        if len(self.experiment.validation_results) > 1:
            val_results_f1 = ",\n".join(
                [f"{r['model']}:{r.get('validation_f1', 0):.4f}" for r in self.experiment.validation_results])
            val_results_precision = ",\n".join(
                [f"{r['model']}:{r.get('validation_precision', 0):.4f}" for r in self.experiment.validation_results])
            val_results_recall = ",\n".join(
                [f"{r['model']}:{r.get('validation_recall', 0):.4f}" for r in self.experiment.validation_results])
        else:
            val_results_f1 = "N/A"
            val_results_precision = "N/A"
            val_results_recall = "N/A"

        # Combine all into one line
        return (f"\"{feature_str}\",{target_str},{best_training_model_name},{best_training_accuracy}, "
                f"{best_val_model_name}, {best_val_accuracy},\"{importance_str}\","
                f"\"{results_summary}\",\"{val_results_summary}\","
                f"\"{validation_cm}\","
                f"\"{best_params_summary}\",\"{stat_results_str}\","
                f"\"{results_f1}\",\"{results_precision}\",\"{results_recall}\","
                f"\"{val_results_f1}\",\"{val_results_precision}\",\"{val_results_recall}\"")

    def _line_exists(self, line):
        """Check if an exact line already exists in the file"""
        try:
            with open(self.file_path, 'r') as f:
                for existing_line in f:
                    if existing_line.strip() == line.strip():
                        return True
            return False
        except FileNotFoundError:
            return False

    def log_results(self):
        """Log the experiment results if they're not already in the file"""
        if not self.experiment.results:
            return

        # Format the results as a line
        result_line = self._format_results_line()
        if not result_line:
            return

        # Check if this exact line already exists
        if self._line_exists(result_line):
            print("This experiment result already exists in the log file.")
            return

        # Append the new result
        with open(self.file_path, 'a') as f:
            f.write(f"{result_line}\n")

    def _get_validation_cm_string(self):
        """Get the validation confusion matrix as a string.

        Handles two storage layouts:
        - run() / generate_confusion_matrices(): flat key 'validation'
        - run_with_folding(): nested keys {model_name: {'validation': cm}}
        """
        if 'validation' in self.experiment.confusion_matrices:
            return self.experiment.confusion_matrices['validation'].to_csv(index=True)
        parts = []
        for model_name, cm_dict in self.experiment.confusion_matrices.items():
            if isinstance(cm_dict, dict) and 'validation' in cm_dict:
                parts.append(f"{model_name}:\n{cm_dict['validation'].to_csv(index=True)}")
        return "\n".join(parts) if parts else "N/A"


if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
import logging

from adamthesis import data_labels
from adamthesis.data_labels import twq_sections

# After batch-specific preprocessing, each batch's 491 semester always encodes to 1
_491_SEMESTER_VALUE = 1


def get_individual_dataframe(data: pd.DataFrame = pd.read_csv("data/individual_preprocessed_data.csv"),
                             not_taken_as_nan=False):
    """
    Load the individual data from data DataFrame.

    Parameters:
        data (df.Dataframe): Dataframe of raw/preprocessed data, defaults to "preprocessed_data.csv".
        not_taken_as_nan (bool): If True, replaces not-taken courses with NaN.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """

    # Replace not-taken courses with NaN
    if not_taken_as_nan:
        numeric_columns = data.columns[data.columns.str.isdigit()]
        data[numeric_columns] = data[numeric_columns].astype(float).replace(-1, np.nan)

    return data


def get_group_dataframe(data: pd.DataFrame = pd.read_csv("data/group_preprocessed_data.csv"),
                        additional_agg_funcs=None):
    """
    Get the group-level DataFrame from the preprocessed data.

    Parameters:
        data (pd.DataFrame): DataFrame of raw/preprocessed data, defaults to "preprocessed_data.csv".
        Additional_agg_funcs (dict): Additional aggregation functions to apply to the DataFrame.
            In the format {column_name: aggregation_function}.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """

    if additional_agg_funcs is not None:
        raise NotImplementedError("additional_agg_funcs has been refactored.")
    return data


def print_dataframe_as_markdown_table(df: pd.DataFrame, dp: int = 2):
    """
    Print a DataFrame as a Markdown table.

    Parameters:
        df (pd.DataFrame): DataFrame to print.
        dp (int): Number of decimal places for float formatting.
    """
    print(df.to_markdown(index=False, floatfmt=f".{dp}f"))


def get_correlation_as_dataframe(x, y, label, method):
    """
    Computes the correlation and p-value between two variables using a specified method
    and returns the results as a pandas DataFrame.
    Args:
        x (array-like): The first variable for correlation computation.
        y (array-like): The second variable for correlation computation.
        label (str): The label to use as the column name for the correlation value in the resulting DataFrame.
        method (callable): A function that computes the correlation and p-value.
                           It should accept two arguments (x, y) and return a tuple (correlation, p-value).
    Returns:
        pandas.DataFrame: A DataFrame containing the correlation value and p-value.
                          The DataFrame has two columns: one with the specified label for the correlation
                          and another named 'pval' for the p-value.
    """

    result = []
    corr, pval = method(x, y)
    result.append({
        label: corr,
        'pval': pval
    })
    return pd.DataFrame(result)


class DatasetLoaderError(Exception):
    """
    Custom exception for DatasetLoader errors.
    """

    def __init__(self, message="Data has not been loaded. Please load the data before attempting to process it."):
        self.message = message
        super().__init__(self.message)


class DatasetLoader:
    def __init__(self):
        self.df = None
        self.features = []
        self.target = None

    def use_dataframe(self, df: pd.DataFrame):
        """
        Use a provided DataFrame as the dataset.

        Parameters:
            df (pd.DataFrame): The DataFrame to use as the dataset.
        """
        self.df = df
        return self

    def use_batch2_individual_data_csv(self):
        """
        Load individual-level data for the dataset.
        """
        self.df = pd.read_csv("data/batch2-individual_preprocessed_data.csv")
        return self

    def use_batch2_group_data_csv(self):
        '''
        Load group-level data for the dataset.
        '''
        self.df = pd.read_csv("data/batch2-group_preprocessed_data.csv")
        return self

    def use_batch1_individual_data_csv(self):
        """
        Load individual-level data for the dataset.
        """
        self.df = get_individual_dataframe(not_taken_as_nan=False)
        return self

    def use_batch1_group_data_csv(self):
        '''
        Load group-level data for the dataset.
        '''
        self.df = get_group_dataframe()
        return self

    def use_combined_individual_data_csv(self):
        batch1_df = pd.read_csv("data/individual_preprocessed_data.csv")
        batch2_df = pd.read_csv("data/batch2-individual_preprocessed_data.csv")
        batch2_df = slim_down_features(batch1_df,batch2_df)

        # used to filter courses taken after 491, dropped on setting target
        batch1_df['batch']  = 1
        batch2_df['batch']  = 2

        combined_df = pd.concat([batch1_df, batch2_df])
        self.df = combined_df
        return self

    def use_combined_group_data_csv(self):
        batch1_df = pd.read_csv("data/group_preprocessed_data.csv")
        batch2_df = pd.read_csv("data/batch2-group_preprocessed_data.csv")
        batch2_df = slim_down_features(batch1_df,batch2_df)
        # used to filter courses taken after 491, dropped on setting target
        batch1_df['batch']  = 1
        batch2_df['batch']  = 2
        combined_df = pd.concat([batch1_df, batch2_df])
        self.df = combined_df
        return self

    def _ensure_data_loaded(self):
        """
        Ensure that the data has been loaded before proceeding.

        Raises:
            DataNotLoadedError: If the data has not been loaded.
        """
        if self.df is None:
            raise DatasetLoaderError()

    def _ensure_features_set(self, error_if_empty: bool = False):
        """
        Ensure that the features have been set before proceeding.

        Args:
            error_if_empty(bool): if True, raises an error if features are empty.
                if False, uses all features in the DataFrame.

        Raises:
            DatasetLoaderError: If the features have not been set and error_if_empty is True.
        """
        if not self.features and error_if_empty:
            raise DatasetLoaderError(
                "Features have not been set. Please set the features before attempting to process them.")

        if not self.features and not error_if_empty:
            logging.warning("Features have not been set. Using all columns in the DataFrame as features.")
            self.use_all_features()

    def _ensure_target_set(self):
        """
        Ensure that the target has been set before proceeding.

        Raises:
            DatasetLoaderError: If the target has not been set.
        """
        if not self.target:
            raise DatasetLoaderError("Target has not been set. Please set the target before attempting to process it.")

    def _drop_other_targets(self):
        """
        Drop other target columns (and workload columns) from the DataFrame if they exist.
        """
        self._ensure_data_loaded()
        if self.target not in ["491", "492", "491_success", "492_success"]:
            raise DatasetLoaderError(
                "Target must be one of '491', '492', '491_success', or '492_success' to drop other targets.")
        possible_targets = ["491", "492", "491_success", "492_success"]
        target_course = "491" if self.target.startswith("491") else "492"
        for possible_target in possible_targets:
            if possible_target in self.df.columns and not possible_target.startswith(target_course):
                self.df.drop(columns=possible_target, inplace=True)

        possible_workload_cols = ["491_SemesterLoad", "492_SemesterLoad"]
        for workload_col in possible_workload_cols:
            if workload_col in self.df.columns and not workload_col.startswith(target_course):
                self.df.drop(columns=workload_col, inplace=True)
                if workload_col in self.features: self.features.remove(workload_col)
        return self

    def replace_missing_values(self, replace_with=-1):
        """
        Replace missing values in the DataFrame with a specified value.

        Parameters:
            replace_with (int): The value to replace missing values with. Default is -1.
        """
        self._ensure_data_loaded()
        self.df.fillna(replace_with, inplace=True)
        return self

    def remove_non_numeric_columns(self):
        """
        Remove non-numeric columns from the DataFrame.
        """
        self._ensure_data_loaded()
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df = self.df[numeric_cols]
        return self

    def create_average_prereq_grade_column(self):
        """ Create a column for the average grade of all prerequisite courses."""
        self._ensure_data_loaded()
        prereq_cols = data_labels.old_prereqs
        self.df['average_prereq_grade'] = self.df[prereq_cols].mean(axis=1, skipna=True)
        self.features += ['average_prereq_grade']
        return self

    def _get_relevant_course_cols(self, row):
        """Get relevant course columns based on relevancy flags,
        excluding courses taken strictly after 491 when target is '491_success'."""
        relevant_cols = [col for col in self.df.columns
                         if col.isdigit()
                         and f"{col}_Relevancy" in self.df.columns
                         and row[f"{col}_Relevancy"]]
        if self.target == '491_success':
            relevant_cols = [col for col in relevant_cols
                             if f"{col}_Semester" not in self.df.columns
                             or not (0 <= row[f"{col}_Semester"] < _491_SEMESTER_VALUE)]
        elif not self.target:
            logging.warning("Target has not been set before relevance calculation. Be careful!")
        return relevant_cols

    def create_average_relevant_grade_column(self, value_if_none_relevant: float = np.nan):
        """ Create a column for the average grade of all relevant courses."""
        self._ensure_data_loaded()

        def average_relevant_grades(row):
            relevant_cols = self._get_relevant_course_cols(row)
            if not relevant_cols and np.isnan(value_if_none_relevant):
                raise DatasetLoaderError(f"No relevant courses found for {row}.")
            if not relevant_cols:
                return value_if_none_relevant
            return row[relevant_cols].mean()

        self.df['average_relevant_grade'] = self.df.apply(average_relevant_grades, axis=1)
        self.features += ['average_relevant_grade']
        return self

    def create_relevance_ratio_column(self, value_if_none_relevant: float = np.nan):
        """ Create a column for the ratio of relevant courses passed to total relevant courses."""
        self._ensure_data_loaded()

        def calculate_relevance_ratio(row):
            relevant_cols = self._get_relevant_course_cols(row)
            if not relevant_cols and np.isnan(value_if_none_relevant):
                raise DatasetLoaderError(f"No relevant courses found for {row}.")
            if not relevant_cols:
                return value_if_none_relevant
            relevant_courses_passed = sum(row[col] > 4 for col in relevant_cols)
            return relevant_courses_passed / len(relevant_cols)

        self.df['relevance_ratio'] = self.df.apply(calculate_relevance_ratio, axis=1)
        self.features += ['relevance_ratio']
        return self

    def create_weighted_grade_columns(self):
        """ Create a weighted grade column which multiplies each course grade by its relevance

        params:
            drop_relevance_cols(bool): if True, drop the relevance columns after creating the weighted grade column
        """
        self._ensure_data_loaded()

        # create empty weighted columns for every course with relevance
        for col in self.df.columns:
            if col.isdigit() and f"{col}_Relevancy" in self.df.columns:
                self.df[f"{col}_weighted"] = 0

        def calculate_weighted_grade(row):
            course_grade_cols = [col for col in self.df.columns if
                                 col.isdigit() and f"{col}_Relevancy" in self.df.columns]
            for col in course_grade_cols:
                if pd.isna(row[col]) or pd.isna(row[f"{col}_Relevancy"]):
                    raise ValueError(f"NaN value found in course grade or relevancy for column {col}.")
                row[f"{col}_weighted"] = (row[col] * row[f"{col}_Relevancy"])
            return row

        self.df = self.df.apply(calculate_weighted_grade, axis=1)
        self.features += [col for col in self.df.columns if col.endswith('_weighted')]
        return self

    def use_all_features(self):
        """ Use all columns except identifiers and target columns as features."""
        self.features = [col for col in self.df.columns if
                         col not in ['Team ID', 'student Id', '491_success', '492_success', '491', '492']]
        return self

    def temp_remove_relevancy_columns(self):
        """ DEPRECATED! Temp function to automate complete gridsearch experiments, removes relevancy columns from features."""
        self._ensure_data_loaded()
        self._ensure_features_set()
        self.features = [col for col in self.features if not col.endswith('_Relevancy')]
        return self

    def temp_remove_semester_columns(self):
        """ DEPRECATED! Temp function to automate complete gridsearch experiments, removes semester columns from features."""
        self._ensure_data_loaded()
        self._ensure_features_set()
        self.features = [col for col in self.features if not col.endswith('_Semester')]
        return self

    def add_feature_columns(self, feature_columns: list, error_if_missing: bool = False):
        """
        Add specific feature columns to the features selection. Not applied until a data returning method is called.

        Parameters:
            error_if_missing (bool): If True, raises an error if any feature column is missing.
                if False, skips missing columns and logs a warning.
            feature_columns (list): List of column names to add as features.
        """
        self._ensure_data_loaded()
        for col in feature_columns:
            if col not in self.df.columns and error_if_missing:
                raise DatasetLoaderError(f"Feature column '{col}' does not exist in the DataFrame.")
            if col not in self.df.columns and not error_if_missing:
                logging.warning(f"Feature column '{col}' does not exist in the DataFrame. Skipping.")
                continue
            self.features.append(col)
        return self

    def use_academic_features(self, add_cgpa_and_gpa: bool = True, add_semester_load: bool = True,
                              add_course_semester: bool = False, add_relevance_cols: bool = False):
        """
        Use academic features to the features selection. Not applied until a data returning method is called.
        Parameters:
            add_relevance_cols (bool): Whether to add course relevance features.
            add_course_semester (bool): Whether to add the semester in which each course was taken as a feature
            add_cgpa_and_gpa (bool): Whether to add GPA and CGPA features.
            add_semester_load (bool): Whether to add semester load features.
        """
        self._ensure_data_loaded()
        self._add_course_grades_features()
        if add_cgpa_and_gpa: self._add_gpa_cgpa_features()
        if add_semester_load: self._add_semester_load_features()
        if add_course_semester: self._add_course_semester_features()
        if add_relevance_cols: self._add_relevance_cols()

        return self

    def use_teamwork_features(self):
        """
        Use Teamwork Quality (TWQ) features as the features selection. Not applied until a data returning method is called.
        """
        self._ensure_data_loaded()
        self.features += twq_sections
        return self

    def use_academic_and_twq_features(self, add_cgpa_and_gpa: bool = True, add_semester_workload: bool = True,
                                      add_course_semester: bool = False, add_relevance_cols: bool = False):
        """
        Use both academic and Teamwork Quality (TWQ) features as the features selection. Not applied until a data returning method is called.

        params:
            add_not_taken (bool): Whether to add 'not taken' course features, if present.
            add_cgpa_and_gpa (bool): Whether to add GPA and CGPA features.
            add_semester_workload (bool): Whether to add semester load features.
        """
        self.use_academic_features(add_cgpa_and_gpa=add_cgpa_and_gpa, add_semester_load=add_semester_workload,
                                   add_course_semester=add_course_semester, add_relevance_cols=add_relevance_cols)
        self.use_teamwork_features()
        return self

    def use_features(self, features: list):
        """ Set specific feature columns to the features selection. Not applied until a data returning method is called.

        Parameters:
            features (list): List of column names to set as features."""

        if features == data_labels.new_prereqs:
            self._ensure_data_loaded()
            self.df['242_or_443'] = self.df[['242', '443']].max(axis=1)
            features = [col if col not in ['242', '443'] else '242_or_443' if col == '242' else None
                       for col in features]
            features = [col for col in features if col is not None]
            features = list(dict.fromkeys(features))

        self.features = []
        self.add_feature_columns(features)
        return self

    def _add_course_grades_features(self):
        """
        Add course grades to the features selection. Not applied until a data returning method is called.
        """
        self._ensure_data_loaded()
        self.features += [col for col in self.df.columns if col.isdigit() and col not in ['491', '492']]
        return self

    def _add_gpa_cgpa_features(self):
        """
        Add GPA and CGPA features to the features selection. Not applied until a data returning method is called.
        """
        self._ensure_data_loaded()
        self.features += ["GPA", "CGPA"]
        return self

    def _add_semester_load_features(self):
        """
        Add semester load features to the features selection. Not applied until a data returning method is called.
        """
        self._ensure_data_loaded()
        semester_load_columns = [col for col in self.df.columns if col.endswith('_SemesterLoad')]
        self.features += semester_load_columns
        return self

    def _nullify_courses_after_491_cutoff(self):
        """Set grade, semester, and relevancy to -1/0 for courses taken strictly after the 491 semester.
        After batch-specific preprocessing, the 491 semester encodes to _491_SEMESTER_VALUE (1).
        Courses taken strictly after have semester values in [0, _491_SEMESTER_VALUE)."""
        for col in self.df.columns:
            if not col.isdigit() or f"{col}_Semester" not in self.df.columns:
                continue
            sem = self.df[f"{col}_Semester"]
            mask = (sem >= 0) & (sem < _491_SEMESTER_VALUE)
            self.df.loc[mask, col] = -1
            self.df.loc[mask, f"{col}_Semester"] = -1
            if f"{col}_Relevancy" in self.df.columns:
                self.df.loc[mask, f"{col}_Relevancy"] = 0

    def set_target(self, target_column):
        """
        Set the target column for the dataset.

        Parameters:
            target_column (str): The name of the target column to set.
        """
        self._ensure_data_loaded()
        if target_column not in self.df.columns:
            raise DatasetLoaderError(f"Target column '{target_column}' does not exist in the DataFrame.")

        if target_column not in ["491_success", "492_success"]:
            logging.warning(
                "You are attempting to perform linear regression instead of binary classification. Are you sure?")
        elif target_column not in ["491", "492"]:
            DatasetLoaderError(f"Target is not a graduation project course.")

        if target_column == "491_success":
            self._nullify_courses_after_491_cutoff()

        if 'batch' in self.df.columns:
            self.df.drop("batch", axis=1, inplace=True)
        self.target = target_column
        return self

    def create_not_taken_columns(self):
        """
        Generate 'not taken' columns for each course that has not taken values in the DataFrame.
        Generated columns are added to the features selection.
        """
        self._ensure_data_loaded()
        self._ensure_features_set()
        for col in self.df.columns:
            # Skip columns that have no missing values
            if not self.df[col].isin([-1]).any() and not self.df[col].isna().any():
                continue

            # Create non-taken course columns
            if col.isdigit():
                self.df[f"{col}_not_taken"] = self.df[col].apply(lambda x: 1 if x == -1 or pd.isna(x) else 0)
                self.features += [f"{col}_not_taken"]
        return self

    def get_features_and_target(self):
        """
        Get the features and target from the DataFrame.

        Returns:
            tuple: A tuple containing the features DataFrame and the target Series.
        """
        self._ensure_data_loaded()

        if not self.features:
            logging.warning("Features have not been set. Using all columns in the DataFrame as features.")
            self.use_all_features()

        if not self.target:
            raise DatasetLoaderError("Target has not been set. Please set the target before attempting to process it.")

        self._drop_other_targets()

        missing_features = [feature for feature in self.features if feature not in self.df.columns]
        for feature in missing_features:
            logging.warning(f"Feature '{feature}' does not exist in the DataFrame. Ignoring.")
            self.features.remove(feature)
        return self.df[self.features], self.df[self.target]

    def get_dataframe(self):
        """
        Get the DataFrame with the current features and target.

        Returns:
            pd.DataFrame: The DataFrame containing the features and target.
        """
        if 'batch' in self.df.columns:
            self.df.drop("batch", axis=1, inplace=True)
        identifier_pd = pd.DataFrame()
        for identifier in data_labels.identifier_cols:
            if identifier in self.df.columns:
                identifier_pd[identifier] = self.df[identifier]
        if not self.target:
            self.target = '491'
        features_pd, _ = self.get_features_and_target()
        target_cols = [col for col in ['491', '492', '491_success', '492_success'] if col in self.df.columns]
        target_pd = self.df[target_cols]
        return pd.concat([identifier_pd, features_pd, target_pd], axis=1)

    def _add_course_semester_features(self):
        """
        Add course semester features to the features selection. Not applied until a data returning method is called.
        """
        self._ensure_data_loaded()
        semester_columns = [col for col in self.df.columns if col.endswith('_Semester')]
        self.features += semester_columns
        return self

    def _add_relevance_cols(self):
        """
        Add course relevance features to the features selection. Not applied until a data returning method is called.
        """
        self._ensure_data_loaded()
        relevance_columns = [col for col in self.df.columns if col.endswith('_Relevancy')]
        self.features += relevance_columns
        return self


def slim_down_features(target_feature_dataset, feature_dataset_to_slim_down):
    features_to_keep = target_feature_dataset.columns
    return feature_dataset_to_slim_down[features_to_keep]

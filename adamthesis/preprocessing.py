import os

import numpy as np
import pandas as pd
import adamthesis.data_labels as data_labels
import matplotlib.pyplot as plt

INDIVIDUAL_OUTPUT_FILE = 'individual_preprocessed_data.csv'
GROUP_OUTPUT_FILE = 'group_preprocessed_data.csv'
BATCH2_INDIVIDUAL_OUTPUT_FILE = 'batch2-individual_preprocessed_data.csv'
BATCH2_GROUP_OUTPUT_FILE = 'batch2-group_preprocessed_data.csv'

# To maintain consistent representation across both batches
TRANSLATION_DICTIONARY = {
    '0': np.nan,
    0: np.nan,
    0.0: np.nan,
    'Pass': 'P'
}
COLUMN_TRANSLATION_DICTIONARY = {
    ' Team ID': 'Team ID',
    'CNG476': '476',
    'CNG476_Relevancy': '476_Relevancy',
}

#NOTE: I wanted to do everything programmatically but some things are a lot more painful so I'm documenting my manual preprocessing changes for batch2 here:
# converting first '1' column to '409'
# converting second '1' column to '477_Relevancy'


def convert_nan_relevancy_to_zero(data):
    relevancy_cols = [col for col in data.columns if col.endswith('_Relevancy')]
    for col in relevancy_cols:
        data[col] = data[col].fillna(0).astype(int)
        if not data[col].between(0, 1).all():
            invalid_values = data[~data[col].between(0, 1)][col]
            raise ValueError(f"Invalid values for {col}: {invalid_values}")


def _preprocess(data, reference_year):
    """Shared preprocessing pipeline for both batches."""
    data.rename(columns=COLUMN_TRANSLATION_DICTIONARY, inplace=True)

    validate_course_grades(data)
    validate_gpa_values(data)
    try:
        validate_twq_answers(data)
    except ValueError as e:
        print(f"TWQ validation error: {e}")
        print("Filling missing TWQ answers with mean values.")
        fill_missing_twq_answers_with_mean(data)
        validate_twq_answers(data)

    drop_semester_number_columns(data)

    encode_gender_as_binary(data)
    encode_nationality_as_binary(data)
    convert_semester_number_to_number_of_semesters(data, reference_year=reference_year)
    encode_letter_grades(data)
    convert_nan_relevancy_to_zero(data)

    calculate_reverse_column_scores(data)
    average_survey_sections(data)
    add_binary_success_columns(data)
    return data


def _save_preprocessed(data, individual_file, group_file):
    """Save individual and group CSVs."""
    individual_csv_path = os.path.join("data", individual_file)
    data.to_csv(individual_csv_path, index=False)

    group_data = aggregate_group_data(data)
    group_csv_path = os.path.join("data", group_file)
    group_data.to_csv(group_csv_path, index=False)


def main():
    data = pd.read_csv('data/data.csv')
    data = _preprocess(data, reference_year=2023)
    _save_preprocessed(data, INDIVIDUAL_OUTPUT_FILE, GROUP_OUTPUT_FILE)

def convert_semester_number_to_number_of_semesters(data, reference_year=2023):
    ''' Convert semester number in format YYYYs to number of semesters relative to end of {reference_year}2.
    Higher values = earlier semesters. The reference semester (e.g. 20232 for reference_year=2023) encodes to 1.'''
    def semester_str_to_number(sem):
        if pd.isna(sem):
            return sem
        sem_str = str(int(sem))
        if len(sem_str) != 5 or not sem_str[:4].isdigit() or sem_str[4] not in ['1', '2']:
            raise ValueError(f"Invalid semester format: {sem_str}")
        year = int(sem_str[:4])
        semester = int(sem_str[4])
        number_of_semesters = (reference_year - year) * 2 + (1 if semester == 2 else 2)
        return int(number_of_semesters)

    if 'Semester Number-SIS' in data.columns:
        data['Semester Number-SIS'] = data['Semester Number-SIS'].apply(semester_str_to_number)

    sem_cols = [col for col in data.columns if col.endswith('_Semester')]
    for col in sem_cols:
        data[col] = data[col].replace(TRANSLATION_DICTIONARY)
        data[col] = data[col].apply(semester_str_to_number)



def drop_semester_number_columns(data):
    """ Drop columns related to semester numbers.
    These columns are not machine-interpretable and the information presented is better captured in Semester Number."""
    data['Length of Study'] = data['Semester Number-SIS']
    semester_cols = [col for col in ['Date of Entry', 'Starting Semester for the Program', 'Semester Number-SIS'] if col in data.columns]
    data.drop(columns=semester_cols, inplace=True)


def _calculate_blau_index(series):
    """
    Calculate the Blau Index diversity for a binary (0/1) series.
    Formula: 1 - Σ(pi²), where pi is the proportion of each category.
    For binary variables: 1 - (p² + (1-p)²), where p is proportion of 1s.
    """
    if series.empty:
        return 0.0

    # Calculate proportion of 1s
    p = series.mean()
    # Blau Index for binary variable
    return 1 - (p**2 + (1-p)**2)

def aggregate_group_data(data):
    agg_funcs = {"CGPA": "mean"}
    agg_funcs.update({"GPA": "mean"})  # I added this later
    agg_funcs.update({"491_SemesterLoad": "mean", "492_SemesterLoad": "mean"})  # I also added these later
    agg_funcs.update({col: "mean" for col in data.columns if col.endswith('_Relevancy')})
    agg_funcs.update({col: "median" for col in data.columns if col.isdigit()})
    agg_funcs.update({col: "mean" for col in data.columns if col in data_labels.twq_sections})
    agg_funcs.update({"491": "min", "492": "min"})
    agg_funcs.update({"491_success": "min", "492_success": "min"})

    # Add Blau Index calculation for Gender and International
    agg_funcs.update({"Gender": _calculate_blau_index})
    agg_funcs.update({"International": _calculate_blau_index})

    data.drop("student Id", axis=1, inplace=True)

    non_defined_cols = set(data.columns) - set(agg_funcs.keys()) - {'Team ID'}
    print(f"Warning: the following columns do not have defined aggregation functions: {non_defined_cols}")

    # Group by 'Team ID' and apply aggregation functions
    data = data.groupby('Team ID').agg(agg_funcs).reset_index()
    return data


def get_all_columns_of_twq_section(section_name):
    """
    Get all columns under a specific TWQ section.

    Parameters:
        section_name (str): Name of the TWQ section.

    Returns:
        list: List of column names related to the specified TWQ section.
    """
    if section_name not in data_labels.twq_sections:
        raise ValueError(f"Invalid section name: {section_name}")

    number_of_questions = data_labels.number_of_questions_per_twq_section.get(section_name)
    return [f"{section_name} {i}" for i in range(1, number_of_questions + 1)]


def load_raw_data() -> pd.DataFrame:
    """
    Load the raw data from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file, defaults to "data/data.csv".

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv("data/data.csv")


def add_binary_success_columns(data):
    for col in ["491", "492"]:
        if col not in data.columns:
            continue
        data[f"{col}_success"] = data[col].apply(lambda x: 1 if x == 10 else 0)

def average_survey_sections(data):
    for section in data_labels.twq_sections:
        question_cols = [f"{section} {i}" for i in range(1, data_labels.number_of_questions_per_twq_section[section] + 1) if f"{section} {i}" in data.columns]
        if question_cols:
            data[section] = data[question_cols].mean(axis=1)
            data.drop(columns=question_cols, inplace=True)



def validate_twq_answers(data):
    # validate that all TWQ columns are present
    validate_all_twq_sections_present(data)
    validate_twq_answers_are_valid(data)


def validate_twq_answers_are_valid(data):

    for col in data_labels.twq_sections:
        for i in range(1, data_labels.number_of_questions_per_twq_section[col] + 1):
            question_col = f"{col} {i}"
            if question_col in data.columns:
                data[question_col] = data[question_col].replace(TRANSLATION_DICTIONARY)
                is_not_nan = data[question_col].notna()
                is_in_range = data[question_col].between(1, 5)
                invalid_entries = data[~(is_not_nan & is_in_range)][question_col]
                if not invalid_entries.empty:
                    raise ValueError(f"Invalid entries found in {question_col}: {invalid_entries.tolist()}")





def fill_missing_twq_answers_with_mean(data):
    def fill_section_with_student_mean(row: pd.Series, section_cols: list):
        section_values = row[section_cols]
        mean_value = section_values.mean()
        return section_values.astype(float).fillna(mean_value)

    for section in data_labels.twq_sections:
        if not any(col.startswith(section) for col in data.columns):
            continue
        section_cols = [col for col in data.columns if col.startswith(section)]
        data[section_cols] = data.apply(
            lambda row: fill_section_with_student_mean(row, section_cols),
            axis=1)


def validate_all_twq_sections_present(data):
    for col in data_labels.twq_sections:
        for i in range(1, data_labels.number_of_questions_per_twq_section[col] + 1):
            question_col = f"{col} {i}"
            if question_col not in data.columns:
                raise ValueError(f"Missing column: {question_col}")


def calculate_reverse_column_scores(data):
    validate_twq_answers(data)
    for col in data_labels.raw_reverse_twq_columns:
        data[col] = 6 - data[col]


def encode_letter_grades(data):
    validate_course_grades(data, print_unique_values=False) # just in case

    course_columns = [col for col in data.columns if col.isnumeric()]
    for course_col in course_columns:
        data[course_col] = data[course_col].map(data_labels.raw_grade_mapping)
        data[course_col] = data[course_col].fillna(-1).astype(int)
        if f"{course_col}_Semester" in data.columns:
            data[f"{course_col}_Semester"] = data[f"{course_col}_Semester"].fillna(-1).astype(int)



def encode_nationality_as_binary(data):
    data['International'] = data['Turkish/International'].map({'T': 0, 'I': 1})
    data.drop(['Turkish/International'], axis=1, inplace=True)
    if data['International'].isnull().any():
        raise ValueError("Invalid values found in 'Turkish/International' column.")


def encode_gender_as_binary(data):
    data['Gender'] = data['Gender'].map({'M': 0, 'F': 1})
    if data['Gender'].isnull().any():
        raise ValueError("Invalid values found in 'Gender' column.")


def validate_course_grades(data, print_unique_values: bool = True):

    # get the columns with numeric names
    course_columns = [col for col in data.columns if col.isnumeric()]

    # print unique values and do a validity check
    for col in course_columns:
        data[col] = data[col].replace(TRANSLATION_DICTIONARY)
        unique_values = data[col].unique()
        if print_unique_values: print(f"Unique values for {col} course: {unique_values}")
        valid_values = data_labels.grade_labels
        invalid_values = data[~data[col].isin(valid_values)].dropna(subset=[col])[col]
        if not invalid_values.empty:
            raise ValueError(f"Invalid values for {col} course: {invalid_values}")



def plot_gpa_boxplot(data: pd.DataFrame):
    data.boxplot(column=data_labels.gpa_cols)
    plt.title('Boxplot for GPA and CGPA')
    plt.xlabel('Columns')
    plt.ylabel('Values')
    plt.show()


def validate_gpa_values(data: pd.DataFrame):
    for col in data_labels.gpa_cols:
        if col in data.columns:
            if not data[col].between(0, 4).all():
                invalid_values = data[~data[col].between(0, 4)][col]
                raise ValueError(f"Invalid values for {col}: {invalid_values}")

def batch2_main():
    data = pd.read_csv('data/batch2-data.csv')
    data = _preprocess(data, reference_year=2024)
    _save_preprocessed(data, BATCH2_INDIVIDUAL_OUTPUT_FILE, BATCH2_GROUP_OUTPUT_FILE)


if __name__ == "__main__":
    main()
    batch2_main()

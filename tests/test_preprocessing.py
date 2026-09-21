import pytest
import pandas as pd
from adamthesis.preprocessing import *
import adamthesis.data_labels as data_labels

@pytest.fixture
def dataframe_with_all_twq_sections_as_1():
    data = {}
    for section in data_labels.twq_sections:
        num_questions = data_labels.number_of_questions_per_twq_section[section]
        for i in range(1, num_questions + 1):
            col_name = f"{section} {i}"
            data[col_name] = [1]
    return pd.DataFrame(data)

@pytest.fixture
def dataframe_with_all_twq_sections_as_1_but_reverse_coded_items_are_5(dataframe_with_all_twq_sections_as_1):
    df = dataframe_with_all_twq_sections_as_1.copy()
    for col in data_labels.raw_reverse_twq_columns:
        if col in df.columns:
            df[col] = 5
    return df


def test_validate_course_grades_valid():

    sample_data = pd.DataFrame({
        'student_id': [1, 2, 3],
        '443': ['AA', 'BB', 'CC'],
        '491': ['BA', 'W', 'FF'],
        '492': ['P', 'DD', 'FD'],
    })

    try:
        validate_course_grades(sample_data, print_unique_values=False)
    except ValueError:
        pytest.fail("validate_course_grades() raised ValueError unexpectedly!")


def test_validate_course_grades_invalid():
    sample_data = pd.DataFrame({
        'student_id': [1, 2, 3],
        '443': ['AA', 'BB', 'ZZ'],
        '491': ['BA', 'W', 'FF'],
        '492': ['P', 'DD', 'FD'],
    })

    with pytest.raises(ValueError):
        validate_course_grades(sample_data, print_unique_values=False)


def test_validate_gpa_values_valid():
    sample_data = pd.DataFrame({
        'student_id': [1, 2, 3],
        'GPA': [3.5, 4.0, 2.8],
        'CGPA': [3.6, 3.9, 2.7],
    })

    try:
        validate_gpa_values(sample_data)
    except ValueError:
        pytest.fail("validate_gpa_values() raised ValueError unexpectedly!")

def test_validate_gpa_values_invalid():
    sample_data = pd.DataFrame({
        'student_id': [1, 2, 3],
        'GPA': [3.5, 4.5, 2.8],
        'CGPA': [3.6, -1.0, 2.7],
    })

    with pytest.raises(ValueError):
        validate_gpa_values(sample_data)

def test_encode_gender_as_binary():
    sample_data = pd.DataFrame({
        'Gender': ['M', 'F', 'F', 'M']
    })

    expected_data = pd.DataFrame({
        'Gender': [0,1,1,0]
    })

    encode_gender_as_binary(sample_data)
    pd.testing.assert_frame_equal(sample_data, expected_data)

def test_encode_gender_as_binary_invalid():
    sample_data = pd.DataFrame({
        'Gender': ['M', 'F', 'X', 'M']
    })

    with pytest.raises(ValueError):
        encode_gender_as_binary(sample_data)

def test_encode_nationality_as_binary():
    sample_data = pd.DataFrame({
        'Turkish/International': ['T', 'I', 'I', 'T']
    })

    expected_data = pd.DataFrame({
        'International': [0,1,1,0]
    })

    encode_nationality_as_binary(sample_data)
    pd.testing.assert_frame_equal(sample_data, expected_data)

def test_encode_nationality_as_binary_invalid():
    sample_data = pd.DataFrame({
        'Turkish/International': ['T', 'I', 'X', 'T']
    })

    with pytest.raises(ValueError):
        encode_nationality_as_binary(sample_data)

def test_encode_letter_grades():
    sample_data = pd.DataFrame({
        '443': ['AA', 'BA', 'CC', None],
        '491': ['BB', 'W', 'FF', 'P'],
        '492': [None, 'DD', 'FD', 'CC'],
    })

    expected_data = pd.DataFrame({
        '443': [10, 9, 6, -1],
        '491': [8, 0, 1, 3],
        '492': [-1, 4, 2, 6],
    })

    encode_letter_grades(sample_data)
    pd.testing.assert_frame_equal(sample_data, expected_data)

def test_encode_letter_grades_all_invalid():
    sample_data = pd.DataFrame({
        '443': ['XX', 'YY', 'ZZ'],
        '491': ['AA', 'BB', 'CC'],
    })

    with pytest.raises(ValueError):
        encode_letter_grades(sample_data)


def test_validate_twq_answers_are_valid():
    sample_data = pd.DataFrame({
        'Communication 1': [1, 2, 3],
        'Coordination 1': [4, 5, 2],
        'Mutual Support 1': [3, 1, 5],
    })

    try:
        validate_twq_answers_are_valid(sample_data)
    except ValueError:
        pytest.fail("validate_twq_answers() raised ValueError unexpectedly!")

def test_validate_twq_answers_with_floats_are_valid():
    # due to mean-filling
    sample_data = pd.DataFrame({
        'Communication 1': [1.0, 2.5, 3.0],
        'Coordination 1': [4.0, 5.0, 2.0],
        'Mutual Support 1': [3.0, 1.0, 5.0],
    })

    try:
        validate_twq_answers_are_valid(sample_data)
    except ValueError:
        pytest.fail("validate_twq_answers() raised ValueError unexpectedly!")

def test_validate_twq_answers_are_invalid():
    sample_data = pd.DataFrame({
        'Communication 1': [1, 2, 6],
        'Coordination 1': [4, 0, 2],
        'Mutual Support 1': [3, 1, 5],
    })

    with pytest.raises(ValueError):
        validate_twq_answers_are_valid(sample_data)

def test_validate_all_twq_sections_present_valid(dataframe_with_all_twq_sections_as_1):
    try:
        validate_all_twq_sections_present(dataframe_with_all_twq_sections_as_1)
    except ValueError:
        pytest.fail("validate_all_twq_sections_present() raised ValueError unexpectedly!")

def test_validate_all_twq_sections_present_invalid():
    sample_data = pd.DataFrame({
        'Communication 1': [1, 2, 3],
        'Coordination 1': [4, 5, 2],
    })

    with pytest.raises(ValueError):
        validate_all_twq_sections_present(sample_data)

def test_calculate_reverse_column_scores(dataframe_with_all_twq_sections_as_1,
                                         dataframe_with_all_twq_sections_as_1_but_reverse_coded_items_are_5):
    sample_df = dataframe_with_all_twq_sections_as_1_but_reverse_coded_items_are_5.copy()
    expected_df = dataframe_with_all_twq_sections_as_1.copy()

    calculate_reverse_column_scores(sample_df)
    pd.testing.assert_frame_equal(sample_df, expected_df)


def test_average_survey_sections():
    sample_data = pd.DataFrame({
        'Communication 1': [1, 2],
        'Communication 2': [3, 4],
        'Coordination 1': [5, 4],
        'Coordination 2': [3, 2],
    })

    expected_data = pd.DataFrame({
        'Communication': [2.0, 3.0],
        'Coordination': [4.0, 3.0],
    })

    average_survey_sections(sample_data)
    pd.testing.assert_frame_equal(sample_data, expected_data)


def test_average_survey_sections_with_fractional_mean():
    sample_data = pd.DataFrame({
        'Communication 1': [1, 2],
        'Communication 2': [2, 3],
        'Coordination 1': [4, 5],
        'Coordination 2': [5, 6],
    })

    expected_data = pd.DataFrame({
        'Communication': [1.5, 2.5],
        'Coordination': [4.5, 5.5],
    })

    average_survey_sections(sample_data)
    pd.testing.assert_frame_equal(sample_data, expected_data)

def test_fill_missing_twq_answers_with_mean():
    sample_data = pd.DataFrame({
        'Communication 1': [1, None, 3],
        'Communication 2': [None, 4, 5],
        'Communication 3': [2, 3, None],
        'Coordination 1': [5, None, None],
        'Coordination 2': [3, 2, 1],
        'Coordination 3': [3, 4, 5],
    })

    expected_data = pd.DataFrame({
        'Communication 1': [1, 3.5, 3],
        'Communication 2': [1.5, 4, 5],
        'Communication 3': [2, 3, 4],
        'Coordination 1': [5, 3, 3],
        'Coordination 2': [3, 2, 1],
        'Coordination 3': [3, 4, 5],
    })

    fill_missing_twq_answers_with_mean(sample_data)
    pd.testing.assert_frame_equal(sample_data, expected_data, check_dtype=False)

def test_add_binary_success_columns():
    sample_data = pd.DataFrame({
        '491': [10, 9, 1, 8],
        '492': [7, 10, 6, 1],
    })

    expected_data = pd.DataFrame({
        '491': [10, 9, 1, 8],
        '492': [7, 10, 6, 1],
        '491_success': [1, 0, 0, 0],
        '492_success': [0, 1, 0, 0],
    })

    add_binary_success_columns(sample_data)
    pd.testing.assert_frame_equal(sample_data, expected_data)

def test_add_binary_success_columns_with_only_491():
    sample_data = pd.DataFrame({
        '491': [10, 9, 1, 8],
    })

    expected_data = pd.DataFrame({
        '491': [10, 9, 1, 8],
        '491_success': [1, 0, 0, 0],
    })

    add_binary_success_columns(sample_data)
    pd.testing.assert_frame_equal(sample_data, expected_data)

def test_converting_semester_number():
    sample_data = pd.DataFrame({
        'Semester Number-SIS': [20232, 20221, 20212, 20201],
        '442_Semester': [20232, 20221, 20212, 20201]
    })

    expected_data = pd.DataFrame({
        'Semester Number-SIS': [1, 4, 5, 8],
        '442_Semester': [1, 4, 5, 8]
    })

    convert_semester_number_to_number_of_semesters(sample_data, reference_year=2023)
    pd.testing.assert_frame_equal(sample_data, expected_data)


def test_converting_semester_number_batch2():
    """Batch 2 preprocessing uses reference_year=2024 to avoid -1 collision."""
    sample_data = pd.DataFrame({
        '442_Semester': [20242, 20241, 20232, 20221]
    })
    expected = pd.DataFrame({'442_Semester': [1, 2, 3, 6]})
    convert_semester_number_to_number_of_semesters(sample_data, reference_year=2024)
    pd.testing.assert_frame_equal(sample_data, expected)


def test_batch2_semesters_no_collision_with_not_taken():
    """Batch 2 semester values must all be > 0 to avoid collision with -1 not-taken sentinel."""
    sample_data = pd.DataFrame({'443_Semester': [20242, 20241, 20232]})
    convert_semester_number_to_number_of_semesters(sample_data, reference_year=2024)
    assert (sample_data['443_Semester'] > 0).all()


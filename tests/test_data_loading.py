import pandas as pd
import pytest
from adamthesis.data_loading import *


def test_load_individual_data_not_taken_as_nan_is_true():
    sample_data = pd.DataFrame(
        {
            "student_id": [1],
            "443": [-1],
            "491": [10],
            "492": [10],
        }
    )
    # Test that the function returns NaN for a course not taken
    result = get_individual_dataframe(sample_data, True)
    assert sample_data["443"].isna().all(), "Expected -1 to be converted to NaN for course 443"


def test_load_individual_data_not_taken_as_nan_is_false():
    sample_data = pd.DataFrame(
        {
            "student_id": [1],
            "443": [-1],
            "491": [10],
            "492": [10],
        }
    )
    # Test that the function returns -1 for a course not taken
    result = get_individual_dataframe(sample_data, False)
    assert (sample_data["443"] == -1).all(), "Expected -1 to remain unchanged for course 443"


def test_dataset_loader_loads_individual_csv():
    expected = get_individual_dataframe()
    result = DatasetLoader().use_batch1_individual_data_csv().df

    pd.testing.assert_frame_equal(result, expected, check_dtype=False, check_like=True)


def test_dataset_loader_loads_group_csv():
    expected = get_group_dataframe()
    result = DatasetLoader().use_batch1_group_data_csv().df

    pd.testing.assert_frame_equal(result, expected, check_dtype=False, check_like=True)


def test_dataset_loader_throws_exception_when_target_not_set():
    with pytest.raises(DatasetLoaderError):
        DatasetLoader().use_batch1_individual_data_csv().get_features_and_target()


def test_dataset_loader_drops_other_targets_correctly():
    sample_df = pd.DataFrame({
        '491': [10, 9, 8],
        '492': [7, 6, 5],
        '491_success': [1, 0, 0],
        '492_success': [0, 0, 0],
        '491_SemesterLoad': [5, 4, 3],
        '492_SemesterLoad': [2, 1, 2],
    })
    result = DatasetLoader().use_dataframe(sample_df).set_target('492').get_dataframe()

    assert not any(col.startswith('491') for col in result.columns), "Columns starting with '491' should be dropped"


def test_create_not_taken_columns():
    sample_df = pd.DataFrame({
        'student_id': [1, 2, 3],
        '443': [10, -1, 8],
        '445': [np.nan, -1, 9],
        '491': [10, 9, 8],
    })
    expected_df = pd.DataFrame({
        'student_id': [1, 2, 3],
        '443': [10, -1, 8],
        '445': [np.nan, -1, 9],
        '443_not_taken': [0, 1, 0],
        '445_not_taken': [1, 1, 0],
        '491': [10, 9, 8],
    })

    result = DatasetLoader().use_dataframe(sample_df).set_target('491').create_not_taken_columns().get_dataframe()

    pd.testing.assert_frame_equal(result, expected_df)


def test_dataset_loader_does_not_load_irrelevant_workload_when_all_features_set():
    result = DatasetLoader().use_batch1_individual_data_csv().use_all_features().set_target('492').get_dataframe()
    assert '491_SemesterLoad' not in result.columns, "Irrelevant workload columns should be dropped when all features are used"


def test_dataset_loader_adds_relevant_workload_to_features():
    result = DatasetLoader().use_batch1_individual_data_csv().use_all_features().set_target('492').get_dataframe()
    assert '492_SemesterLoad' in result.columns, "Relevant workload column should be included in the features"


def test_dataset_loader_works_with_binary_target():
    sample_df = pd.DataFrame({
        'student_id': [1, 2, 3],
        '443': [10, 9, 8],
        '491': [10, 4, 10],
        '491_success': [1, 0, 1],
        '491_SemesterLoad': [5, 4, 3],
        '492': [10, 6, 5],
        '492_success': [1, 0, 0],
        '492_SemesterLoad': [2, 1, 2],

    })
    expected_features = pd.DataFrame({
        'student_id': [1, 2, 3],
        '443': [10, 9, 8],
        '491_SemesterLoad': [5, 4, 3],
    })
    expected_target = pd.Series([1, 0, 1], name='491_success')

    features, target = DatasetLoader().use_dataframe(sample_df).set_target('491_success').get_features_and_target()

    pd.testing.assert_frame_equal(features, expected_features)
    pd.testing.assert_series_equal(target, expected_target)


def test_create_average_prereq_grade_column():
    sample_df = pd.DataFrame({
        '140': [8, 9, 10],
        '213': [10, 9, 9],
        '350': [9, 9, 8],
        '491': [10, 9, 8],
    })
    expected_df = pd.DataFrame({
        '140': [8, 9, 10],
        '213': [10, 9, 9],
        '350': [9, 9, 8],
        'average_prereq_grade': [9.0, 9.0, 9.0],
        '491': [10, 9, 8],
    })

    result = DatasetLoader().use_dataframe(sample_df).set_target(
        '491').use_all_features().create_average_prereq_grade_column().get_dataframe()
    pd.testing.assert_frame_equal(result, expected_df)


def test_create_average_relevant_grade_column():
    sample_df = pd.DataFrame({
        '443': [9, 9, 10],
        '443_Relevancy': [1, 0, 1],
        '445': [10, 9, 9],
        '445_Relevancy': [0, 1, 1],
        "491": [10, 9, 8],
    })

    expected_df = pd.DataFrame({
        '443': [9, 9, 10],
        '443_Relevancy': [1, 0, 1],
        '445': [10, 9, 9],
        '445_Relevancy': [0, 1, 1],
        'average_relevant_grade': [9.0, 9.0, 9.5],
        "491": [10, 9, 8],
    })

    result = DatasetLoader().use_dataframe(sample_df).set_target(
        "491").use_all_features().create_average_relevant_grade_column().get_dataframe()
    pd.testing.assert_frame_equal(result, expected_df)


def test_create_relevance_ratio_column():
    sample_df = pd.DataFrame({
        '443': [9, 9, 10],
        '443_Relevancy': [1, 0, 1],
        '445': [10, 9, 3],
        '445_Relevancy': [0, 1, 1],
        '352': [3, 7, 3],
        '352_Relevancy': [1, 1, 1],
        '491': [10, 9, 8],
    })

    expected_df = pd.DataFrame({
        '443': [9, 9, 10],
        '443_Relevancy': [1, 0, 1],
        '445': [10, 9, 3],
        '445_Relevancy': [0, 1, 1],
        '352': [3, 7, 3],
        '352_Relevancy': [1, 1, 1],
        'relevance_ratio': [0.5, 1, 1 / 3],
    })

    result, _ = DatasetLoader().use_dataframe(sample_df).use_all_features().set_target(
        '491').create_relevance_ratio_column().get_features_and_target()
    pd.testing.assert_frame_equal(result, expected_df)


def test_create_weighted_grade_columns():
    sample_df = pd.DataFrame({
        '443': [9, 9, 10],
        '443_Relevancy': [1, 0, 1],
        '445': [10, 9, 3],
        '445_Relevancy': [0, 1, 1],
        '352': [3, 7, 3],
        '352_Relevancy': [1, 1, 1],
        '491': [10, 9, 8],
    })

    expected_df = pd.DataFrame({
        '443': [9, 9, 10],
        '443_Relevancy': [1, 0, 1],
        '443_weighted': [9, 0, 10],
        '445': [10, 9, 3],
        '445_weighted': [0, 9, 3],
        '445_Relevancy': [0, 1, 1],
        '352': [3, 7, 3],
        '352_weighted': [3, 7, 3],
        '352_Relevancy': [1, 1, 1],
    })

    result, _ = DatasetLoader().use_dataframe(sample_df).use_all_features().set_target(
        '491').create_weighted_grade_columns().get_features_and_target()
    pd.testing.assert_frame_equal(result, expected_df, check_like=True)


def test_create_weighted_grade_columns_if_not_taken():
    sample_df = pd.DataFrame({
        '443': [9, 9, -1],
        '443_Relevancy': [1, 0, 1],
        '445': [-1, 9, 3],
        '445_Relevancy': [0, 1, 1],
        '352': [3, 7, 3],
        '352_Relevancy': [1, 1, 1],
        '491': [10, 9, 8],
    })

    expected_df = pd.DataFrame({
        '443': [9, 9, -1],
        '443_Relevancy': [1, 0, 1],
        '443_weighted': [9, 0, -1],
        '445': [-1, 9, 3],
        '445_weighted': [0, 9, 3],
        '445_Relevancy': [0, 1, 1],
        '352': [3, 7, 3],
        '352_weighted': [3, 7, 3],
        '352_Relevancy': [1, 1, 1],
    })

    result, _ = DatasetLoader().use_dataframe(sample_df).use_all_features().set_target(
        '491').create_weighted_grade_columns().get_features_and_target()
    pd.testing.assert_frame_equal(result, expected_df, check_like=True)


def test_create_weighted_grade_columns_no_relevancy():
    sample_df = pd.DataFrame({
        '443': [9, 9, 10],
        '443_Relevancy': [1, 0, 1],
        '445': [10, 9, 3],
        '445_Relevancy': [0, 1, 1],
        '352': [3, 7, 3],
        '352_Relevancy': [1, 1, 1],
        '491': [10, 9, 8],
    })

    expected_df = pd.DataFrame({
        '443': [9, 9, 10],
        '443_weighted': [9, 0, 10],
        '445': [10, 9, 3],
        '445_weighted': [0, 9, 3],
        '352': [3, 7, 3],
        '352_weighted': [3, 7, 3],
    })
    result, _ = DatasetLoader().use_dataframe(sample_df).use_academic_features().set_target(
        '491').create_weighted_grade_columns().get_features_and_target()
    pd.testing.assert_frame_equal(result, expected_df, check_like=True)

def test_batch1individual_not_equal_to_batch2individual():
    batch1 = DatasetLoader().use_batch1_individual_data_csv().get_dataframe()
    batch2 = DatasetLoader().use_batch2_individual_data_csv().get_dataframe()
    assert not batch1.equals(batch2), "Batch 1 and Batch 2 individual data should not be equal"

def test_batch1group_not_equal_to_batch2group():
    batch1 = DatasetLoader().use_batch1_group_data_csv().get_dataframe()
    batch2 = DatasetLoader().use_batch2_group_data_csv().get_dataframe()
    assert not batch1.equals(batch2), "Batch 1 and Batch 2 group data should not be equal"
    
def test_batch1group_is_smaller_than_batch1_individual():
    individual = DatasetLoader().use_batch1_individual_data_csv().get_dataframe()
    group = DatasetLoader().use_batch1_group_data_csv().get_dataframe()
    assert individual.shape[0] > group.shape[0], "Group data should have fewer rows than individual data"

def test_batch2group_is_smaller_than_batch2_individual():
    individual = DatasetLoader().use_batch2_individual_data_csv().get_dataframe()
    group = DatasetLoader().use_batch2_group_data_csv().get_dataframe()
    assert individual.shape[0] > group.shape[0], "Group data should have fewer rows than individual data"


# --- 491 semester cutoff filtering tests ---

def _make_491_cutoff_df(semester_values, batch_values, grade=8):
    """Helper to build a synthetic df for 491 cutoff tests.
    semester_values: list of ints for '443_Semester'
    batch_values: list of ints for 'batch'"""
    return pd.DataFrame({
        '443': [grade] * len(semester_values),
        '443_Semester': semester_values,
        '491': [10] * len(semester_values),
        '491_success': [1] * len(semester_values),
        '492': [10] * len(semester_values),
        '492_success': [1] * len(semester_values),
        'batch': batch_values,
    })


def test_491_cutoff_post_491_course_is_nullified():
    """Course with semester_value == 0 (strictly after 491) is set to -1."""
    df = _make_491_cutoff_df([0], [1])
    result = DatasetLoader().use_dataframe(df).set_target('491_success').get_dataframe()
    assert result['443'].iloc[0] == -1
    assert result['443_Semester'].iloc[0] == -1


def test_491_cutoff_same_semester_course_not_nullified():
    """Course in the same semester as 491 (semester_value == 1) is NOT nullified."""
    df = _make_491_cutoff_df([1], [1])
    result = DatasetLoader().use_dataframe(df).set_target('491_success').get_dataframe()
    assert result['443'].iloc[0] == 8


def test_491_cutoff_pre_491_course_not_nullified():
    """Course before 491 semester (semester_value > 1) is NOT nullified."""
    df = _make_491_cutoff_df([2], [1])
    result = DatasetLoader().use_dataframe(df).set_target('491_success').get_dataframe()
    assert result['443'].iloc[0] == 8


def test_491_cutoff_not_taken_sentinel_not_modified():
    """Course with semester_value == -1 (not taken) stays -1."""
    df = _make_491_cutoff_df([-1], [1], grade=-1)
    result = DatasetLoader().use_dataframe(df).set_target('491_success').get_dataframe()
    assert result['443'].iloc[0] == -1
    assert result['443_Semester'].iloc[0] == -1


def test_491_cutoff_semester_column_also_nullified():
    """When grade is nullified, semester column is also set to -1."""
    df = _make_491_cutoff_df([0], [1])
    result = DatasetLoader().use_dataframe(df).set_target('491_success').get_dataframe()
    assert result['443_Semester'].iloc[0] == -1


def test_491_cutoff_batch2_pre_491_course_not_nullified():
    """Batch 2 student with course at semester_value=2 (before their 491) is NOT nullified."""
    df = _make_491_cutoff_df([2], [2])
    result = DatasetLoader().use_dataframe(df).set_target('491_success').get_dataframe()
    assert result['443'].iloc[0] == 8


def test_491_cutoff_batch_column_dropped():
    """'batch' column is dropped after set_target('491_success')."""
    df = _make_491_cutoff_df([1], [1])
    result = DatasetLoader().use_dataframe(df).set_target('491_success').get_dataframe()
    assert 'batch' not in result.columns


def test_491_cutoff_get_relevant_course_cols_excludes_post_cutoff():
    """Average relevant grade excludes courses with semester after 491."""
    df = pd.DataFrame({
        '443': [8],
        '443_Semester': [0],  # after 491
        '443_Relevancy': [1],
        '445': [10],
        '445_Semester': [2],  # before 491
        '445_Relevancy': [1],
        '491': [10],
        '491_success': [1],
        '492': [10],
        '492_success': [1],
        'batch': [1],
    })
    result = (DatasetLoader().use_dataframe(df)
              .set_target('491_success')
              .use_all_features()
              .create_average_relevant_grade_column()
              .get_dataframe())
    assert result['average_relevant_grade'].iloc[0] == 10.0


def test_492_target_does_not_apply_491_filtering():
    """When target is '492_success', courses with semester_value=0 are NOT nullified."""
    df = pd.DataFrame({
        '443': [8],
        '443_Semester': [0],
        '491': [10],
        '491_success': [1],
        '491_SemesterLoad': [5],
        '492': [10],
        '492_success': [1],
        '492_SemesterLoad': [3],
        'batch': [1],
    })
    result = DatasetLoader().use_dataframe(df).set_target('492_success').get_dataframe()
    assert result['443'].iloc[0] == 8

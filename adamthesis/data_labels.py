# Define TWQ columns
twq_sections = [
    'Communication', 'Coordination', 'Mutual Support', 'Effort',
    'Cohesion', 'Balance', 'Satisfaction', 'Learning'
]

number_of_questions_per_twq_section ={
    'Communication': 10,
    'Coordination': 4,
    'Mutual Support': 7,
    'Effort': 4,
    'Cohesion': 10,
    'Balance': 3,
    'Satisfaction': 4,
    'Learning': 4
}

# Define raw (pre-preprocessing) TWQ question columns
raw_twq_columns = []

# Define the list of course columns
course_columns = [
    '140', '213', '350', '242', '351',
    '443', '491', '492', '352', '371',
    '409', '436', '438', '445', '457',
    '462', '465', '466', '477', '483',
    '487', '495', '514']

# Define higher-level groupings of columns
identifier_cols = ['Team ID', 'student Id']

demographics_cols = ['Gender', 'International']

length_of_study_cols = ['Starting Semester for the Program', 'Semester Number-SIS']

gpa_cols = ['GPA', 'CGPA']


# Define the list of prerequisites before change
old_prereqs = ['140', '213', '350']

# Define the list of prerequisites after change
new_prereqs = ['140','213','350','351','242','443']

# Define the Likert scale labels
likert_labels = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]

# Define the list of values for the y-axis
grade_labels = ["W","FF","FD","P","DD","DC","CC","CB","BB","BA","AA"]

# Define the mapping of letter grades to numeric values
raw_grade_mapping = {'AA': 10, 'BA': 9, 'BB': 8, 'CB': 7, 'CC': 6, 'DC': 5, 'DD': 4, 'P': 3, 'FD': 2, 'FF': 1, 'W': 0}

# Define the reverse valued columns of the TWQ survey
raw_reverse_twq_columns = ['Balance 3', 'Cohesion 3', 'Cohesion 6', 'Effort 4', 'Coordination 4', 'Communication 6', 'Communication 7']

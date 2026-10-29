from data_analysis.query_functions import create_supabase_connection, query_test_training_result

supabase = create_supabase_connection()

def test_query_test_training_result():
    table = query_test_training_result(supabase)
    keys = []
    for key in table.data[0]:
        keys.append(key)
    assert keys[0] == "training_id", "training_id not the first column"
    assert keys[1] == "epoch_no", "epoch_no not the second column"
    assert keys[2] == "accuracy", "accuracy not the third column"
    try:
        no_key = keys[3]
    except IndexError:
        error_raised = True
    assert error_raised == True, "A fourth column exists in training_result table"
    
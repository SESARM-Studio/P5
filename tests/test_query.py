from data_analysis.query_functions import create_supabase_connection, query_test_training_result

supabase = create_supabase_connection()

def test_query_test_training_result():
    table = query_test_training_result(supabase)
    keys = table.data[0].keys
    assert keys[0] == "training_id"
    assert keys[1] == "epoch_no"
    assert keys[2] == "accuracy"
    assert keys[3] == False

from helper_func import *
from database_func import *
from create_client import *

### test helper functions ###
def test_generate_results_array():
    test_results = generate_results_array(10)
    assert len(test_results) == 600, "generate_results_array() should return an array of length 600"
    assert test_results[0] == [10, 1, 0.501], "generate_results_array() should return an array of arrays with seed, epoch, and accuracy"
    assert test_results[100] == [10, 101, 0.601], "generate_results_array() should return an array of arrays with seed, epoch, and accuracy"


def test_results_to_dict():
    test_results = [[10, 1, 0.501], [10, 2, 0.502], [10, 3, 0.503]]
    test_dict = results_to_dict(test_results)
    assert len(test_dict) == 3, "results_to_dict() should return a list of dictionaries with length equal to the input array"
    assert test_dict[0] == {"seed": 10, "epoch": 1, "accuracy": 0.501}, "results_to_dict() should return a list of dictionaries with keys 'seed', 'epoch', and 'accuracy'"
    assert test_dict[1] == {"seed": 10, "epoch": 2, "accuracy": 0.502}, "results_to_dict() should return a list of dictionaries with keys 'seed', 'epoch', and 'accuracy'"
    assert test_dict[2] == {"seed": 10, "epoch": 3, "accuracy": 0.503}, "results_to_dict() should return a list of dictionaries with keys 'seed', 'epoch', and 'accuracy'"
    
    
### test database functions ###
client = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))

def test_insert_results_to_database():
    test_results = [[40, 1, 0.51], [40, 2, 0.52], [40, 3, 0.53]]

    try:
        response = insert_results_to_database(client, test_results)
        assert len(response.data) == len(test_results)
    finally:
        client.table("result").delete().eq("seed", 40).execute()
        
    
def test_insert_results_to_database_with_invalid_results():
    test_results = [[50, 1, "invalid_accuracy"], [50, 2, 0.52], [50, 3, 0.53]]
    response = insert_results_to_database(client, test_results)
    assert response is None, "insert_results_to_database() should return None when an error occurs during insertion" 
    
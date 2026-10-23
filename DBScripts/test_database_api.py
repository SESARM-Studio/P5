from datetime import datetime
from datetime import timezone

from .database_api import simple_insert_results_to_database

"""
### Database integration test ###
def test_simple_insert_results_to_database():
    test_results = [[1, 0.51], [2, 0.52], [3, 0.53]]
    client = None
    training_id = None
    
    start_time = datetime.now(timezone.utc).isoformat()
    end_time = datetime.now(timezone.utc).isoformat()
    

    try:
        # Insert training run and results
        client, training_id, response = simple_insert_results_to_database(
            "integrationTest", 100, test_results, start_time, end_time
        )        
        
        # Verify insertion
        assert len(response.data) == len(test_results)
        assert response.data[0]["epoch_no"] == 1
        assert response.data[0]["accuracy"] == 0.51
        

    finally:
        # Clean up database
        if client is not None and training_id is not None:
            client.table("test_training_result").delete().eq(
                "training_id", training_id
            ).execute()

            client.table("test_training_run").delete().eq(
                "training_id", training_id
            ).execute()

"""


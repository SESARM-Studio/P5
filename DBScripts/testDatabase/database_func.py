from helper_func import *

### Function to insert results from an experiment. Results must be a two-dimensional array with seed, epoch, and accuracy ###
def insert_results_to_database(client, results):
    try:
        rows = results_to_dict(results)
        response = (
            client.table("result")
            .insert(rows)
            .execute()
        )
        
    except Exception as e:
        print(f"Error occurred while inserting results: {e}")
        response = None
        
    return response


"""
#insert data using a function in the database
response = (
    supabase.rpc("insert_results", {"p_results": rows})
    .execute()
)
"""

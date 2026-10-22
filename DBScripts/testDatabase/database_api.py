from .database_login import setup_supabase_client

TEMP_CONFIG_ID = "ce483888-9231-44e2-bf95-f2a271f8d95a" #temporary hardcoded config id to ensure foreign key in training_run


### Database API ###

# temporary functions to insert results into the database
def results_to_dict(training_id, results):
    return [
        {"training_id": training_id, "epoch_no": epoch, "accuracy": accuracy}
        for epoch, accuracy in results
    ]

def simple_insert_results_to_database(user_name, seed, results, start_time, end_time):
    
    client = None
    training_id = None
    response = None
    
    try:
        client = setup_supabase_client("service_role")

        # insert training run
        response = (
            client.table("test_training_run")
            .insert({"config_id":TEMP_CONFIG_ID, "run_by": user_name, "seed": seed, "start_time": start_time, "end_time": end_time})
            .select("training_id")
            .execute()
        )

        # collect training ID
        training_id = response.data[0]["training_id"]

        # insert results
        rows = results_to_dict(training_id, results)
        response = (
            client.table("test_training_result")
            .insert(rows)
            .execute()
        )

    except Exception as e:
        print(f"Error occurred while inserting results: {e}")
        response = None

    return client, training_id, response





### Permanent functions handling database operations for training runs and results ###

# Login to the database
def database_login(user):
    client = None
    #Ensure proper login with correct permissions
    return client


# Insert a training run with related configuration details and produce a warning if the seed is already used with that configuration
def initialize_training (client, seed, config):
    training_ID = None
    # 1. Check_client
    # 2. Check_data
    # 3. Perform database operations
    # 4. Return results
    
    return training_ID

#Insert the results and update the status of the training run.
def finalize_training (client, training_ID, results):
    success = False
    
    # 1. Check_client
    # 2. Check_data
    # 3. Perform database operations
    # 4. Return results
    
    return success

#Must return a list with the final accuracy of all finalized experiments with a unique pair of configuration and seed and produce a warning if there are experiments that are not yet finalized.
def collect_final_results (client):
    results = []  
    
    # 1. Check_client
    # 2. Check_data
    # 3. Perform database operations
    # 4. Return results
    
    return results


# Return an object with a list of results from all epochs and information on the training and configuration.
def collect_training_results (client, training_ID):
    results = []
    
    # 1. Check_client
    # 2. Check_data
    # 3. Perform database operations
    # 4. Return results
    
    return results

from .database_login import setup_supabase_client
from supabase import Client
import datetime

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
        client = setup_supabase_client()

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
def initialize_training (client, seed, config, user_name):
    # 1. Check_client
    # 2. Check_data
    # 3. Perform database operations
    # 4. Return results
    if not isinstance(client, Client):
        raise TypeError("Expected a supabase client")

    response = (
        client.table("training_run")
        .insert({"config_id": config, "run_by": user_name, "seed": seed, "start_time": datetime.datetime.now()})
        .select("training_id")
        .execute()
    )
    return response.data[0]['training_id']

#Insert the results and update the status of the training run.
def finalize_training (client, training_id, results, end_time):
    success_insert = False
    success_update = False
    finalized_results = results_to_dict(training_id, results)

    if not isinstance(client, Client):
        raise TypeError("Expected a supabase client")

    try:
        response_insert = (
                    client.table("training_result")
                    .insert(finalized_results)
                    .execute()
                )

        if len(response_insert.data) == len(finalized_results):
            success_insert = True
        else:
            print("The amount of rows added to the database does not fit with the amount of rows from the training")
            print(f"Rows added: {len(response_insert)} \n Rows from training: {len(finalized_results)}")
            success_insert = False

        
        response_update = (
                    client.table("training_run")
                    .update({"end_time": end_time})
                    .eq("training_id", training_id)
                    .execute()
        )

        if len(response_update.data) == 1:
            if  response_update.data[0]["training_id"] == training_id and response_update.data[0]["end_time"] == end_time:
                success_update = True
            else:
                print("The training ID and the end time does not match the row that got changed in the database")
                success_update = False
        else:
            print(f"{response_update.count} rows got updated when only 1 should")
            success_update = False

    except Exception as e:
        print(f"Error occurred while inserting results: {e}")
        success_update = False
        success_insert = False
    
    if success_update and success_insert:
        return True
    else:
        return False

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

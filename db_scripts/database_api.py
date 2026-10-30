from datetime import datetime, timezone
from db_scripts.validation import validate_integer, validate_client, validate_results, validate_string, validate_response

### Database API ###

# temporary functions to insert results into the database
def results_to_dict(training_id, results):
    return [
        {"training_id": training_id, "epoch_no": epoch, "accuracy": accuracy}
        for epoch, accuracy in results
    ]

### Permanent functions handling database operations for training runs and results ###

# Login to the database
def database_login(user):
    client = None
    #Ensure proper login with correct permissions
    return client


# Insert a training run with related configuration details and produce a warning if the seed is already used with that configuration
def initialize_training (client, seed, config_id, user_name):
    validate_client(client)

    # Check that parameters are ready to be inserted into the database
    validate_integer(seed)
    validate_string(config_id, "config_id", allow_space_in_string=False)
    validate_string(user_name, "user_name",allow_space_in_string=True)

    # Insert initialisation data into the database
    try:
        response = (
            client.table("training_run")
            .insert({"config_id": config_id, "run_by": user_name, "seed": seed, "start_time": datetime.now().isoformat()})
            .select("training_id")
            .execute()
        )

    except Exception as e:
        print(f"Error occurred while initialising training run to database: {e}")

    # Validate the data inserted into the database corresponds to the data returned by the database
    validate_response(response.data[0]["config_id"], config_id, "config_id")
    validate_response(response.data[0]["run_by"], user_name, "user_name")
    validate_response(response.data[0]["seed"], seed, "seed")

    return response.data[0]['training_id']

#Insert the results and update the status of the training run.
def finalize_training (client, training_id, results, end_time):
    success_insert = False
    success_update = False

    validate_client(client)

    # Check that parameters are ready to be inserted into the database
    validate_string(training_id, "training_id", allow_space_in_string=False)
    validate_results(results)
    validate_string(end_time, "end_time", allow_space_in_string=True)

    finalized_results = results_to_dict(training_id, results)

    # Insert results to the database
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
            print(f"Rows added: {len(response_insert.data)} \n Rows from training: {len(finalized_results)}")
            success_insert = False
    except Exception as e:
        print(f"Error occurred while inserting results to database: {e}")

    # Update the end time of the training.
    try:
        response_update = (
                    client.table("training_run")
                    .update({"end_time": end_time})
                    .eq("training_id", training_id)
                    .execute()
        )

        if len(response_update.data) == 1:
            if  response_update.data[0]["training_id"] == training_id and response_update.data[0]["end_time"] == end_time.replace(tzinfo=timezone.utc):
                success_update = True
            else:
                print("The training ID and/or the end time does not match the row that got changed in the database")
                print(f"Training ID from training: {training_id}")
                print(f"Training ID from database: {response_update.data[0]["training_id"]}")
                print(f"End time from training: {end_time.replace(tzinfo=timezone.utc)}")
                print(f"End time from database: {response_update.data[0]["end_time"]}")
                success_update = False
        else:
            print(f"{len(response_update.data)} rows got updated when only 1 should")
            success_update = False
    except Exception as e:
        print(f"Error occurred while updating end time to database: {e}")
    
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

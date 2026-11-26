from dotenv import load_dotenv
import os
from supabase import create_client, Client, ClientOptions

CONVERGENCE_EXPERIMENT_NO = 3

def create_supabase_connection():
    # Load environment variables from .env
    load_dotenv()

    # Setup the connection to Supabase
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_KEY")
    supabase: Client = create_client(url, key)

    return supabase

# Query structure: Contains a list variable called "data", that is a list of dictionaries.
# The index of the list represents the row number, the key (dictionary) is the name of the column, and the value (dictionary) is the data point of the column and row
def query_training_result_experiment(supabase, experiment_no):
    if experiment_no == CONVERGENCE_EXPERIMENT_NO:
        response = (
            supabase.table("training_result_convergence")
            .select("*")
            .execute()
        )
        return response.data
    else:
        response = (
            supabase.table("training_run")
            .select("training_result(*)")
            .eq("experiment_no", experiment_no)
            .execute()
        )

        # When using a join on the supabase table it returns the data as a list of the seeds with a dictionary called 'training_result' which contains the data we want.
        # Our data analysis code expects the results to be in 1 list of dictionaries therefore this for loop is vital.
        data = []
        for i in range(len(response.data)):
            data.extend(response.data[i]['training_result'])

        return data

def query_distinct_training_epoch_no(supabase):
    response = (
        supabase.table("distinct_training_result_epoch_no")
        .select("*")
        .execute()
    )
    return response.data

def query_training_epoch_accuracies_experiment(supabase, epoch, experiment_no):
    if experiment_no == CONVERGENCE_EXPERIMENT_NO:
        response = (
            supabase.table("training_result_convergence")
            .select("accuracy")
            .eq("epoch_no", epoch)
            .execute()
        )
        return response.data
    else:
        response = (
            supabase.table("training_run")
            .select("training_result(accuracy)")
            .eq("experiment_no", experiment_no)
            .eq("training_result.epoch_no", epoch)
            .execute()
        )
    
        # When using a join on the supabase table it returns the data as a list of the seeds with a dictionary called 'training_result' which contains the data we want.
        # Our data analysis code expects the results to be in 1 list of dictionaries therefore this for loop is vital.
        data = []
        for i in range(len(response.data)):
            data.extend(response.data[i]['training_result'])

        return data


def query_training_converged_epochs(supabase, tolerance, required_consecutive_passes):
    """
    Function calculates, which calculate which epoch each seed of training_result_convergence converges.
    Outputs dictionary with seed as key, and convergence epoch as value.
    """
    if required_consecutive_passes < 1:
        exit("There must be at least one pass")

    seeds = dict()
    response = (
        supabase.table("training_result_convergence")
        .select("training_run(seed), epoch_no, validation_loss")
        .order("training_run(seed)").order("epoch_no")
        .execute()
    )

    for tup in response.data:
        seed = tup["training_run"]["seed"]
        val_loss = tup["validation_loss"]
        seeds.setdefault(seed, []).append(val_loss)

    seed_converged_epoch = dict()

    for seed in seeds.keys():
        passes = 0
        for epoch, validation_loss in enumerate(seeds.get(seed)):
            # Return None, if there are fewer epochs left, than required consecutive passes
            epochs_left = len(seeds[seed]) - (epoch+1)
            if epochs_left + passes < required_consecutive_passes:
                seed_converged_epoch[seed] = None
                break

            # If absolute difference is greater than tolerance, go to next epoch.
            # Else, check for consecutive passes.
            if abs(validation_loss - seeds.get(seed)[epoch+1]) > tolerance:
                passes = 0
                continue
            else:
                passes += 1
                if passes == required_consecutive_passes:
                    # +1 is to get start position of 0-index
                    seed_converged_epoch[seed] = epoch+1-required_consecutive_passes
                    break

    return seed_converged_epoch
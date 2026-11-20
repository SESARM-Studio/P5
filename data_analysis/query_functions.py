from dotenv import load_dotenv
import os
from supabase import create_client, Client, ClientOptions

CONVERGENCE_EXPERIMENT_NR = 3

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
def query_training_result_experiment(supabase, experiment_nr):
    if experiment_nr == CONVERGENCE_EXPERIMENT_NR:
        response = (
            supabase.table("training_result_convergence")
            .select("*")
            .execute()
        )
        return response.data
    else:
        response = (
            supabase.table("training_run")
            .select("training_result(training_id, epoch_no, accuracy)")
            .eq("experiment_nr", experiment_nr)
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

def query_training_epoch_accuracies_experiment(supabase, epoch, experiment_nr):
    if experiment_nr == CONVERGENCE_EXPERIMENT_NR:
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
            .eq("experiment_nr", experiment_nr)
            .eq("training_result.epoch_no", epoch)
            .execute()
        )
    
        # When using a join on the supabase table it returns the data as a list of the seeds with a dictionary called 'training_result' which contains the data we want.
        # Our data analysis code expects the results to be in 1 list of dictionaries therefore this for loop is vital.
        data = []
        for i in range(len(response.data)):
            data.extend(response.data[i]['training_result'])

        return data

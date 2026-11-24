from dotenv import load_dotenv
import os
from supabase import create_client, Client

EXPERIMENT_1_SEED_UPPER_BOUND = 200

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
def query_test_training_result(supabase):
    return (
        supabase.table("training_result")
        .select("*")
        .execute()
    )

def query_distinct_training_epoch_no(supabase):
    return (
        supabase.table("distinct_training_result_epoch_no")
        .select("*")
        .execute()
    )

def query_last_training_epoch_accuracies(supabase, last_epoch):
    return (
        supabase.table("training_result")
        .select("accuracy")
        .eq("epoch_no", last_epoch)
        .execute()
    )

def query_last_training_epoch_accuracies(supabase, last_epoch):
    return (
        supabase.table("training_result")
        .select("accuracy")
        .eq("epoch_no", last_epoch)
        .execute()
    )

def query_epoch_accuracies_experiment_1(supabase, epoch):
    """
    Returns training_id, seed, epoch_no, accuracy (testing) for seed 1-200 in experiment 1 for the specified epoch. 
    """

    return (
        supabase.from_("training_run")
        .select("seed, training_result(epoch_no, accuracy)")
        .eq("training_result.epoch_no", epoch)
        .eq("experiment_no", 1)
        .lt("seed", EXPERIMENT_1_SEED_UPPER_BOUND+1)
        .order("seed", desc=False)
        .execute()
    )

def query_epoch_accuracies_experiment_2_300_seeds(supabase, epoch):
    """
    Returns training_id, seed, epoch_no, accuracy (testing) for seed 201-500 in experiment 2 for the specified epoch. 
    """

    return (
        supabase.from_("training_run")
        .select("seed, training_result(epoch_no, accuracy)")
        .eq("training_result.epoch_no", epoch)
        .eq("experiment_no", 2)
        .gt("seed", EXPERIMENT_1_SEED_UPPER_BOUND)
        .order("seed", desc=False)
        .execute()
    )


def query_epoch_accuracies_experiment_2_500_seeds(supabase, epoch):
    """
    Returns training_id, seed, epoch_no, accuracy (testing) for seed 1-500 in experiment 2 for the specified epoch. 
    """

    return (
        supabase.from_("training_run")
        .select("seed, training_result(epoch_no, accuracy)")
        .eq("training_result.epoch_no", epoch)
        .eq("experiment_no", 2)
        .order("seed", desc=False)
        .execute()
    )

def query_epoch_accuracies_experiment_3(supabase, epoch):
    """
    Returns training_id, seed, epoch_no, validation accuracy for seed 1-200 in experiment 3 for the specified epoch. 
    """

    return (
        supabase.from_("training_run")
        .select("seed, training_result_convergence(epoch_no, validation_accuracy)")
        .eq("training_result_convergence.epoch_no", epoch)
        .eq("experiment_no", 3)
        .order("seed", desc=False)
        .execute()
    )

# def query_selection_test(supabase):
#     """
#     Returns training_id, seed, epoch_no, accuracy (testing) for seed 1-200 in experiment 1 for the specified epoch. 
#     """

#     return (
#         supabase.from_("training_run")
#         .select("seed, training_result(epoch_no, accuracy)")
#         .eq("training_result.epoch_no", 32)
#         .in_("seed", [1,2,3,7])
#         .lt("seed", EXPERIMENT_1_SEED_UPPER_BOUND+1)
#         .order("seed", desc=False)
#         .execute()
#     )
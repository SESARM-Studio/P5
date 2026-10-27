from dotenv import load_dotenv
import os
from supabase import create_client, Client, ClientOptions

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
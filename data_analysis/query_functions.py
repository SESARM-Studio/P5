from dotenv import load_dotenv
import os
from supabase import create_client, Client, ClientOptions
import matplotlib.pyplot as plt
import numpy as np

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
        supabase.table("test_training_result")
        .select("*")
        .execute()
    )

def query_distinct_training_id(supabase):
    return (
        supabase.table("distinct_test_training_result_id")
        .select("*")
        .execute()
    )

def query_distinct_training_epoch_no(supabase):
    return (
        supabase.table("distinct_test_training_result_epoch_no")
        .select("*")
        .execute()
    )
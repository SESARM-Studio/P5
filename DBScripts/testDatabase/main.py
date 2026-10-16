import os
from supabase import create_client, Client

from helper_func import *
from create_client import setup_supabase_client
from database_func import insert_results_to_database


### initialize results to test database ###
results = generate_results_array(12)
rows = results_to_dict(results)

### Setup the supabase client and insert results into the database ###
email = os.environ.get("APP_PROGRAMMER_EMAIL")
password = os.environ.get("APP_PROGRAMMER_PASSWORD")
#email = os.environ.get("SOPHISTICATED_USER_EMAIL")
#password = os.environ.get("SOPHISTICATED_USER_PASSWORD")
client = setup_supabase_client(email, password)

insert_results_to_database(client, results)


"""
### Access data from the database using the supabase client with authentication to CRUD###
SUPABASE_URL="https://heoxipczkyjboaopsydp.supabase.co"
SUPABASE_KEY="sb_publishable_V-rH_i8IWm0uYxU5TKao_g_tI3AlWRu"
client = create_client(SUPABASE_URL, SUPABASE_KEY)

"""
import os
from dotenv import load_dotenv

# Load variables from .env file into environment
load_dotenv()
url = os.getenv("SUPABASE_URL")
api_key = os.getenv("SUPABASE_PRIVATE_KEY")

from supabase import create_client


### setup the supabase client ###
def setup_supabase_client(user_type):

    try:
        client = create_client(url, api_key)
        return client

    except Exception as e:
        print(f"Error creating Supabase client: {e}")
        return None
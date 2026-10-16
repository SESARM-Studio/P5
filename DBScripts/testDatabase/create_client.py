from dotenv import load_dotenv
load_dotenv()
import os
from supabase import create_client
from supabase_auth.errors import AuthApiError


### setup the supabase client ###
def setup_supabase_client(email, password):
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    supabase = create_client(url, key)

    appprogrammer_user_email = email
    appprogrammer_user_password = password

    try:
        session = supabase.auth.sign_in_with_password({
            "email": appprogrammer_user_email,
            "password": appprogrammer_user_password
        })
        return supabase

    except AuthApiError as e:
        print(f"Error signing in: {e}")
        return None
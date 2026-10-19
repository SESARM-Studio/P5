from dotenv import load_dotenv
import time
import os
from supabase import create_client, Client
import matplotlib

# Load environment variables from .env

load_dotenv()

# Fetch variables

DATABASE_URL = os.getenv("DATABASE_URL")

# Connect to the database

connection = psycopg2.connect(DATABASE_URL)



def db_init_experiment(seed: int, no_epochs: int = 600) -> None:
  """
  - Initialises DB table with a row for each epoch for the given seed, filling in seed number and epoch number.\n
  - Apart from columns 'seed_no' and 'epoch_no', DB default values are inserted for remaining columns.
  """

  # Creates initialisation tuples for the DB insertion

  # Will contain the tuples (seed_no, epoch_no), e.g. (1, 349) for seed 1 at epoch 349 

  data_tuple = []
  i = 1
  while i < no_epochs + 1:
    data_tuple.insert(0,(seed, i))
    i = i + 1 

  # Cursor classes bind to the DB connection and is used as our endpoint to the DB connection

  with connection.cursor() as cur:
    # Creating the value tuples for insertation, e.g. (1,349), (1,350), (1,351),... 

    arg_str = ', '.join(cur.mogrify("(%s, %s)", x).decode("utf-8") for x in data_tuple)
    
    cur.execute("""INSERT INTO test1(seed_no, epoch_no) VALUES """ + arg_str)

  try:
    connection.commit()

  except Exception as e:
    print(f"Committing queries failed due to error: {e}")



def insert_experiment_data(experiment_seed: int, no_epochs: int = 600, experiment_data_accuracy: float = []):

  # Creates tuples for the DB insertion of experiment data

  # Will contain the tuples (seed_no, accuracy), e.g. (1, 33.4)

  # This is assuming the accuracies are provided in ascending epoch order, first element is accuracy at epoch 1.

  data_tuple = []
  i = 0
  while i < no_epochs:
    data_tuple.insert(0,(experiment_seed, experiment_data_accuracy[i]))
    i = i + 1

  # Cursor classes bind to the DB connection and is used as our endpoint to the DB connection

  with connection.cursor() as cur:
    # Creating the value tuples for insertation, e.g. (1,349), (1,350), (1,351),... 

    arg_str = ', '.join(cur.mogrify("(%s, %s)", x).decode("utf-8") for x in data_tuple)
    print(arg_str)

    query = sql.SQL("""UPDATE test1 AS t SET accuracy = v.accuracy 
                          FROM (VALUES """ + arg_str + """)""" + """AS v(seed_no, accuracy)""" + 
                          """WHERE t.seed_no = v.seed_no"""
                    )
    
    cur.execute(query)

  try:
    connection.commit()

  except Exception as e:
    print(f"Committing queries failed due to error: {e}")


data_accuracies = [0.166666]
i = 1
while i < 600:
  data_accuracies.append(data_accuracies[i-1] + 0.166666)
  i = i + 1

print(data_accuracies)
insert_experiment_data(1, 600, data_accuracies)




def db_init_experiment2(seed: int, no_epochs: int = 600):

  data_tuple = []
  i = 1
  while i < no_epochs + 1:
    data_tuple.insert(0,(seed, i))
    i = i + 1 

  with connection.cursor() as cur:
    cur.executemany("INSERT INTO test1(seed_no, epoch_no) VALUES (%s, %s)", data_tuple)

  try:
    connection.commit()

  except Exception as e:
    print(f"Committing queries failed due to error: {e}")

start_time = time.time()
# db_init_experiment(4, 600) # Around 0.15 seconds

# db_init_experiment2(3, 600) # Around 28 seconds

print("--- %s seconds ---" % (time.time() - start_time))
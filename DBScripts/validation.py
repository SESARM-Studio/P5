from supabase import Client

def validate_string(variable, variable_name:str, *, allow_space_in_string:bool):
    if type(variable) is not str:
        raise TypeError(f"Parameter \"{variable_name}\" expected type string but got {type(variable)}")
    if not variable.strip():
        raise ValueError(f"Parameter \"{variable_name}\" expected non-empty string")
    if allow_space_in_string:
        return True
    else:
        if " " in variable:
            raise ValueError(f"Parameter \"{variable_name}\" contains a whitespace")
        else:
            return True

def validate_results(results):
    if not results:
        raise ValueError("Expected a non-empty array for results")

    for epoch_result in results:
        if len(epoch_result) != 2:
            raise ValueError(f"Expected an array of size 2 but got {len(epoch_result)}")
        
        if type(epoch_result[0]) is not int:
            raise TypeError(f"Expected epoch to be of type int but got {type(epoch_result[0])}")
        if type(epoch_result[1]) is not float:
            raise TypeError(f" Expected accuracy to be of type float but got {type(epoch_result[1])}")
    return True

def validate_client(client):
    if not isinstance(client, Client):
        raise TypeError("Expected a supabase client")
    return True

def validate_integer(variable, variable_name:str):
    if type(variable) is not int:
        raise TypeError(f"Parameter \"{variable_name}\" expected to be of type int but got {type(variable)}")
    return True

def validate_response(response_data, inserted_data, data_name:str):
    if response_data != inserted_data: 
        print(f"{data_name} returned by database does not match the {data_name} inserted")
        print(f"Inserted {data_name}: {inserted_data} \n Response {data_name}: {response_data}")
        raise ValueError(f"The value meant to be inserted to the database does not fit the returned value from the database")
    else:
        return True
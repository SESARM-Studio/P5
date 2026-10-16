### generate 600 test results from seed ###
def generate_results_array(seed):
    results = []
    for x in range(1, 600 + 1):
        results.append([seed, x, 0.5 + x * 0.001])
    return results

        
        
#convert results to a list of dictionaries for insertion into the database
def results_to_dict(results):
    return [
        {"seed": seed, "epoch": epoch, "accuracy": accuracy}
        for seed, epoch, accuracy in results
    ]

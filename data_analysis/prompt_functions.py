def prompt_user(first_experiment_no, last_experiment_no):
    valid_input = False
    while valid_input == False:
        experiment_choice = int(input("From which experiment do you wish to retrieve data from: EX1 (1) - EX2 (2) - EX3 (3): "))
        if experiment_choice < first_experiment_no or experiment_choice > last_experiment_no:
            print("Invalid input.")
        else:
            valid_input = True
    return experiment_choice
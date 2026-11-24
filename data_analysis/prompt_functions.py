def prompt_user(first_experiment_no, last_experiment_no):
    valid_input = False
    print("From which experiment do you wish to retrieve data from?")
    while valid_input == False:
        for i in range(first_experiment_no, last_experiment_no+1):
            print(f"EX{str(i)} ({str(i)})")
        experiment_choice = int(input("Choice: "))
        if experiment_choice < first_experiment_no or experiment_choice > last_experiment_no:
            print("Invalid input.")
        else:
            valid_input = True
    return experiment_choice
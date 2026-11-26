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

def prompt_chose_accuracy():
    valid_input = False
    print("From which epoch should the accuracy by chosen from?")
    while valid_input == False:
        print(f"The last epoch (1) \nThe best epoch (2)")
        accuracy_choice = int(input("Choice: "))
        if accuracy_choice not in [1,2]:
            print("Invalid input.")
        else:
            valid_input = True
    return accuracy_choice
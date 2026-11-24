from math import sqrt

from data_analysis.query_functions import *

EQUAL = 0
LESS_THAN = -1
LARGER_THAN = 1
MIN_ELEMENTS = 2

def compare(x, y):
    if x < y:
        return LESS_THAN
    if x > y:
        return LARGER_THAN
    return EQUAL


def kendalls_tau_b(list1, list2):
    if len(list1) != len(list2):
        raise Exception("List must be equal length")

    elements = len(list1)
    if elements < MIN_ELEMENTS:
        raise Exception("Lists must contain at least 2 elements")

    # initialize accumulators
    concordant = 0
    discordant = 0
    ties_first = 0
    ties_second = 0

    # compare each unique pair of elements
    # and find concordant, discordant, and tied pairs
    # for Kendall's tau-b
    for i in range(elements - 1):
        for j in range(i + 1, elements):
            item1_order = compare(list1[i], list1[j])
            item2_order = compare(list2[i], list2[j])

            if item1_order == EQUAL and item2_order == EQUAL:
                continue

            if item1_order == EQUAL:
                ties_first += 1
            elif item2_order == EQUAL:
                ties_second += 1
            elif item1_order == item2_order:
                concordant += 1
            else:
                discordant += 1

    denominator = sqrt((concordant + discordant + ties_first) * (concordant + discordant + ties_second))

    if denominator == 0:
        raise Exception("Cannot devide by zero")

    return (concordant - discordant) / denominator


def kendalls_tau_b_between_specific_epochs(epoch_no_a: int, epoch_no_b: int, data_flag: int):
    """
    Returns the rank-biased overlap for all seeds between two specified epochs, epoch_no_a and epoch_no_a.\n
    p is the persistance parameter used for Rank-Biased Overlap.\n
    --------------
    data_flag = 0 // experiment 1 data\n
    data_flag = 1 // experiment 2 (300 new seeds) data\n
    data_flag = 2 // experiment 2 (all 500 seeds) data\n
    data_flag = 3 // experiment 3 data\n
        """
    training_result_table_name: str
    accuracy_column_name: str

    list_a_accuracies = []
    list_b_accuracies = []
    result_epoch_data_a: any
    result_epoch_data_b: any

    match data_flag:
        case 0:
            training_result_table_name = "training_result"
            accuracy_column_name = "accuracy"
            result_epoch_data_a = query_epoch_accuracies_experiment_1(supabase, epoch_no_a)
            result_epoch_data_b = query_epoch_accuracies_experiment_1(supabase, epoch_no_b)

        case 1:
            training_result_table_name = "training_result"
            accuracy_column_name = "accuracy"
            result_epoch_data_a = query_epoch_accuracies_experiment_2_300_seeds(supabase, epoch_no_a)
            result_epoch_data_b = query_epoch_accuracies_experiment_2_300_seeds(supabase, epoch_no_b)

        case 2:
            training_result_table_name = "training_result"
            accuracy_column_name = "accuracy"
            result_epoch_data_a = query_epoch_accuracies_experiment_2_500_seeds(supabase, epoch_no_a)
            result_epoch_data_b = query_epoch_accuracies_experiment_2_500_seeds(supabase, epoch_no_b)

        case 3:
            training_result_table_name = "training_result_convergence"
            accuracy_column_name = "validation_accuracy"
            result_epoch_data_a = query_epoch_accuracies_experiment_3(supabase, epoch_no_a)
            result_epoch_data_b = query_epoch_accuracies_experiment_3(supabase, epoch_no_b)




    for entry in result_epoch_data_a.data:
        # What an entry looks like
        # {'training_id': '941beba0-0036-4660-8b75-6c49f4b94936', 'seed': 200, 'training_result_table_name': [{'accuracy_column_name': 0.3811, 'epoch_no': 2, 'training_id': '941beba0-0036-4660-8b75-6c49f4b94936'}]}

        # Adding accuracies for epoch a
        list_a_accuracies.append(entry[training_result_table_name][0][accuracy_column_name])

    for entry in result_epoch_data_b.data:
        # Adding accuracies for epoch a
        list_b_accuracies.append(entry[training_result_table_name][0][accuracy_column_name])

    return kendalls_tau_b(list_a_accuracies, list_b_accuracies)


if __name__ == "__main__":
    print("run")
    assert kendalls_tau_b([1, 2, 3], [1, 2, 3]) == 1.0
    assert kendalls_tau_b([1, 2, 3], [3, 2, 1]) == -1.0
    assert kendalls_tau_b([1, 1, 2], [1, 2, 3]) > 0
    assert kendalls_tau_b([1, 2, 3], [1, 1, 2]) > 0
    print("done")
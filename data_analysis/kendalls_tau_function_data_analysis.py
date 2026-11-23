from math import sqrt

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


if __name__ == "__main__":
    print("run")
    assert kendalls_tau_b([1, 2, 3], [1, 2, 3]) == 1.0
    assert kendalls_tau_b([1, 2, 3], [3, 2, 1]) == -1.0
    assert kendalls_tau_b([1, 1, 2], [1, 2, 3]) > 0
    assert kendalls_tau_b([1, 2, 3], [1, 1, 2]) > 0
    print("done")
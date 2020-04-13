import numpy as np
import random
import time
from itertools import combinations

def process_data(filepath):

    ''' We need to set out how the inputs will be.
    First, how would we set out our subsets in the dataset. 
    Will it be randomly sized sets (perhaps based on some other factors like geography)?
    Or will it just be pairs or triplets, each with a total bandwidth cost as a constraint?

    Second, should the accuracy cost which we are max be associated with each subset, or computed dynamically for each subset?
    But even if computed dynamically, we still need an accuracy associated with each camera element or subset to do the exponentially decaying number
    I've tried to do that below.
    '''

    ''' This should return a list or array of subsets; 
    and another 1 or 2 lists or arrays of associated costs/constraints; or
    use a dictionary where the keys are the subsets with bandwidth and accuracy keys as the associated values'''

    ''' Here I've assumed that this takes the data file, and returns the total number of subsets, a set of all cameras, the list of subsets, the accuracy of the subsets and the bandwidth'''
    return total_num_subsets, set_all_cameras, subsets, subsets_acc, subsets_bandwidth


'''
100

1,2,3,4,5...100

Subset Bandwidth
1,2,3  100
2,3,4  200


Camera Acc
1       0.5
2       0.7
....

'''

# compute_accuracy is just a function that takes the solution we are considering, iterates through every camera (selected_camera)
# and then iterates through every other camera in the set (using the set.difference function)
# and applies the exp function to compute the accuracy
# but we need to know what k stands for
# solution is actually colelction of subsets
# subset is collection of cameras
# and we need an accuracy number for each solution or subset
def compute_subset_accuracy(solution/subset):

    # note: solution is a set of subsets of camera candidate elements
    computed_accuracy = 0
    sol = solution.copy()
    for selected_camera in subsets of sol:
        # set of all cameras except selected
        # collection of subsets without that camera
        sol_ = sol.difference(selected_camera)
        for camera in sol_:
            # I'm assuming a dictionary with a accuracy key
            # what is k?
            accuracy = (camera.accuracy - selected_camera.accuracy)(1 - np.exp(k))

    return subset_computed_accuracy

# This function checks if we have all the cameras covered
# It does require a separate list of all cameras
# I've also assume that we have a sets of subsets of cameras
# So the union is just to get a list of all unique cameras in the solution
# which could be also done with a unique function if necc
def check_camera_coverage(solution, set_all_cameras):

    # set_all_cameras is just a set of all cameras
    # we can also treat each camera as an attribute of the set; which we will need for path relinking
    if set.union(solution) == set_all_cameras:
        return True
    else:
        return False

# This is the core part for doing the RCL and returning the solution
# We set up an empty solution
# We also set up an solution_camera_atributes which is actually the set of subsets
# candidates should hold all the possible candidate subsets of cameras

def greedy_randomized_adaptive_selection(alpha, total_num_subsets, set_all_cameras):

    # set of subsets in solution
    solution = set()
    solution_camera_attributes = set() # this is actually a set of subsets or set of cameras
    candidates = set(total_num_subsets) # this is actually a set of all the subset that we have by index

    # we need to determine the range of the RCL with the alpha and some metric
    # for example, we can use the ratio of accuracy/bandwidth which is what I have done here
    # or we can just directly use accuracy
    # this controls the range of candidates that can go into the RCL

    # While we have not covered all cameras, run the parts below
    while not check_camera_coverage(solution, set_all_cameras):
        # add to RCL while not all cameras not covered yet
        ratio = dict()

        # Calculate the ratio for every subset in the candidate set
        # Basically we run through every subset possible (i.e. each subset is a candidate)
        # See if this subset is already in the solution_camera_attributes
        # if it is, then we compute a ratio
        # This ratio is used to determine the GRASP alpha
        for i in candidate:
            # basically we want to check through whether the subset has the camera or not
            attribute_to_add = len(subsets[i] - solution_camera_attributes)

            if attribute_to_add > 0:
                # we need to see whether this greedy metric works
                ratio[i] = subsets_acc[i] / subsets_bandwidth[i]
                # ratio[i] = subsets_acc[i] /total accuracy

        c_min = min(ratio.values())
        c_max = max(ratio.values())

        # we basically add all the indices of subsets that can go in RCL
        # candidates = subsets; elements/attributes = cameras; solutions = collection of subsets
        # greedy if alpha = 0; totally random if alpha = 1

        # each candidate is a subset
        RCL = [i for i in candidates if i in ratio.keys() and ratio[i] <= c_min + alpha * (c_max - c_min)]

        # randomly pick a subset from the RCL; note we are picking a subset to add to solution, not a camera element
        selected_index = random.choice(RCL)

        # take out that specific subset's index from the set of candidates (subset)
        candidates -= {selected_index}
        # add the index of the subset to the solution
        solution.add(selected_index)
        # add the subset to the solution_camera_attributes list
        solution_camera_attributes.update(subsets[selected_index])

        # technically, we need an adaptive update of the candidates list's accuracy, based on the accuracy formula
        # if we want that to be dynamic

        return solution
    

# Not used. Kept in case we need it later
# def check_redundancy(solution):
#     # iterate through the subsets and see if after removing it, we can still get a complete solution
#     # unclear if we need it
#     for i in solution:
#         solution_ = solution.copy()
#         solution_.remove(i)
#         if check_camera_coverage(solution_):
#             solution = solution_.copy()

#     return solution

# This is a simple greedy local search
def local_search(solution):
    # we take the current solution and the unused subsets
    # iterate and swap if it leads to a better solution in terms of higher accuracy

    # Subsets not in solution
    subsets_not_in_solution = set([s for s in range(total_num_subsets) if s not in solution])
    best_accuracy = compute_accuracy(solution)
    best_solution_set = solution.copy()

    temp_solution = solution.copy()

    # s_out and s_in are sub-sets that we are considering whether to swap, based on whether we get better accuracy
    for s_out in solution:
        for s_in in subsets_not_in_solution:
            # recall we want higher accuracy so we swap in s_in if it has higher acc
            if subsets_acc[s_in] > subsets_acc[s_out]:
                # do the swap if we can get better accuracy based on simple comparison of the 2 subsets accuracy
                temp_solution.update({s_in})
                temp_solution.difference_update({i_out})

                accuracy = compute_accuracy(temp_solution)

                if accuracy > best_accuracy and check_camera_coverage(temp_solution):
                    # update the current best accuracy and solution
                    best_accuracy = accuracy
                    best_solution_set = temp_solution

    return best_solution_set

# This needs to be looked through more carefully
# not completely sure yet
# but essentially we have an initial and final, and we also take the symmetric difference, i.e. the
# subsets that are not in intersection
# Then we slowly move from initial to final by adding more subsets
def path_relinking(initial_solution, final_solution):

    # set of subsets that are only in either the initial or final solution
    subsets_not_intersection = initial_solution.symmetric_difference(final_solution)

    acc_init_sol = self.compute_accuracy(initial_solution)
    acc_final_sol = self.compute_accuracy(final_solution)

    best_acc = max(acc_init_sol, acc_final_sol)

    if acc_init_sol > acc_final_sol:
        best_solution = initial_solution
    else:
        best_solution = final_solution

    current_solution = initial_solution.copy()

    best_dif_acc = 0
    best_change_acc = best_acc

# Basically we go though all subsest that are not at intersection and progessively add them to the initial solution
    while len(subsets_not_intersection) > 0:

        for s in subsets_not_intersection:
            if s in current_solution:
                dif_acc = -subsets_acc[i]
            else:
                dif_acc = subsets_acc[i]

            current_solution.symmetric_difference_update({s})

            if (dif_acc > best_dif_acc) and check_camera_coverage(current_solution):
                best_s = s
                best_dif_acc = dif_acc

            current_solution.symmetric_difference_update({s})

        current_solution.symmetric_difference_update({best_s})

        best_change_acc += best_dif_acc

        if best_change_acc > best_acc:
            best_acc = best_change_acc
            best_solution = current_solution.copy()

        subsets_not_intersection.remove(best_s)

    return best_solution

def repair_unfeasible():
    # to be done
    # probably need a check on bandwidth
    # but then how do we repair.
    # Backtrack or local search?
    # This is checking and repairing bandwidth violation
    # Should also check coverage
    pass

# number of iterations to run
# alpha for the RCL 
# num_sample_pool allows us to control how many pairs to run path relinking on 

def GRASP(num_iterations, alpha, num_sample_pool, path_relinking_flag=True):
    best_acc = 0
    # list of solutions that we have gone through
    S = []

    while num_iterations > 0:
        iteration -= 1
        solution = greedy_randomized_algorithm(alpha, total_num_subsets)
        solution = local_search(solution)

        # we need to add check on feasibility based on bandwidth and a repair procedure here
        if not check_feasible(solution):
            solution = repair_unfeasible(solution)

        S.append(solution)
        acc = compute_accuracy(S[-1])

        if accuracy > best_acc:
            best_acc = acc
            best_sol = S[-1]

        # missing random restart

        if path_relinking_flag:

            # num_sample_pool
            pool = random.sample(range(len(S)), num_sample_pool)

            # How many pairs we do here will depend on the num of samples we take from the solution list thus far
            # elite solutions in S[]
            for s, t in combinations(pool, 2):

                s_p = path_relinking(S[s], S[t])

                acc_s_p = compute_accuracy(x_p)

                if acc_s_p > best_acc:

                    best_acc = acc_s_p
                    best_solution = s_p

    return best_acc, best_solution

from misc import vectorSchRepresentation, flatten_link_names, convert_unique_path, filter_arcs, get_total_gap
from chs import constructiveSearch
import copy
import random
import math

# random.seed(9001)
def check_overlap(elementList, otherlist):
    return [i for i in elementList if i in otherlist]

def new_solution_checker(rosters, trips_df):
    """
    To ensure that all nodes are assigned to a roster of an e-Bus
    """
    trip_nodes = set(trips_df.loc[trips_df.type == 'trip', 'ID'])
    non_trips = set(trips_df.loc[trips_df.type != 'trip', 'ID'])
    rosters = set(flatten_link_names(rosters)) - non_trips
    return trip_nodes == rosters

def early_return_task(states, trips_df, arcs, recharge_arcs, max_it=10):
    """
    instead of maximum trip roster, let other buses to complete the rest of the schedules with recharging.
    if i -> j -> k, i -> j -> 0
    """
    neighbors = copy.deepcopy(states)
    rndState = random.choice(neighbors)
    nontrips = set(trips_df.loc[trips_df.type != 'trip', 'ID'])
    cs = set(trips_df.loc[trips_df.type == "cs", "ID"])
    array_nodes = list(set(rndState) - set(nontrips))
    random_node = random.choice(array_nodes)
    rndIdx = rndState.index(random_node)
    newState  = rndState[:rndIdx] + [0] if rndIdx < 3 else rndState[:rndIdx-1] + [0]
    removed_nodes = rndState[rndIdx:-1]
    new_states = []
    changeIdxs = [neighbors.index(rndState)]
    neighbors.remove(rndState)
    for rnode in removed_nodes:
        if rnode in nontrips:
            pass
        else:
            _, maxPath, g = constructiveSearch((0, rnode), arcs, recharge_arcs, 1, [(0, rnode)], {}, cs, 0, max_it)
            path = vectorSchRepresentation([maxPath]) ## 0,10,0
            candidates = [i for i, e in enumerate(neighbors) if len(check_overlap(e, path[1:-1])) > 0] ## if there exists any overlap, return idx of the overlapped lists
            if len(candidates) > 0:
                ## get the random roster
                chosen_candidate = candidates[random.choice(candidates)]
                changeIdxs.append(neighbors.index(chosen_candidate))
                insertIdx = [i for i, e in enumerate(chosen_candidate) if e == path[2:-1]][0]
                new_states.append(chosen_candidate[:insertIdx-1] + [rnode] + chosen_candidate[insertIdx:])
            else:
                new_states.append([rnode, 0])
    if len(newState) > 1:
        new_states.append(newState)
    new_states = new_states + [state for i, state in enumerate(states) if i not in set(changeIdxs)]
    return new_states if new_solution_checker(new_states, trips_df) else states

def swap_cs_task(states, trips_df, arcs, recharge_arcs):
    neighbors = copy.deepcopy(states)
    recharging_nodes = trips_df.loc[trips_df['type'] == 'cs', :].ID.tolist()
    recharges = [state for state in neighbors if len(check_overlap(state, recharging_nodes)) > 0]
    nonrecharge = [state for state in neighbors if len(check_overlap(state, recharging_nodes)) < 1]
    new_states = nonrecharge
    for rroster in recharges:
        cs_idx = [i for i, e in enumerate(rroster) if e in recharging_nodes][0]
        recharge_nodes = list(map(lambda y: y[2], list(filter(lambda x: x[0] == rroster[cs_idx-1] and x[1] == rroster[cs_idx+1], recharge_arcs))))
        tmp_nodes = copy.deepcopy(recharge_nodes)
        new_cs = random.choice(tmp_nodes)
        new = rroster[:cs_idx] + [new_cs] + rroster[cs_idx+1:]
        new_states.append(new)
    return new_states

def find_least_schedules(rosters):
    min_val = 100
    idx_chosen = -1
    for idx, roster in enumerate(rosters):
        if len(roster) < min_val:
            min_val = len(roster)
            idx_chosen = idx
    return idx_chosen, min_val

def insert_schedules(states, trips_df, arcs, recharge_arcs):
    """
    Extract either least number of schedules or randomly chosen schedule, iterate and reassign/ insert
    the node of the chosen schedule to the rest of the unchosen schedules to check if reassignment
    would return better result.
    Return:
        neighbors: new roster of schedules for different buses
    """
    neighbors = copy.deepcopy(states)
    idx_least = find_least_schedules(neighbors)
    roster_chosen = copy.deepcopy(neighbors[idx_least[0]]) if random.choice([0, 1]) else random.choice(neighbors)
    idx_chosen = neighbors.index(roster_chosen)
    nontrips = trips_df.loc[trips_df.type != 'trip'].ID.tolist()
    unassigned_nodes = list(set(roster_chosen) - set(nontrips))
    while len(unassigned_nodes) > 0:
        node = unassigned_nodes.pop(0)
        if node in nontrips:
            pass
        else:
            idx, new_schedule, removed_node = compare_rosters(neighbors, node, trips_df, arcs, recharge_arcs, idx_chosen)
            if idx < 0: # none of the schedules produce better result
                neighbors.append([node, 0])
            else:
                neighbors.insert(idx, new_schedule[idx])
                neighbors.pop(idx+1)
                if len(removed_node) > 0:
                    unassigned_nodes += removed_node
    
    neighbors.pop(idx_chosen)
    return neighbors

def compare_rosters(rest, node, trips_df, arcs, recharge_arcs, idx_chosen):
    """
    Return the best new assigned roster among rosters from the reassigned node.
    Return:
        changed_idx: index integer of the reassigned roster
        best_soln: new roster from the comparison between previous roster against new roster (0,1,0) <=> (0,1,2,0)
        unassigned_node: iff there exists a replacement from previous roster against new roster (0,1,2,0) <=> (0,1,3,0) => [2]
    """
    best_soln = copy.deepcopy(rest)
    changed_idx = -1
    unassigned_node = []
    nontrips = trips_df.loc[trips_df.type != 'trip'].ID.tolist()
    for idx, roster in enumerate(rest):
        if idx == idx_chosen:
            pass
        else:
            removed_node, ret_roster = insertion(node, roster, trips_df, arcs, recharge_arcs)
            set_roster = set(ret_roster) - set(nontrips)
            prev_roster = set(best_soln[idx]) - set(nontrips)
            if len(prev_roster) < len(set_roster):
                best_soln[idx] = ret_roster
                changed_idx = idx
                if len(removed_node) > 0:
                    unassigned_node += removed_node
    return changed_idx, best_soln, unassigned_node

def insertion(node, roster, trips_df, arcs, recharge_arcs, max_it=10):
    """
    Extract the candidate roster with the removed node to check if they are the best
    Return:
        removed_sch: a list of unassigned nodes
        soln: a new roster from the memo
    """
    removed_sch = []
    soln = roster    
    nontrips = trips_df.loc[trips_df.type != 'trip'].ID.tolist()
    rechargenodes = trips_df.loc[trips_df.type == 'cs'].ID.tolist()
    trips_sort = trips_df.loc[trips_df.type == 'trip'].sort_values('dep_time').ID.tolist()
    nodes = [0] + [num for num in trips_sort if num in list(set(roster + [node]) - set(nontrips))]
    new_arcs = filter_arcs(arcs, nodes, rechargenodes)
    new_recharges = filter_arcs(recharge_arcs, nodes, rechargenodes)
    dur, new_path, temp_g = constructiveSearch((0,nodes[1]), new_arcs, new_recharges, 1, [(0,nodes[1])], {}, set(rechargenodes), 0, max_it)
    if not (node, 0) in temp_g:
        pass
    else:
        new_dur, new_path = dur, new_path
        new = set(convert_unique_path(new_path)) - set(rechargenodes) ## measuring new completed nodes
        # print(f"new = {new}")
        prev = set(roster) - set(nontrips)
        replaced = prev - new
        # print(f"replace = {replaced}")
        if len(replaced) == 0: ## all prev schedules are assigned
            soln = vectorSchRepresentation([new_path])[0]
            # print(f"solumn = {soln}")
        else:
            if len(prev) == len(new):
                new_bus = random.choice([0, 1])
                if new_bus:
                    soln = new
                else:
                    pass
            else:
                removed_sch += list(replaced)
                soln = vectorSchRepresentation([new_path])[0]
    return removed_sch, soln


def swap_recharge_task(states, trips_df, arcs, recharge_arcs):
    """
    exclusively for schedule with recharging: to remove recharging and allowing other schedules to complete possible schedule.
    if h -> i -> r -> j, remove r, h->i->0, and return back to the previous schedule and depot.
    """
    neighbors = copy.deepcopy(states)
    recharging_nodes = trips_df.loc[trips_df['type'] == 'cs', :].ID.tolist()
    trip_nodes = set(trips_df.loc[trips_df['type'] == 'trip', :].ID.tolist())
    rechargeRosters = {i: state for i, state in enumerate(neighbors) if len(check_overlap(state, recharging_nodes)) > 0}
#     print(list(rechargeRosters.keys()))
    if len(list(rechargeRosters.keys())):
        random_idx = random.choice(list(rechargeRosters.keys())) #list(rechargeRosters.keys())[1] #
    new_states = []; changeIdxs = []
    ## Removing charging station tour Destroy operator
    for idx, rroster in rechargeRosters.items():
        if idx == random_idx:
            cs_idx = [i for i, e in enumerate(rroster) if e in recharging_nodes][0]
            new = rroster[:cs_idx] + [0]
            removed_schs = rroster[cs_idx+1:]
            idxs = {k:idx for k in removed_schs}
            neighbors.pop(idx)
            neighbors.insert(idx, new)
    #             print(f"new = {new}... removed_schs = {removed_schs}...")
            neighbors = generate_new_solution(new, removed_schs, neighbors, arcs, recharge_arcs, idxs, trip_nodes, recharging_nodes)
    new_states = neighbors
    return new_states if new_solution_checker(new_states, trips_df) else states

def compare_tasks(remove_node, rest, arcs, recharge_arcs, idx, cs, max_it=10):
    """
    Finding the least cost swap schedule among roster (bus schedules) 
    """
    candidate_schedules = []
    idx_sch = -1
    new_rest = {k:v for k,v in enumerate(rest)}
    start_nodes = [pair[0] for pair in arcs.keys() if pair[1] == remove_node and pair[0] != 0] ## possibly starting nodes to removedNode   
    selection = {i: state for i, state in enumerate(rest) if len(check_overlap(state, start_nodes)) > 0 and i != idx} # check the roster that has possible starting nodes to removedNode    
    t = {i: state for i, state in enumerate(rest)}
    for i, roster in selection.items():
        completed_task = [r for j, r in enumerate(rest) if j != idx and  j != i]
        len_schedule_change = len(roster)
        ## GET ALL POSSIBLE STARTING NODE WITHIN A ROSTER (ONE SCHEDULE) e.g., a=1-> 10 -> 0, 4, 5; b=2 -> 10 -> 0. return a
        starting_nodes = check_overlap(start_nodes, roster)
        min_ends = len([pair for pair in arcs.keys() if pair[0] == remove_node and pair[1] != 0])
        # If there exists multiple possible starting nodes, find the node with the max ending
        max_start = sorted([s for s in starting_nodes if len([pair for pair in arcs.keys() if pair[0] == s and pair[1] != 0]) > min_ends], reverse=True)[0] if len(starting_nodes) > 1 else starting_nodes[0]
        # Compare the proposed next node with the expected next node
        initial_next = roster[:roster.index(max_start)+1] + [pair[1] for pair in arcs.keys() if pair[0] == remove_node]
        new_arcs = filter_arcs(arcs, initial_next, cs, remove_node)
        new_recharges = filter_arcs(recharge_arcs, initial_next, cs, remove_node)
        # print(initial_next[0])
        # print(new_arcs)
        _, new_path, temp_g = constructiveSearch((0,initial_next[0]), new_arcs, new_recharges, 1, [(0,initial_next[0])], {}, set(cs), 0, max_it)
        # print(f"prev path = {roster}... new path = {new_path}...")
        path_list = vectorSchRepresentation([new_path])[0]
        cond = any(any(item in sublist for item in path_list if item != 0 and item not in cs) for sublist in completed_task)
#         if len(new_path) >= len(roster) and len_schedule_change < len(new_path) and not cond and random.choice([0, 1]): ## better or not change in previous and proposed schedule
        if len(new_path) >= len(roster) and len(new_path) >= len(candidate_schedules) and not cond and random.choice([0, 1]): ## better or not change in previous and proposed schedule

            candidate_schedules = new_path
            idx_sch = i
#             print(f"i = {i}... idx = {idx_sch}.... candidate = {new_path}...")
            len_schedule_change = len(new_path)
    if idx_sch > -1:
        new_rest[idx_sch] = vectorSchRepresentation([candidate_schedules])[0]
        return idx_sch, list(new_rest.values())
    else:
        return idx_sch, list(new_rest.values())
    
def generate_new_solution(new, removed_schs, neighbors, arcs, recharge_arcs, idxs, trip_nodes, cs):
    while len(removed_schs) > 0:
        sch = removed_schs.pop(0)
        if sch == 0:
            pass
        else:
            idx = idxs[sch]
#             print(f"removed node = {sch} removed_schs={removed_schs}... new = {new}")
            tmp = copy.deepcopy(neighbors)
            idx_sch, new_schedule = compare_tasks(sch, tmp, arcs, recharge_arcs, idx, cs)
#             print(f"idx_sch = {idx_sch}... new_schedule = {new_schedule}...")
            if idx_sch < 0:
                if [sch,0] not in neighbors:
                    neighbors.append([sch, 0]) # introduce a new single-trip task
            else:
                checker = set(flatten_link_names(new_schedule))
                if not trip_nodes - checker:
                    removed_schs = []
                    neighbors = new_schedule
                    break
                unassigned = trip_nodes-checker-set(removed_schs)
                if unassigned: # to check unassigned trip nodes.
                    unassigned -= set(removed_schs)
                    removed_schs = list(trip_nodes - checker)
                    # print(f"new unassigned = {unassigned}... removed_sch = {removed_schs}...")
                    for i in unassigned:
                        # print(f"i = {i}...")
                        idxs[i] = idx_sch
                    # print(f"new_schedule... = {new_schedule}")
                    neighbors = new_schedule
                else:
                    neighbors = new_schedule
    return neighbors


def get_neighbors(states, trips_df, arcs, recharge_arcs):
    """Returns neighbor of  your solution."""
    neighbor = copy.deepcopy(states)
    func = random.choice([0,1,2,3])
    if func == 0:
        print("swapping recharging task...")
        neighbor = swap_recharge_task(neighbor, trips_df, arcs, recharge_arcs)
    elif func == 1:
        print("swapping other CS...")
        neighbor = swap_cs_task(neighbor, trips_df, arcs, recharge_arcs)
    ### perturb for try to connect the rosters.
#     elif func == 2 :
#         neighbor = early_return_task(neighbor, trips_df, arcs, recharge_arcs)
    else:
        print("insertion...")
        neighbor = insert_schedules(neighbor, trips_df, arcs, recharge_arcs)
    return neighbor


9# def ahp(criterias):
def beneficial_normalized(col, col_max): ## higher value is desired!
    return col/col_max
def nonbeneficial_normalized(x, col_min): ## lower value is desired 
    return col_min/x

def minmax_normalized(x, col_min, col_max):
    return (x - col_min) / (col_max - col_min) if col_max != col_min else 1
    
def get_cost(states, prev_state, all_nodes, arcs, recharge_arc, solution_spaces):
    """
    Calculates cost/fitness for the solution/route.
    Cost function is defined as 
        - the number of buses dispatch and,
        - the waiting time (the time difference between the next and previous tasks 
            recharging task comes with deadhead from the terminals to the departure time 
            of the next tasks)
    """
    alpha = 0.95
    curr_nbuses = len(states) #Obj1
    prev_nbuses = len(prev_state)
    prev_buses = [len(prev_state) for prev_state in solution_spaces]
    norm_curr_buses = minmax_normalized(curr_nbuses, min(prev_buses), max(prev_buses))
    norm_prev_buses = minmax_normalized(prev_nbuses, min(prev_buses), max(prev_buses))
    if curr_nbuses == prev_nbuses:
        prev_gaps = [get_total_gap(prev, all_nodes, recharge_arc) for prev in solution_spaces]
        curr_gap = get_total_gap(states, all_nodes, recharge_arc)
        norm_curr_gap = minmax_normalized(curr_gap, min(prev_gaps), max(prev_gaps))
        return (alpha) * (norm_curr_buses) + (1-alpha) * norm_curr_gap
    else:
        return norm_curr_buses
    

from numpy.random import rand
def annealing(initial_state, trips_df, arcs, recharge_arc):
    
    """Peforms simulated annealing to find a solution"""
    initial_temp = 100
   
    alpha = 0.99
    
    current_temp = initial_temp

    # Start by initializing the current state with the initial state
    solution = initial_state
    same_solution = 0
    same_cost_diff = 0
    temp = []
    it = 0
    cost_diffs = []
    costs = []
    best_costs = []
    solution_spaces = [[], solution]
    best_cost, best_soln = get_cost(solution, solution, trips_df, arcs, recharge_arc, solution_spaces), solution
    curr_cost, curr_solution = best_cost, best_soln
    while same_cost_diff < 200 and current_temp > 0 and it < 1000:
        print(f"Iteration {it+1}...")
        neighbor = get_neighbors(curr_solution, trips_df, arcs, recharge_arc)
        print(f"proposed solution = {neighbor}")
        solution_spaces.append(neighbor)
        # Check if neighbor is best so far
        neighbor_cost = get_cost(neighbor, curr_solution, trips_df, arcs, recharge_arc, solution_spaces)
        cost_diff = neighbor_cost - best_cost
        cost_diffs.append(cost_diff)
        # print(f"Prev = {curr_solution} with cost = {curr_cost}\ndiff = {cost_diff}\ncurr_st = {neighbor} with cost {neighbor_cost}")
        it += 1
        if cost_diff < 0 or rand() < math.exp(-float(neighbor_cost - curr_cost) / float(current_temp)):
            curr_cost, curr_solution = neighbor_cost, neighbor
            if cost_diff < 0:
                best_cost, best_soln = neighbor_cost, neighbor
                same_solution = 0
                same_cost_diff = 0
        else:
            same_cost_diff+=1
            same_solution+=1
        costs.append(curr_cost)
        best_costs.append(best_cost)
        temp.append(current_temp)
        # decrement the temperature
        current_temp = current_temp*alpha
        print('-'*100)
    return best_soln, best_cost, cost_diffs, temp, it, costs, solution_spaces, best_costs



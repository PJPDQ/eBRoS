from misc import vectorSchRepresentation, flatten_link_names
max_it = 10
D_MAX = 200
def constructiveScheduler(df2, gamma_lists, phi_lists, charging_stations):
    gamma = gamma_lists.copy()
    phi = phi_lists.copy()
    schedules_tab = {}
#     df2['ID'] = range(len(df2))
    S1 = df2.loc[df2.type == 'trip', 'ID']
    sorted_trips = df2[df2['ID'].isin(S1)].sort_values('dep_time').ID.tolist()
    starting_nodes = [(0, x) for x in sorted_trips]
    schedules = []
    while len(starting_nodes) > 0:
        has_recharged = 1
        start_node = starting_nodes.pop(0)
        total_completed, maxPath, g = constructiveSearch(start_node, gamma, phi, has_recharged, [start_node], {}, charging_stations, 0, max_it)
        schedules_tab[tuple(maxPath)] = total_completed
        completed_schs = list(map(lambda x: x[1], maxPath))
        completed_schs = vectorSchRepresentation([maxPath])[0]
        starting_nodes = list(filter(lambda x: not x[1] in completed_schs, starting_nodes))
        gamma = {x: v for x,v in gamma.items() if not (x[1] in completed_schs[:-1] or x[0] in completed_schs[:-1])}
        phi = {x: v for x,v in phi.items() if not (x[1] in completed_schs[:-1] or x[0] in completed_schs[:-1])}
        schedules.append(maxPath)
    return schedules_tab, schedules

def constructiveSearch(s, gamma, phi, has_recharged, path, g, charging_stations, num_it, max_it):
    if s in g and len(set(flatten_link_names(vectorSchRepresentation([path])))) == g[s][0]:
        trips, new_path = g[s]
        return trips, new_path, g
    if s[1] == 0 or num_it > max_it:
        total_duration = sum([g[trip][0]  if len(trip) < 3 else 0 for trip in path[:-1]] + [gamma[s]['duration']])
        if num_it > max_it:
            g[s] = (total_duration, path[:-1])
            return total_duration, path, g
        curr_trips, path, g, has_recharged = rechargingTask(total_duration, path, gamma, phi, has_recharged, g, charging_stations, num_it, max_it)
        g[s] = (curr_trips, path)
        return curr_trips, path, g
    else:
        maxNTrips = -1 #max duration span
        max_path = None
        completed_trips = flatten_link_names(vectorSchRepresentation([path]))
        g[s] = (gamma[s]['duration'], path) #if len(s) < 3 else (phi[s]['duration'], path)
        temp = list(filter(lambda x: x[0] == s[1] and x[1] not in completed_trips, gamma))
        for dest in temp:
            new_duration, new_path, new_g = constructiveSearch(dest, gamma, phi, has_recharged, path+[dest], g, charging_stations, num_it+1, max_it)
            new_duration = len(set(flatten_link_names(vectorSchRepresentation([new_path]))) - set(charging_stations))
            if maxNTrips < new_duration:
                maxNTrips = new_duration
                max_path = new_path
        if max_path is None:
            max_path = []
        return maxNTrips, max_path, g

def rechargingTask(curr_trips, path, gamma, phi, has_recharged, g, charging_stations, num_it, max_it):
    if curr_trips >= D_MAX * has_recharged:
        recharge_dest = [i for i in path[:-1]][-1]
        recharges = list(filter(lambda x: x[0] == recharge_dest[0], phi)) #and x[1] == recharge_dest[1])
        if len(recharges) > 0:
            completed_trips = flatten_link_names(vectorSchRepresentation([path[:-1]]))
            feasible_next = [x for x in recharges if x[1] not in completed_trips]
            least_cost = curr_trips; best_least_recharge = (least_cost, path)
            for next_recharge in feasible_next:
                if num_it > max_it:
                    current = [x for key, x in g.items()]
                    curr_max = max(current)
                    curr_key = [key for key, val in g.items() if val == curr_max][0]
                    best_least_recharge = g[curr_key]
                if next_recharge in g:
                    best_least_recharge = g[next_recharge]
                else:
                    possible_trips = curr_trips + phi[next_recharge]['duration']
                    possible_path = path[:path.index(recharge_dest)] + [next_recharge]
                    max_path = (possible_trips, possible_path)
                    g[next_recharge] = max_path
                    max_trips = len(set(flatten_link_names(vectorSchRepresentation([possible_path]))) - set(charging_stations))
                    tmp = list(filter(lambda x: x[0] == next_recharge[1] and x[1] not in completed_trips, gamma))
                    if len(tmp)> 0:
                        for i in tmp:
                            new_trips, new_path, g = constructiveSearch(i, gamma, phi, has_recharged+1, possible_path+[i], g, charging_stations, num_it, max_it)
                            visited_schedule = len(set(flatten_link_names(vectorSchRepresentation([new_path]))) - set(charging_stations))
                            if max_trips < visited_schedule:
                                max_trips = visited_schedule
                                max_path = (new_trips, new_path)
                                g[i] = max_path
                    curr_cost = sum([g[trip][0] if len(trip) < 3 else 0 for trip in max_path[1]])
                    if least_cost < curr_cost:
                        least_cost = curr_cost
                        best_least_recharge = max_path
            curr_trips, path = best_least_recharge
        else: ## no recharge is possible so, immediate trip to DEPOT [0]
            ### Replace new path to the path to depot!
            path = path[:path.index(recharge_dest)] + [(recharge_dest[0], 0)]
            curr_trips = len(set(flatten_link_names(vectorSchRepresentation([path]))) - set(charging_stations))
    return curr_trips, path, g, has_recharged


# def constructiveScheduler(df2, gamma_lists, phi_lists, charging_stations, max_it=10, D_MAX=350):
#     gamma = gamma_lists.copy()
#     phi = phi_lists.copy()
#     schedules_tab = {} 
#     df2['ID'] = range(len(df2))
#     S1 = df2.loc[df2.type == 'trip', 'ID']
#     sorted_trips = df2[df2['ID'].isin(S1)].sort_values('dep_time').ID.tolist()
#     starting_nodes = [(0, x) for x in sorted_trips]
#     schedules = []
#     g = {}
#     while len(starting_nodes) > 0:
#         has_recharged = 1
#         start_node = starting_nodes.pop(0)
#         path = [start_node]
#         g[start_node] = (gamma[start_node]['duration'], path)
#         total_completed, maxPath, g = constructiveSearch(start_node, gamma, phi, has_recharged, path, g, charging_stations, 0, max_it, D_MAX)
#         schedules_tab[tuple(maxPath)] = total_completed
#         completed_schs = list(map(lambda x: x[1], maxPath))
#         completed_schs = vectorSchRepresentation([maxPath])[0]
#         starting_nodes = list(filter(lambda x: not x[1] in completed_schs, starting_nodes))
#         gamma = {x: v for x,v in gamma.items() if not (x[1] in completed_schs[:-1] or x[0] in completed_schs[:-1])}
#         phi = {x: v for x,v in phi.items() if not (x[1] in completed_schs[:-1] or x[0] in completed_schs[:-1])}
#         schedules.append(maxPath)
#     return schedules_tab, schedules

# def constructiveSearch(s, gamma, phi, has_recharged, path, g, charging_stations, num_it, max_it, D_MAX=350):
#     if s in g and len(set(flatten_link_names(vectorSchRepresentation([path])))) == g[s][0]:
#         trips, new_path = g[s]
#         return trips, new_path, g
#     if s[1] == 0 or num_it > max_it:
#         total_duration = sum([g[trip][0]  if len(trip) < 3 else 0 for trip in path[:-1]] + [gamma[s]['duration']])
#         if num_it > max_it:
#             g[s] = (total_duration, path[:-1])
#             return total_duration, path, g
#         curr_trips, path, g, has_recharged = rechargingTask(total_duration, path, gamma, phi, has_recharged, g, charging_stations, num_it, max_it)
#         g[s] = (curr_trips, path)
#         return curr_trips, path, g
#     else:
#         maxNTrips = -1 #max duration span
#         max_path = None
#         #max_duration = sum([g[x][0] for x in path[:-1]])
#         completed_trips = flatten_link_names(vectorSchRepresentation([path]))
#         g[s] = (gamma[s]['duration'], path)
#         temp = list(filter(lambda x: x[0] == s[1] and x[1] not in completed_trips, gamma))
#         for dest in temp:
#             new_duration, new_path, new_g = constructiveSearch(dest, gamma, phi, has_recharged, path+[dest], g, charging_stations, num_it+1, max_it)
#             newNTrips = len(set(flatten_link_names(vectorSchRepresentation([new_path]))) - set(charging_stations))
#             if maxNTrips <= newNTrips:
#                 #if new_duration < max_duration:
#                 max_duration = new_duration
#                 maxNTrips = newNTrips
#                 max_path = new_path
#             # new_duration = len(set(flatten_link_names(vectorSchRepresentation([new_path]))) - set(charging_stations))
#             # if maxNTrips < new_duration:
#             #     maxNTrips = new_duration
#             #     max_path = new_path
#         if max_path is None:
#             max_path = []
#         return maxNTrips, max_path, g

# def rechargingTask(curr_trips, path, gamma, phi, has_recharged, g, charging_stations, num_it, max_it, D_MAX=350):
#     if curr_trips >= D_MAX * has_recharged:
#         recharge_dest = [i for i in path[:-1]][-1]
#         recharges = list(filter(lambda x: x[0] == recharge_dest[0], phi)) #and x[1] == recharge_dest[1])
#         if len(recharges) > 0:
#             completed_trips = flatten_link_names(vectorSchRepresentation([path[:-1]]))
#             feasible_next = [x for x in recharges if x[1] not in completed_trips]
#             least_cost = curr_trips; best_least_recharge = (least_cost, path)
#             for next_recharge in feasible_next:
#                 if num_it > max_it:
#                     current = [x for _, x in g.items()]
#                     curr_max = max(current)
#                     curr_key = [key for key, val in g.items() if val == curr_max][0]
#                     best_least_recharge = g[curr_key]
#                 if next_recharge in g:
#                     best_least_recharge = g[next_recharge]
#                 else:
#                     possible_trips = curr_trips + phi[next_recharge]['duration']
#                     possible_path = path[:path.index(recharge_dest)] + [next_recharge]
#                     g[next_recharge] = (possible_trips, possible_path)
#                     max_trips = len(set(flatten_link_names(vectorSchRepresentation([possible_path]))) - set(charging_stations))
#                     tmp = list(filter(lambda x: x[0] == next_recharge[1] and x[1] not in completed_trips, gamma))
#                     if len(tmp)> 0:
#                         for i in tmp:
#                             new_trips, new_path, g = constructiveSearch(i, gamma, phi, has_recharged+1, possible_path+[i], g, charging_stations, num_it, max_it)
#                             visited_schedule = len(set(flatten_link_names(vectorSchRepresentation([new_path]))) - set(charging_stations))
#                             if max_trips <= visited_schedule:
#                                 if new_trips <= possible_trips:
#                                     possible_trips = new_trips
#                                     max_trips = visited_schedule
#                                     max_path = (new_trips, new_path)
#                                     g[i] = max_path
#                                 # max_trips = visited_schedule
#                                 # max_path = (new_trips, new_path)
#                                 # g[i] = max_path
#                     curr_cost = sum([g[trip][0] if len(trip) < 3 else 0 for trip in max_path[1]])
#                     if least_cost < curr_cost:
#                         least_cost = curr_cost
#                         best_least_recharge = max_path
#             curr_trips, path = best_least_recharge
#         else: ## no recharge is possible so, immediate trip to DEPOT [0]
#             ### Replace new path to the path to depot!
#             path = path[:path.index(recharge_dest)] + [(recharge_dest[0], 0)]
#             curr_trips = len(set(flatten_link_names(vectorSchRepresentation([path]))) - set(charging_stations))
#     return curr_trips, path, g, has_recharged
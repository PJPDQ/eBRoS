from itertools import chain
import pandas as pd
import time
import matplotlib.pyplot as plt
max_it = 10
from misc import vectorSchRepresentation, feasible_pairs, feasible_recharge, visualizeResult, visualizeSolution, drawBusTrips, apply_custom_shift
from chs import constructiveScheduler
from sa import annealing, get_total_gap
# cs_ids = ["CS1", "CS1a", "CS2c"]
MIN_BUSES = 4
def concat_str(x):
    return ','.join(x)

def countRecharge(x):
    trips = x.to_list()
    return len(list(filter(lambda x: x in cs_ids,trips)))

def countTrips(x):
    print(x.to_list())
    trips = x.to_list()
    return len(set(trips) - set(cs_ids))

# # EXAMPLE DATA SET, DEPARTURE AND ARRIVAL TIMES IN MINUTES
# trips = {
#     0: {'name': 'DEPOT', 'type': 'depot', 'duration': 0, 'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
#     1: {'name': 'AC1', 'type': 'trip', 'duration': 140,  'dep_time': 555, 'arr_time': 695, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
#     2: {'name': 'AC2', 'type': 'trip', 'duration': 140, 'dep_time': 675, 'arr_time': 815, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
#     3: {'name': 'AC3', 'type': 'trip', 'duration': 140,  'dep_time': 795, 'arr_time': 935, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
#     4: {'name': 'AC4', 'type': 'trip', 'duration': 140,  'dep_time': 1155, 'arr_time': 1295, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
#     5: {'name': 'AC5', 'type': 'trip', 'duration': 140,  'dep_time': 1215, 'arr_time': 1355, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
#     6: {'name': 'CA1', 'type': 'trip', 'duration': 140,  'dep_time': 570, 'arr_time': 710, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
#     7: {'name': 'CA2', 'type': 'trip', 'duration': 140,  'dep_time': 740, 'arr_time': 880, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
#     8: {'name': 'CA3', 'type': 'trip', 'duration': 140,  'dep_time': 860, 'arr_time': 1000, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
#     9: {'name': 'CA4', 'type': 'trip', 'duration': 140,  'dep_time': 920, 'arr_time': 1060, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
#     10: {'name': 'CA5', 'type': 'trip', 'duration': 140,  'dep_time': 1040, 'arr_time': 1180, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
# }
# charging_station = {
#     len(trips): {'name': 'CS1', 'type': 'cs', 'duration': 50,  'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
# }

# charging_stations1 = {
#     len(trips): {'name': 'CS1', 'type': 'cs', 'duration': 50,  'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
#     len(trips)+1: {'name': 'CS1a', 'type': 'cs', 'duration': 50,  'dep_time': 0, 'arr_time': 1440, 'dep_term': 'Terminal A', 'arr_term': 'Terminal A'},
# }

# charging_stations2 = {
#     len(trips): {'name': 'CS1', 'type': 'cs', 'duration': 50,  'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
#     len(trips)+1: {'name': 'CS1a', 'type': 'cs', 'duration': 50,  'dep_time': 0, 'arr_time': 1440, 'dep_term': 'Terminal A', 'arr_term': 'Terminal A'},
#     len(trips)+2: {'name': 'CS2c', 'type': 'cs', 'duration': 50,  'dep_time': 0, 'arr_time': 1440, 'dep_term': 'Terminal C', 'arr_term': 'Terminal C'},
# }

# # TERMINALS AND MINIMUM INTERVALS BETWEEN TRIPS
# terminals = {
#     'Terminal A': {'max_interval': 15},
#     'Terminal C': {'max_interval': 15},
# }
# ### Add extra parameter that allows e-Bus to complete schedule that is not continuous (aka the start node does not have to be the end node of the previous)
# ## Defining the deadhead parameter from Terminals
# cs_deadheads = {
#     'Terminal A': {'CS1': 30, 'CS1a': 10, 'CS2c': 75},
#     'Terminal C': {'CS1': 30, 'CS1a': 75, 'CS2c': 10},
# }

# D_MAX = 350
# CHARGING_TIME = 50
# trips_df = pd.DataFrame.from_dict(trips, orient='index')
# print(trips_df)
# ## Creating Gamma and Delta
# arcs = feasible_pairs(trips_df, terminals)
# print(charging_station)

# ######################################################## 1CS #####################################################################
# recharge_1cs_arcs = feasible_recharge(trips_df, cs_deadheads, recharge=charging_station, terminals=terminals)
# cs_ids = set(list(charging_station.keys()))
# all_schedules_1cs = {**trips, **charging_station}
# all_schs_1cs = pd.DataFrame.from_dict(all_schedules_1cs, orient='index')
# all_schs_1cs['ID'] = range(len(all_schs_1cs))
# durations_1cs = {idx: {'duration': all_schs_1cs.loc[idx, 'duration']} for idx in all_schs_1cs.index if idx != 0}


# # for i in range(3, 6):
# ################################################################### CHS-SA ###########################################################################################
# ####### CHS
# import time
# start_time = time.time()
# print(f"{time.ctime()}")
# schedules_tab, schedules_10_1 = constructiveScheduler(all_schs_1cs, arcs, recharge_1cs_arcs, cs_ids)
# solution = vectorSchRepresentation(schedules_10_1)
# end_time = time.time()
# print(f"{time.ctime()}\nTime elapse to compute the solution = {(end_time - start_time)} seconds")
# print(f"number of buses = {len(schedules_10_1)}")
# print(schedules_10_1)
# g_T10CS1_time = end_time-start_time

# ####### SA
# print('-'*100)
# print("starting simulated annealing....")
# start_time = time.time()
# # test_new_schedule, test_cost, test_cost_diffs, test_temp, test_it, test_costs, test_solutionspaces, test_best
# new_schedule_10_1, cost_10_1, cost_diffs_10_1, temp_10_1, it_10_1, costs_10_1, solution_spaces_10_1, best_costs_10_1 = annealing(solution, all_schs_1cs, arcs, recharge_1cs_arcs)

# end_time = time.time()
# T10CS1_time = (end_time-start_time) + g_T10CS1_time
# print(f"{time.ctime()}\nTime elapse to compute the solution = {(end_time - start_time)} seconds")
# print(f"prev_schedule = {solution} with number of buses = {len(solution)}... \nnext_schedule = {new_schedule_10_1} with number of buses = {len(new_schedule_10_1)}")
# print(f"Number of buses previous = {len(schedules_10_1)}... new = {len(new_schedule_10_1)} ")
# print(f"Prev: {vectorSchRepresentation(schedules_10_1)}\nNew: {new_schedule_10_1}")
# print(f"Total Gap among buses=> prev = {get_total_gap(solution, all_schs_1cs, recharge_1cs_arcs)}... new = {get_total_gap(new_schedule_10_1, all_schs_1cs, recharge_1cs_arcs)}")

# # fig1, ax1 = plt.subplots()
# plt.plot(range(it_10_1), costs_10_1, label="Current_Solution Cost")
# plt.plot(range(it_10_1), best_costs_10_1, label="Best_Solution Cost")
# # plt.gca().invert_xaxis()
# plt.xlabel("Iteration (it)")
# # plt.ylabel("[weighted_sum] #ofBuses + Total Gap Time (cost)")
# plt.ylabel("Priority Normalized #ofBuses & Next Total Gaps(cost)")
# plt.title("10Trips1ChargingStation_CHS-SA")
# plt.legend(loc="upper right")
# # plt.savefig("10Trips1CS.png")

# TenTrips_1cs_df = visualizeSolution(solution_spaces_10_1[1:], "10Trips1CSs-CHS-SA Pareto Front", all_schs_1cs, recharge_1cs_arcs)
# newdf_10_1cs = visualizeResult(new_schedule_10_1, all_schs_1cs, "CHS_10Trips-1CS", cs_deadheads)

# trips10_df_1cs = newdf_10_1cs.groupby('bus_id', group_keys=False).apply(apply_custom_shift)
# trips10_df_1cs['next_dep'] = trips10_df_1cs['next_dep'].fillna(0)
# trips10_df_1cs['difference'] = trips10_df_1cs['next_dep'] - trips10_df_1cs['arr_time']
# trips10_df_1cs['difference'] = trips10_df_1cs['difference'].apply(lambda x: 0 if x < 0 else x)
# chs_10Trips1cs_IDLE_soln = trips10_df_1cs.groupby(['bus_id']).agg(
#     trips=('trip_id', concat_str),
#     numRecharge=('trip_id',countRecharge),
#     numTrips=('trip_id', countTrips),
#     gapTime=('difference', 'sum')
# )

# ################################################################### 2CS #############################################################################################
# recharge_2cs_arcs = feasible_recharge(trips_df, cs_deadheads, recharge=charging_stations1, terminals=terminals)
# cs_ids = set(list(charging_stations1.keys()))
# all_schedules_2cs = {**trips, **charging_stations1}
# all_schs_2cs = pd.DataFrame.from_dict(all_schedules_2cs, orient='index')
# all_schs_2cs['ID'] = range(len(all_schs_2cs))
# durations_2cs = {idx: {'duration': all_schs_2cs.loc[idx, 'duration']} for idx in all_schs_2cs.index if idx != 0}

# ####### CHS
# import time
# start_time = time.time()
# print(f"{time.ctime()}")
# test_schedules_tab, schedules_10_2 = constructiveScheduler(all_schs_2cs, arcs, recharge_2cs_arcs, set(cs_ids))
# test_solution = vectorSchRepresentation(schedules_10_2)
# end_time = time.time()
# g_T10CS2_time = end_time - start_time
# print(f"{time.ctime()}\nTime elapse to compute the solution = {g_T10CS2_time} seconds")
# print(f"number of buses = {len(schedules_10_2)}")
# print(schedules_10_2)
# print('-'*100)
# print("starting simulated annealing....")
# start_time = time.time()
# new_schedule_10_2, cost_10_2, cost_diffs_10_2, temp_10_2, it_10_2, costs_10_2, solutionspaces_10_2, best_10_2 = annealing(test_solution, all_schs_2cs, arcs, recharge_2cs_arcs)
# end_time = time.time()
# T10CS2_time = (end_time-start_time)+g_T10CS2_time
# print(f"{time.ctime()}\nTime elapse to compute the solution = {(T10CS2_time)} seconds")
# print(f"prev_schedule = {test_solution} with number of buses = {len(test_solution)}... \nnext_schedule = {new_schedule_10_2} with number of buses = {len(new_schedule_10_2)}")

# plt.plot(range(it_10_2), costs_10_2, label="Current_Solution Cost")
# plt.plot(range(it_10_2), best_10_2, label="Best_Solution Cost")
# plt.xlabel("Iteration (it)")
# plt.ylabel("Priority Normalized #ofBuses & Next Total Gaps(cost)")
# plt.title("10Trips2ChargingStations_CHS-SA")
# plt.legend(loc="upper right")
# # plt.savefig("10Trips2CSs.png")

# TenTrips_2cs_df = visualizeSolution(solutionspaces_10_2[1:], "10Trips2CSs-CHS-SA Pareto Front", all_schs_2cs, recharge_2cs_arcs)
# newdf_10_2cs = visualizeResult(new_schedule_10_2, all_schs_2cs, "CHS_10Trips-2CS")
# trips10_df_2cs = newdf_10_2cs.copy(deep=True)
# trips10_df_2cs['next_dep'] = trips10_df_2cs.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
# trips10_df_2cs['difference'] = trips10_df_2cs['next_dep'] - trips10_df_2cs['arr_time']
# trips10_df_2cs['difference'] = trips10_df_2cs['difference'].apply(lambda x: 0 if x < 0 else x)
# trips10_soln_2cs = trips10_df_2cs.groupby(['bus_id'])['difference'].sum()
# # trips10_df.to_csv("10Trips2CS.csv")
# trips10_2cs = trips10_df_2cs.groupby(['bus_id']).agg({'trip_id':concat_str, 'dep_time': 'first', 'arr_time': 'last', 'dep_terminal': 'first', 'arr_terminal': 'last', 'difference': 'sum'})
# chs_10Trips2cs_IDLE_soln = trips10_df_2cs.groupby(['bus_id']).agg(
#     numRecharge=('trip_id',countRecharge),
#     numTrips=('trip_id', countTrips),
#     gapTime=('difference', 'sum')
# )

# ################################################################### 3CS #############################################################################################
# recharge_3cs_arcs = feasible_recharge(trips_df, cs_deadheads, recharge=charging_stations2, terminals=terminals)
# print(recharge_3cs_arcs)
# cs_ids = set(list(charging_stations2.keys()))
# all_schedules_3cs = {**trips, **charging_stations2}
# all_schs_3cs = pd.DataFrame.from_dict(all_schedules_3cs, orient='index')
# all_schs_3cs['ID'] = range(len(all_schs_3cs))
# durations_3cs = {idx: {'duration': all_schs_3cs.loc[idx, 'duration']} for idx in all_schs_3cs.index if idx != 0}

# # for i in range(3, 6):
# ####### CHS
# import time
# start_time = time.time()
# print(f"{time.ctime()}")
# test_schedules_tab, schedules_10_3 = constructiveScheduler(all_schs_3cs, arcs, recharge_3cs_arcs, set(cs_ids))
# test_solution = vectorSchRepresentation(schedules_10_3)
# end_time = time.time()
# g_T10CS3_time = end_time - start_time
# print(f"{time.ctime()}\nTime elapse to compute the solution = {(g_T10CS3_time)} seconds")
# print(f"number of buses = {len(schedules_10_3)}")
# print(schedules_10_3)
# ####### SA
# print('-'*100)
# print("starting simulated annealing....")
# start_time = time.time()
# new_schedule_10_3, cost_10_3, cost_diffs_10_3, temp_10_3, it_10_3, costs_10_3, solutionspaces_10_3, best_10_3 = annealing(test_solution, all_schs_3cs, arcs, recharge_3cs_arcs)
# end_time = time.time()
# T10CS3_time = (end_time-start_time)+g_T10CS3_time
# print(f"{time.ctime()}\nTime elapse to compute the solution = {(T10CS3_time)} seconds")
# print(f"prev_schedule = {test_solution} with number of buses = {len(test_solution)}... \nnext_schedule = {new_schedule_10_3} with number of buses = {len(new_schedule_10_3)}")

# fig1, ax1 = plt.subplots()
# ax1.plot(range(it_10_3), costs_10_3, label="Current_Solution Cost")
# ax1.plot(range(it_10_3), best_10_3, label="Best_Solution Cost")
# # plt.gca().invert_xaxis()
# ax1.set_xlabel("Iteration (it)")
# # plt.ylabel("[weighted_sum] #ofBuses + Total Gap Time (cost)")
# ax1.set_ylabel("Priority Normalized #ofBuses & Next Total Gaps(cost)")
# ax1.set_title("10Trips3ChargingStations_CHS-SA")
# ax1.legend(loc="upper right")
# # plt.savefig("10Trips3CSs.png")

# TenTrips_3cs_df = visualizeSolution(solutionspaces_10_3[1:], "10Trips3CSs-CHS-SA Pareto Front", all_schs_3cs, recharge_3cs_arcs)
# newdf_10_3cs = visualizeResult(new_schedule_10_3, all_schs_3cs, "CHS_10Trips-3CS", cs_deadheads)

# trips10_df_3cs = newdf_10_3cs.groupby('bus_id', group_keys=False).apply(apply_custom_shift)
# trips10_df_3cs['next_dep'] = trips10_df_3cs['next_dep'].fillna(0)
# trips10_df_3cs['difference'] = trips10_df_3cs['next_dep'] - trips10_df_3cs['arr_time']
# trips10_df_3cs['difference'] = trips10_df_3cs['difference'].apply(lambda x: 0 if x < 0 else x)
# chs_10Trips3cs_IDLE_soln = trips10_df_3cs.groupby(['bus_id']).agg(
#     trips=('trip_id', concat_str),
#     numRecharge=('trip_id',countRecharge),
#     numTrips=('trip_id', countTrips),
#     gapTime=('difference', 'sum')
# )


# # ################################################################################# 20 TRIPS ##################################################################################
# trips = {
#     0: {'name': 'DEPOT', 'type': 'depot', 'duration': 0, 'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
#     1: {'name': 'AB1', 'type': 'trip', 'duration': 135, 'dep_time': 560, 'arr_time': 695, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
#     2: {'name': 'AB2', 'type': 'trip', 'duration': 135, 'dep_time': 740, 'arr_time': 875, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
#     3: {'name': 'AB3', 'type': 'trip', 'duration': 135, 'dep_time': 830, 'arr_time': 965, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
#     4: {'name': 'AB4', 'type': 'trip', 'duration': 135, 'dep_time': 920, 'arr_time': 1055, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
#     5: {'name': 'AB5', 'type': 'trip', 'duration': 135, 'dep_time': 980, 'arr_time': 1115, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
#     6: {'name': 'AC1', 'type': 'trip', 'duration': 140, 'dep_time': 555, 'arr_time': 695, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
#     7: {'name': 'AC2', 'type': 'trip', 'duration': 140, 'dep_time': 675, 'arr_time': 815, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
#     8: {'name': 'AC3', 'type': 'trip', 'duration': 140, 'dep_time': 795, 'arr_time': 935, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
#     9: {'name': 'AC4', 'type': 'trip', 'duration': 140, 'dep_time': 1155, 'arr_time': 1295, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
#     10: {'name': 'AC5', 'type': 'trip', 'duration': 140, 'dep_time': 1215, 'arr_time': 1355, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
#     11: {'name': 'BA1', 'type': 'trip', 'duration': 135, 'dep_time': 540, 'arr_time': 675, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
#     12: {'name': 'BA2', 'type': 'trip', 'duration': 135, 'dep_time': 630, 'arr_time': 765, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
#     13: {'name': 'BA3', 'type': 'trip', 'duration': 135, 'dep_time': 720, 'arr_time': 855, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
#     14: {'name': 'BA4', 'type': 'trip', 'duration': 135, 'dep_time': 810, 'arr_time': 945, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
#     15: {'name': 'BA5', 'type': 'trip', 'duration': 135, 'dep_time': 900, 'arr_time': 1035, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
#     16: {'name': 'CA1', 'type': 'trip', 'duration': 140, 'dep_time': 570, 'arr_time': 710, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
#     17: {'name': 'CA2', 'type': 'trip', 'duration': 140, 'dep_time': 740, 'arr_time': 880, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
#     18: {'name': 'CA3', 'type': 'trip', 'duration': 140, 'dep_time': 860, 'arr_time': 1000, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
#     19: {'name': 'CA4', 'type': 'trip', 'duration': 140, 'dep_time': 920, 'arr_time': 1060, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
#     20: {'name': 'CA5', 'type': 'trip', 'duration': 140, 'dep_time': 1040, 'arr_time': 1180, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
# }

# charging_station = {
#     len(trips): {'name': 'CS1', 'type': 'cs', 'duration': 50,  'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
# }

# charging_stations1 = {
#     len(trips): {'name': 'CS1', 'type': 'cs', 'duration': 50,  'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
#     len(trips) + 1: {'name': 'CS1a', 'type': 'cs', 'duration': 50,  'dep_time': 0, 'arr_time': 1440, 'dep_term': 'Terminal A', 'arr_term': 'Terminal A'},
# }

# charging_stations2 = {
#     len(trips): {'name': 'CS1', 'type': 'cs', 'duration': 50,  'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
#     len(trips)+1: {'name': 'CS1a', 'type': 'cs', 'duration': 50,  'dep_time': 0, 'arr_time': 1440, 'dep_term': 'Terminal A', 'arr_term': 'Terminal A'},
#     len(trips)+2: {'name': 'CS2b', 'type': 'cs', 'duration': 50,  'dep_time': 0, 'arr_time': 1440, 'dep_term': 'Terminal C', 'arr_term': 'Terminal C'},
# }

# # TERMINALS AND MINIMUM INTERVALS BETWEEN TRIPS
# terminals = {
#     'Terminal A': {'max_interval': 15},
#     'Terminal B': {'max_interval': 15},
#     'Terminal C': {'max_interval': 15},
#     'Terminal D': {'max_interval': 15}
# }
# ### Add extra parameter that allows e-Bus to complete schedule that is not continuous (aka the start node does not have to be the end node of the previous)
# ## Defining the deadhead parameter from Terminals
# cs_deadheads = {
#     'Terminal A': {'CS1': 30, 'CS1a': 10, 'CS2b': 30, 'CS3c': 30, 'CS4d': 30},
#     'Terminal B': {'CS1': 30, 'CS1a': 30, 'CS2b': 10, 'CS3c': 30, 'CS4d': 30},
#     'Terminal C': {'CS1': 30, 'CS1a': 30, 'CS2b': 30, 'CS3c': 10, 'CS4d': 30},
#     'Terminal D': {'CS1': 30, 'CS1a': 30, 'CS2b': 30, 'CS3c': 10, 'CS4d': 10},
# }

# D_MAX = 350
# CHARGING_TIME = 50
# trips_df = pd.DataFrame.from_dict(trips, orient='index')
# print(trips_df)
# ## Creating Gamma and Delta
# arcs = feasible_pairs(trips_df, terminals)
# print(charging_station)

# ######################################################## 1CS #####################################################################
# recharge_1cs_arcs = feasible_recharge(trips_df, cs_deadheads, recharge=charging_station, terminals=terminals)
# cs_ids = set(list(charging_station.keys()))
# all_schedules_1cs = {**trips, **charging_station}
# all_schs_1cs = pd.DataFrame.from_dict(all_schedules_1cs, orient='index')
# all_schs_1cs['ID'] = range(len(all_schs_1cs))
# durations_1cs = {idx: {'duration': all_schs_1cs.loc[idx, 'duration']} for idx in all_schs_1cs.index if idx != 0}

# # for i in range(8, 11):
# ###################################################### CHS-SA ####################################################################
# import time
# start_time = time.time()
# print(f"{time.ctime()}")
# schedules_tab, schedules_20_1 = constructiveScheduler(all_schs_1cs, arcs, recharge_1cs_arcs, cs_ids)
# solution = vectorSchRepresentation(schedules_20_1)
# end_time = time.time()
# g_T20CS1_time = end_time - start_time
# print(f"{time.ctime()}\nTime elapse to compute the solution = {(g_T20CS1_time)} seconds")
# print(f"number of buses = {len(schedules_20_1)}")
# print(schedules_20_1)

# ####### SA
# print('-'*100)
# start_time = time.time()
# print("starting simulated annealing....")
# new_schedule_20_1, cost_20_1, cost_diffs_20_1, temp_20_1, it_20_1, costs_20_1, solution_spaces_20_1, best_costs_20_1 = annealing(solution, all_schs_1cs, arcs, recharge_1cs_arcs)
# end_time = time.time()
# T20CS1_time = (end_time-start_time) + g_T20CS1_time
# print(f"{time.ctime()}\nTime elapse to compute the solution = {(T20CS1_time)} seconds")
# print(f"prev_schedule = {solution} with number of buses = {len(solution)}... \nnext_schedule = {new_schedule_20_1} with number of buses = {len(new_schedule_20_1)}")

# print(f"Number of buses previous = {len(schedules_20_1)}... new = {len(new_schedule_20_1)} ")
# print(f"Prev: {vectorSchRepresentation(schedules_20_1)}\nNew: {new_schedule_20_1}")
# print(f"Total Gap among buses=> prev = {get_total_gap(solution, all_schs_1cs, recharge_1cs_arcs)}... new = {get_total_gap(new_schedule_20_1, all_schs_1cs, recharge_1cs_arcs)}")

# #### PLOTTING
# fig1, ax1 = plt.subplots()
# ax1.plot(range(it_20_1), costs_20_1, label="Current_Solution Cost")
# ax1.plot(range(it_20_1), best_costs_20_1, label="Best_Solution Cost")
# # plt.gca().invert_xaxis()
# ax1.set_xlabel("Iteration (it)")
# # plt.ylabel("[weighted_sum] #ofBuses + Total Gap Time (cost)")
# ax1.set_ylabel("Priority Normalized #ofBuses & Next Total Gaps(cost)")
# ax1.set_title("20Trips1ChargingStation_CHS-SA")
# ax1.legend(loc="upper right")
# # plt.savefig("20Trips1CS.png")

# TwentyTrips_1cs_df = visualizeSolution(solution_spaces_20_1[1:], "20Trips1CSs-CHS-SA Pareto Front", all_schs_1cs, recharge_1cs_arcs)
# newdf_20_1cs = visualizeResult(new_schedule_20_1, all_schs_1cs, "CHS_20Trips-1CS", cs_deadheads)

# trips20_df_1cs = newdf_20_1cs.groupby('bus_id', group_keys=False).apply(apply_custom_shift)
# trips20_df_1cs['next_dep'] = trips20_df_1cs['next_dep'].fillna(0)
# trips20_df_1cs['difference'] = trips20_df_1cs['next_dep'] - trips20_df_1cs['arr_time']
# trips20_df_1cs['difference'] = trips20_df_1cs['difference'].apply(lambda x: 0 if x < 0 else x)
# chs_20Trips1cs_IDLE_soln = trips20_df_1cs.groupby(['bus_id']).agg(
#     trips=('trip_id', concat_str),
#     numRecharge=('trip_id',countRecharge),
#     numTrips=('trip_id', countTrips),
#     gapTime=('difference', 'sum')
# )

# ################################################################### 3CS #############################################################################################
# ## Creating Phi and Delta
# recharge_3cs_arcs = feasible_recharge(trips_df, cs_deadheads, recharge=charging_stations2, terminals=terminals)
# cs_ids = set(list(charging_stations2.keys()))
# all_schedules_3cs = {**trips, **charging_stations2}
# all_schs_3cs = pd.DataFrame.from_dict(all_schedules_3cs, orient='index')
# all_schs_3cs['ID'] = range(len(all_schs_3cs))
# durations_3cs = {idx: {'duration': all_schs_3cs.loc[idx, 'duration']} for idx in all_schs_3cs.index if idx != 0}

# ####### CHS
# start_time = time.time()
# print(f"{time.ctime()}")
# test_schedules_tab, schedules_20_3 = constructiveScheduler(all_schs_3cs, arcs, recharge_3cs_arcs, set(cs_ids))
# test_solution = vectorSchRepresentation(schedules_20_3)
# end_time = time.time()
# g_T20CS3_time = end_time - start_time
# print(f"{time.ctime()}\nTime elapse to compute the solution = {(end_time - start_time)} seconds")
# print(f"number of buses = {len(schedules_20_3)}")
# print(schedules_20_3)
# print('-'*100)
# print("starting simulated annealing....")
# start_time = time.time()
# new_schedule_20_3, cost_20_3, cost_diffs_20_3, temp_20_3, it_20_3, costs_20_3, solutionspaces_20_3, best_20_3 = annealing(test_solution, all_schs_3cs, arcs, recharge_3cs_arcs)
# end_time = time.time()
# T20CS3_time = (end_time-start_time) + g_T20CS3_time
# print(f"{time.ctime()}\nTime elapse to compute the solution = {(T20CS3_time)} seconds")
# print(f"prev_schedule = {test_solution} with number of buses = {len(test_solution)}... \nnext_schedule = {new_schedule_20_3} with number of buses = {len(new_schedule_20_3)}")

# #### PLOTTING
# fig1, ax1 = plt.subplots()
# ax1.plot(range(it_20_3), costs_20_3, label="Current_Solution Cost")
# ax1.plot(range(it_20_3), best_20_3, label="Best_Solution Cost")
# # plt.gca().invert_xaxis()
# ax1.set_xlabel("Iteration (it)")
# # plt.ylabel("[weighted_sum] #ofBuses + Total Gap Time (cost)")
# ax1.set_ylabel("Priority Normalized #ofBuses & Next Total Gaps(cost)")
# ax1.set_title("20Trips3ChargingStations_CHS-SA")
# ax1.legend(loc="upper right")
# #plt.savefig("20Trips3CSs.png")
# TwentyTrips_3cs_df = visualizeSolution(solutionspaces_20_3[1:], "20Trips3CSs-CHS-SA Pareto Front", all_schs_3cs, recharge_3cs_arcs)
# newdf_20_3cs = visualizeResult(new_schedule_20_3, all_schs_3cs, "CHS_20Trips-3CS", cs_deadheads)

# trips20_df_3cs = newdf_20_3cs.groupby('bus_id', group_keys=False).apply(apply_custom_shift)
# trips20_df_3cs['next_dep'] = trips20_df_3cs['next_dep'].fillna(0)
# trips20_df_3cs['difference'] = trips20_df_3cs['next_dep'] - trips20_df_3cs['arr_time']
# trips20_df_3cs['difference'] = trips20_df_3cs['difference'].apply(lambda x: 0 if x < 0 else x)
# chs_20Trips3cs_IDLE_soln = trips20_df_3cs.groupby(['bus_id']).agg(
#     trips=('trip_id', concat_str),
#     numRecharge=('trip_id',countRecharge),
#     numTrips=('trip_id', countTrips),
#     gapTime=('difference', 'sum')
# )

# cs_ids = ["CS1", "CS1a", "CS2b"]
# # ################################################################################# 30 TRIPS ##################################################################################
# trips = {
#     0: {'name': 'DEPOT', 'type': 'depot', 'duration': 0, 'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
#     1: {'name': 'AB1', 'type': 'trip', 'duration': 135, 'dep_time': 560, 'arr_time': 695, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
#     2: {'name': 'AB2', 'type': 'trip', 'duration': 135, 'dep_time': 740, 'arr_time': 875, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
#     3: {'name': 'AB3', 'type': 'trip', 'duration': 135, 'dep_time': 830, 'arr_time': 965, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
#     4: {'name': 'AB4', 'type': 'trip', 'duration': 135, 'dep_time': 920, 'arr_time': 1055, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
#     5: {'name': 'AB5', 'type': 'trip', 'duration': 135, 'dep_time': 980, 'arr_time': 1115, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
#     6: {'name': 'AB6', 'type': 'trip', 'duration': 135, 'dep_time': 1040, 'arr_time': 1175, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
#     7: {'name': 'AB7', 'type': 'trip', 'duration': 135, 'dep_time': 1100, 'arr_time': 1235, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
#     8: {'name': 'AB8', 'type': 'trip', 'duration': 135, 'dep_time': 1190, 'arr_time': 1325, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
#     9: {'name': 'AC1', 'type': 'trip', 'duration': 140, 'dep_time': 555, 'arr_time': 695, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
#     10: {'name': 'AC2', 'type': 'trip', 'duration': 140, 'dep_time': 675, 'arr_time': 815, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
#     11: {'name': 'AC3', 'type': 'trip', 'duration': 140, 'dep_time': 795, 'arr_time': 935, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
#     12: {'name': 'AC4', 'type': 'trip', 'duration': 140, 'dep_time': 1155, 'arr_time': 1295, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
#     13: {'name': 'AC5', 'type': 'trip', 'duration': 140, 'dep_time': 1215, 'arr_time': 1355, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
#     14: {'name': 'AD1', 'type': 'trip', 'duration': 210, 'dep_time': 560, 'arr_time': 770, 'dep_term': 'Terminal A', 'arr_term': 'Terminal D'},
#     15: {'name': 'BA1', 'type': 'trip', 'duration': 135, 'dep_time': 540, 'arr_time': 675, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
#     16: {'name': 'BA2', 'type': 'trip', 'duration': 135, 'dep_time': 630, 'arr_time': 765, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
#     17: {'name': 'BA3', 'type': 'trip', 'duration': 135, 'dep_time': 720, 'arr_time': 855, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
#     18: {'name': 'BA4', 'type': 'trip', 'duration': 135, 'dep_time': 810, 'arr_time': 945, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
#     19: {'name': 'BA5', 'type': 'trip', 'duration': 135, 'dep_time': 900, 'arr_time': 1035, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
#     20: {'name': 'BA6', 'type': 'trip', 'duration': 135, 'dep_time': 990, 'arr_time': 1125, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
#     21: {'name': 'BA7', 'type': 'trip', 'duration': 135, 'dep_time': 1080, 'arr_time': 1215, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
#     22: {'name': 'BA8', 'type': 'trip', 'duration': 135, 'dep_time': 1140, 'arr_time': 1275, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
#     23: {'name': 'CA1', 'type': 'trip', 'duration': 140, 'dep_time': 570, 'arr_time': 710, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
#     24: {'name': 'CA2', 'type': 'trip', 'duration': 140, 'dep_time': 740, 'arr_time': 880, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
#     25: {'name': 'CA3', 'type': 'trip', 'duration': 140, 'dep_time': 860, 'arr_time': 1000, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
#     26: {'name': 'CA4', 'type': 'trip', 'duration': 140, 'dep_time': 920, 'arr_time': 1060, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
#     27: {'name': 'CA5', 'type': 'trip', 'duration': 140, 'dep_time': 1040, 'arr_time': 1180, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
#     28: {'name': 'CD1', 'type': 'trip', 'duration': 270, 'dep_time': 450, 'arr_time': 720, 'dep_term': 'Terminal C', 'arr_term': 'Terminal D'},
#     29: {'name': 'DA1', 'type': 'trip', 'duration': 210, 'dep_time': 860, 'arr_time': 1070, 'dep_term': 'Terminal D', 'arr_term': 'Terminal A'},
#     30: {'name': 'DC1', 'type': 'trip', 'duration': 270, 'dep_time': 810, 'arr_time': 1080, 'dep_term': 'Terminal D', 'arr_term': 'Terminal C'}
# }

# charging_station = {
#     len(trips): {'name': 'CS1', 'type': 'cs', 'duration': 50,  'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
# }

# charging_stations1 = {
#     len(trips): {'name': 'CS1', 'type': 'cs', 'duration': 50,  'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
#     len(trips) + 1: {'name': 'CS1a', 'type': 'cs', 'duration': 50,  'dep_time': 0, 'arr_time': 1440, 'dep_term': 'Terminal A', 'arr_term': 'Terminal A'},
# }

# charging_stations2 = {
#     len(trips): {'name': 'CS1', 'type': 'cs', 'duration': 50,  'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
#     len(trips)+1: {'name': 'CS1a', 'type': 'cs', 'duration': 50,  'dep_time': 0, 'arr_time': 1440, 'dep_term': 'Terminal A', 'arr_term': 'Terminal A'},
#     len(trips)+2: {'name': 'CS2b', 'type': 'cs', 'duration': 50,  'dep_time': 0, 'arr_time': 1440, 'dep_term': 'Terminal C', 'arr_term': 'Terminal C'},
# }

# # TERMINALS AND MINIMUM INTERVALS BETWEEN TRIPS
# terminals = {
#     'Terminal A': {'max_interval': 15},
#     'Terminal B': {'max_interval': 15},
#     'Terminal C': {'max_interval': 15},
#     'Terminal D': {'max_interval': 15}
# }
# ### Add extra parameter that allows e-Bus to complete schedule that is not continuous (aka the start node does not have to be the end node of the previous)
# ## Defining the deadhead parameter from Terminals
# cs_deadheads = {
#     'Terminal A': {'CS1': 30, 'CS1a': 10, 'CS2b': 30, 'CS3c': 30, 'CS4d': 30},
#     'Terminal B': {'CS1': 30, 'CS1a': 30, 'CS2b': 10, 'CS3c': 30, 'CS4d': 30},
#     'Terminal C': {'CS1': 30, 'CS1a': 30, 'CS2b': 30, 'CS3c': 10, 'CS4d': 30},
#     'Terminal D': {'CS1': 30, 'CS1a': 30, 'CS2b': 30, 'CS3c': 10, 'CS4d': 10},
# }

# D_MAX = 350
# CHARGING_TIME = 50
# trips_df = pd.DataFrame.from_dict(trips, orient='index')
# print(trips_df)
# ## Creating Gamma and Delta
# arcs = feasible_pairs(trips_df, terminals)
# print(charging_station)

# ######################################################## 1CS #####################################################################
# recharge_1cs_arcs = feasible_recharge(trips_df, cs_deadheads, recharge=charging_station, terminals=terminals)
# cs_ids = set(list(charging_station.keys()))
# all_schedules_1cs = {**trips, **charging_station}
# all_schs_1cs = pd.DataFrame.from_dict(all_schedules_1cs, orient='index')
# all_schs_1cs['ID'] = range(len(all_schs_1cs))
# durations_1cs = {idx: {'duration': all_schs_1cs.loc[idx, 'duration']} for idx in all_schs_1cs.index if idx != 0}

# # for i in range(10, 15):
# ###################################################### CHS-SA ####################################################################
# import time
# start_time = time.time()
# print(f"{time.ctime()}")
# schedules_tab, schedules_30_1 = constructiveScheduler(all_schs_1cs, arcs, recharge_1cs_arcs, cs_ids)
# solution = vectorSchRepresentation(schedules_30_1)
# end_time = time.time()
# g_T30CS1_time = end_time - start_time
# print(f"{time.ctime()}\nTime elapse to compute the solution = {(g_T30CS1_time)} seconds")
# print(f"number of buses = {len(schedules_30_1)}")
# print(schedules_30_1)

# ####### SA
# print('-'*100)
# start_time = time.time()
# print("starting simulated annealing....")
# new_schedule_30_1, cost_30_1, cost_diffs_30_1, temp_30_1, it_30_1, costs_30_1, solution_spaces_30_1, best_costs_30_1 = annealing(solution, all_schs_1cs, arcs, recharge_1cs_arcs)
# end_time = time.time()
# T30CS1_time = (end_time-start_time) + g_T30CS1_time
# print(f"{time.ctime()}\nTime elapse to compute the solution = {(T30CS1_time)} seconds")
# print(f"prev_schedule = {solution} with number of buses = {len(solution)}... \nnext_schedule = {new_schedule_30_1} with number of buses = {len(new_schedule_30_1)}")

# print(f"Number of buses previous = {len(schedules_30_1)}... new = {len(new_schedule_30_1)} ")
# print(f"Prev: {vectorSchRepresentation(schedules_30_1)}\nNew: {new_schedule_30_1}")
# print(f"Total Gap among buses=> prev = {get_total_gap(solution, all_schs_1cs, recharge_1cs_arcs)}... new = {get_total_gap(new_schedule_30_1, all_schs_1cs, recharge_1cs_arcs)}")

# #### PLOTTING
# fig1, ax1 = plt.subplots()
# ax1.plot(range(it_30_1), costs_30_1, label="Current_Solution Cost")
# ax1.plot(range(it_30_1), best_costs_30_1, label="Best_Solution Cost")
# # plt.gca().invert_xaxis()
# ax1.set_xlabel("Iteration (it)")
# # plt.ylabel("[weighted_sum] #ofBuses + Total Gap Time (cost)")
# ax1.set_ylabel("Priority Normalized #ofBuses & Next Total Gaps(cost)")
# ax1.set_title("30Trips1ChargingStation_CHS-SA")
# ax1.legend(loc="upper right")
# # plt.savefig("30Trips1CS.png")

# ThirtyTrips_1cs_df = visualizeSolution(solution_spaces_30_1[1:], "30Trips1CSs-CHS-SA Pareto Front", all_schs_1cs, recharge_1cs_arcs)
# newdf_30_1cs = visualizeResult(new_schedule_30_1, all_schs_1cs, "CHS_30Trips-1CS", cs_deadheads)

# trips30_df_1cs = newdf_30_1cs.groupby('bus_id', group_keys=False).apply(apply_custom_shift)
# trips30_df_1cs['next_dep'] = trips30_df_1cs['next_dep'].fillna(0)
# trips30_df_1cs['difference'] = trips30_df_1cs['next_dep'] - trips30_df_1cs['arr_time']
# trips30_df_1cs['difference'] = trips30_df_1cs['difference'].apply(lambda x: 0 if x < 0 else x)
# chs_30Trips1cs_IDLE_soln = trips30_df_1cs.groupby(['bus_id']).agg(
#     trips=('trip_id', concat_str),
#     numRecharge=('trip_id',countRecharge),
#     numTrips=('trip_id', countTrips),
#     gapTime=('difference', 'sum')
# )

# # ################################################################### 2CS #############################################################################################
# # ## Creating Phi and Delta
# # recharge_2cs_arcs = feasible_recharge(trips_df, cs_deadheads, recharge=charging_stations1, terminals=terminals)
# # cs_ids = set(list(charging_stations1.keys()))
# # all_schedules_2cs = {**trips, **charging_stations1}
# # all_schs_2cs = pd.DataFrame.from_dict(all_schedules_2cs, orient='index')
# # all_schs_2cs['ID'] = range(len(all_schs_2cs))
# # durations_2cs = {idx: {'duration': all_schs_2cs.loc[idx, 'duration']} for idx in all_schs_2cs.index if idx != 0}

# # ####### CHS
# # import time
# # start_time = time.time()
# # print(f"{time.ctime()}")
# # test_schedules_tab, schedules_30_2 = constructiveScheduler(all_schs_2cs, arcs, recharge_2cs_arcs, set(cs_ids))
# # test_solution = vectorSchRepresentation(schedules_30_2)
# # end_time = time.time()
# # g_T30CS2_time = end_time - start_time
# # print(f"{time.ctime()}\nTime elapse to compute the solution = {g_T30CS2_time} seconds")
# # print(f"number of buses = {len(schedules_30_2)}")
# # print(schedules_30_2)
# # print('-'*100)
# # print("starting simulated annealing....")
# # start_time = time.time()
# # # test_new_schedule, test_cost, test_cost_diffs, test_temp, test_it, test_costs, test_solutionspaces, test_best = annealing(schedules_tab_2cs, all_schs_2cs, arcs, recharge_2cs_arcs)

# # new_schedule_30_2, cost_30_2, cost_diffs_30_2, temp_30_2, it_30_2, costs_30_2, solutionspaces_30_2, best_30_2 = annealing(test_solution, all_schs_2cs, arcs, recharge_2cs_arcs)
# # end_time = time.time()
# # T30CS2_time = (end_time-start_time) + g_T30CS2_time
# # print(f"{time.ctime()}\nTime elapse to compute the solution = {(T30CS2_time)} seconds")
# # print(f"{time.ctime()}\nTime elapse to compute the solution = {(end_time - start_time)} seconds")
# # print(f"prev_schedule = {test_solution} with number of buses = {len(test_solution)}... \nnext_schedule = {new_schedule_30_2} with number of buses = {len(new_schedule_30_2)}")

# # #### PLOTTING
# # plt.plot(range(it_30_2), costs_30_2, label="Current_Solution Cost")
# # plt.plot(range(it_30_2), best_30_2, label="Best_Solution Cost")
# # # plt.gca().invert_xaxis()
# # plt.xlabel("Iteration (it)")
# # # plt.ylabel("[weighted_sum] #ofBuses + Total Gap Time (cost)")
# # plt.ylabel("Priority Normalized #ofBuses & Next Total Gaps(cost)")
# # plt.title("30Trips2ChargingStations_CHS-SA")
# # plt.legend(loc="upper right")
# # plt.savefig("30Trips2CSs.png")

# # ThirtyTrips_2cs_df = visualizeSolution(solutionspaces_30_2[1:], "30Trips2CSs-CHS-SA Pareto Front", all_schs_2cs, recharge_2cs_arcs)
# # newdf_30_2cs = visualizeResult(new_schedule_30_2, all_schs_2cs, "CHS_30Trips-2CS")
# # trips30_df_2cs = newdf_30_2cs.copy(deep=True)
# # trips30_df_2cs['next_dep'] = trips30_df_2cs.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
# # trips30_df_2cs['difference'] = trips30_df_2cs['next_dep'] - trips30_df_2cs['arr_time']
# # trips30_df_2cs['difference'] = trips30_df_2cs['difference'].apply(lambda x: 0 if x < 0 else x)
# # trips30_soln_2cs = trips30_df_2cs.groupby(['bus_id'])['difference'].sum()

# # # trips30_df.to_csv("30Trips2CS.csv")
# # chs_30Trips2cs_IDLE_soln = trips30_df_2cs.groupby(['bus_id']).agg({'trip_id':concat_str, 'dep_time': 'first', 'arr_time': 'last', 'dep_terminal': 'first', 'arr_terminal': 'last', 'difference': 'sum'})
# # chs_30Trips2cs_IDLE_soln = trips30_df_2cs.groupby(['bus_id']).agg(
# #     trips=('trip_id', concat_str),
# #     numRecharge=('trip_id',countRecharge),
# #     numTrips=('trip_id', countTrips),
# #     gapTime=('difference', 'sum')
# # )
# ################################################################### 3CS #############################################################################################
# ## Creating Phi and Delta
# recharge_3cs_arcs = feasible_recharge(trips_df, cs_deadheads, recharge=charging_stations2, terminals=terminals)
# cs_ids = set(list(charging_stations2.keys()))
# all_schedules_3cs = {**trips, **charging_stations2}
# all_schs_3cs = pd.DataFrame.from_dict(all_schedules_3cs, orient='index')
# all_schs_3cs['ID'] = range(len(all_schs_3cs))
# durations_3cs = {idx: {'duration': all_schs_3cs.loc[idx, 'duration']} for idx in all_schs_3cs.index if idx != 0}

# ####### CHS
# start_time = time.time()
# print(f"{time.ctime()}")
# test_schedules_tab, schedules_30_3 = constructiveScheduler(all_schs_3cs, arcs, recharge_3cs_arcs, set(cs_ids))
# test_solution = vectorSchRepresentation(schedules_30_3)
# end_time = time.time()
# g_T30CS3_time = end_time - start_time
# print(f"{time.ctime()}\nTime elapse to compute the solution = {(end_time - start_time)} seconds")
# print(f"number of buses = {len(schedules_30_3)}")
# print(schedules_30_3)
# print('-'*100)
# print("starting simulated annealing....")
# start_time = time.time()
# new_schedule_30_3, cost_30_3, cost_diffs_30_3, temp_30_3, it_30_3, costs_30_3, solutionspaces_30_3, best_30_3 = annealing(test_solution, all_schs_3cs, arcs, recharge_3cs_arcs)
# end_time = time.time()
# T30CS3_time = (end_time-start_time) + g_T30CS3_time
# print(f"{time.ctime()}\nTime elapse to compute the solution = {(T30CS3_time)} seconds")
# print(f"prev_schedule = {test_solution} with number of buses = {len(test_solution)}... \nnext_schedule = {new_schedule_30_3} with number of buses = {len(new_schedule_30_3)}")

# #### PLOTTING
# fig1, ax1 = plt.subplots()
# ax1.plot(range(it_30_3), costs_30_3, label="Current_Solution Cost")
# ax1.plot(range(it_30_3), best_30_3, label="Best_Solution Cost")
# # plt.gca().invert_xaxis()
# ax1.set_xlabel("Iteration (it)")
# # plt.ylabel("[weighted_sum] #ofBuses + Total Gap Time (cost)")
# ax1.set_ylabel("Priority Normalized #ofBuses & Next Total Gaps(cost)")
# ax1.set_title("30Trips3ChargingStations_CHS-SA")
# ax1.legend(loc="upper right")
# #plt.savefig("30Trips3CSs.png")
# ThirtyTrips_3cs_df = visualizeSolution(solutionspaces_30_3[1:], "30Trips3CSs-CHS-SA Pareto Front", all_schs_3cs, recharge_3cs_arcs)
# newdf_30_3cs = visualizeResult(new_schedule_30_3, all_schs_3cs, "CHS_30Trips-3CS", cs_deadheads)

# trips30_df_3cs = newdf_30_3cs.groupby('bus_id', group_keys=False).apply(apply_custom_shift)
# trips30_df_3cs['next_dep'] = trips30_df_3cs['next_dep'].fillna(0)
# trips30_df_3cs['difference'] = trips30_df_3cs['next_dep'] - trips30_df_3cs['arr_time']
# trips30_df_3cs['difference'] = trips30_df_3cs['difference'].apply(lambda x: 0 if x < 0 else x)
# chs_30Trips3cs_IDLE_soln = trips30_df_3cs.groupby(['bus_id']).agg(
#     trips=('trip_id', concat_str),
#     numRecharge=('trip_id',countRecharge),
#     numTrips=('trip_id', countTrips),
#     gapTime=('difference', 'sum')
# )

################################################################################### 40 TRIPS ###############################################################################################
trips = {
    0: {'name': 'DEPOT', 'type': 'depot', 'duration': 0, 'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
    1: {'name': 'AB1', 'type': 'trip', 'duration': 135, 'dep_time': 560, 'arr_time': 695, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
    2: {'name': 'AB2', 'type': 'trip', 'duration': 135, 'dep_time': 740, 'arr_time': 875, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
    3: {'name': 'AB3', 'type': 'trip', 'duration': 135, 'dep_time': 830, 'arr_time': 965, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
    4: {'name': 'AB4', 'type': 'trip', 'duration': 135, 'dep_time': 920, 'arr_time': 1055, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
    5: {'name': 'AC1', 'type': 'trip', 'duration': 140, 'dep_time': 555, 'arr_time': 695, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
    6: {'name': 'AC2', 'type': 'trip', 'duration': 140, 'dep_time': 675, 'arr_time': 815, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
    7: {'name': 'AC3', 'type': 'trip', 'duration': 140, 'dep_time': 795, 'arr_time': 935, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
    8: {'name': 'AC4', 'type': 'trip', 'duration': 140, 'dep_time': 1155, 'arr_time': 1295, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
    9: {'name': 'AD1', 'type': 'trip', 'duration': 140, 'dep_time': 560, 'arr_time': 700, 'dep_term': 'Terminal A', 'arr_term': 'Terminal D'},
    10: {'name': 'AD2', 'type': 'trip', 'duration': 140, 'dep_time': 630, 'arr_time': 770, 'dep_term': 'Terminal A', 'arr_term': 'Terminal D'},
    11: {'name': 'AD3', 'type': 'trip', 'duration': 135, 'dep_time': 790, 'arr_time': 925, 'dep_term': 'Terminal A', 'arr_term': 'Terminal D'},
    12: {'name': 'BA1', 'type': 'trip', 'duration': 135, 'dep_time': 540, 'arr_time': 675, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
    13: {'name': 'BA2', 'type': 'trip', 'duration': 135, 'dep_time': 630, 'arr_time': 765, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
    14: {'name': 'BA3', 'type': 'trip', 'duration': 135, 'dep_time': 720, 'arr_time': 855, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
    15: {'name': 'BA4', 'type': 'trip', 'duration': 135, 'dep_time': 810, 'arr_time': 945, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
    16: {'name': 'BA5', 'type': 'trip', 'duration': 135, 'dep_time': 900, 'arr_time': 1035, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
    17: {'name': 'BC1', 'type': 'trip', 'duration': 140, 'dep_time': 435, 'arr_time': 570, 'dep_term': 'Terminal B', 'arr_term': 'Terminal C'},
    18: {'name': 'BC2', 'type': 'trip', 'duration': 140, 'dep_time': 595, 'arr_time': 735, 'dep_term': 'Terminal B', 'arr_term': 'Terminal C'},
    19: {'name': 'BC3', 'type': 'trip', 'duration': 120, 'dep_time': 800, 'arr_time': 920, 'dep_term': 'Terminal B', 'arr_term': 'Terminal C'},
    20: {'name': 'BC4', 'type': 'trip', 'duration': 130, 'dep_time': 950, 'arr_time': 1130, 'dep_term': 'Terminal B', 'arr_term': 'Terminal C'},
    21: {'name': 'CA1', 'type': 'trip', 'duration': 140, 'dep_time': 570, 'arr_time': 710, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
    22: {'name': 'CA2', 'type': 'trip', 'duration': 140, 'dep_time': 740, 'arr_time': 880, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
    23: {'name': 'CA3', 'type': 'trip', 'duration': 140, 'dep_time': 860, 'arr_time': 1000, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
    24: {'name': 'CA4', 'type': 'trip', 'duration': 140, 'dep_time': 920, 'arr_time': 1060, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
    25: {'name': 'CB1', 'type': 'trip', 'duration': 120, 'dep_time': 300, 'arr_time': 420, 'dep_term': 'Terminal C', 'arr_term': 'Terminal B'},
    26: {'name': 'CB2', 'type': 'trip', 'duration': 140, 'dep_time': 400, 'arr_time': 680, 'dep_term': 'Terminal C', 'arr_term': 'Terminal B'},
    27: {'name': 'CB3', 'type': 'trip', 'duration': 140, 'dep_time': 645, 'arr_time': 785, 'dep_term': 'Terminal C', 'arr_term': 'Terminal B'},
    28: {'name': 'CB4', 'type': 'trip', 'duration': 120, 'dep_time': 900, 'arr_time': 1020, 'dep_term': 'Terminal C', 'arr_term': 'Terminal B'},
    29: {'name': 'CD1', 'type': 'trip', 'duration': 140, 'dep_time': 450, 'arr_time': 585, 'dep_term': 'Terminal C', 'arr_term': 'Terminal D'},
    30: {'name': 'CD2', 'type': 'trip', 'duration': 135, 'dep_time': 550, 'arr_time': 685, 'dep_term': 'Terminal C', 'arr_term': 'Terminal D'},
    31: {'name': 'CD3', 'type': 'trip', 'duration': 135, 'dep_time': 600, 'arr_time': 735, 'dep_term': 'Terminal C', 'arr_term': 'Terminal D'},
    32: {'name': 'CD4', 'type': 'trip', 'duration': 135, 'dep_time': 765, 'arr_time': 900, 'dep_term': 'Terminal C', 'arr_term': 'Terminal D'},
    33: {'name': 'DA1', 'type': 'trip', 'duration': 140, 'dep_time': 720, 'arr_time': 860, 'dep_term': 'Terminal D', 'arr_term': 'Terminal A'},
    34: {'name': 'DA2', 'type': 'trip', 'duration': 135, 'dep_time': 750, 'arr_time': 885, 'dep_term': 'Terminal D', 'arr_term': 'Terminal A'},
    35: {'name': 'DA3', 'type': 'trip', 'duration': 140, 'dep_time': 800, 'arr_time': 940, 'dep_term': 'Terminal D', 'arr_term': 'Terminal A'},
    36: {'name': 'DA4', 'type': 'trip', 'duration': 140, 'dep_time': 1000, 'arr_time': 1140, 'dep_term': 'Terminal D', 'arr_term': 'Terminal A'},
    37: {'name': 'DC1', 'type': 'trip', 'duration': 140, 'dep_time': 600, 'arr_time': 740, 'dep_term': 'Terminal D', 'arr_term': 'Terminal C'},
    38: {'name': 'DC2', 'type': 'trip', 'duration': 120, 'dep_time': 765, 'arr_time': 885, 'dep_term': 'Terminal D', 'arr_term': 'Terminal C'},
    39: {'name': 'DC3', 'type': 'trip', 'duration': 120, 'dep_time': 780, 'arr_time': 900, 'dep_term': 'Terminal D', 'arr_term': 'Terminal C'},
    40: {'name': 'DC4', 'type': 'trip', 'duration': 135, 'dep_time': 965, 'arr_time': 1100, 'dep_term': 'Terminal D', 'arr_term': 'Terminal C'},
}

charging_station = {
    len(trips): {'name': 'CS1', 'type': 'cs', 'duration': 50,  'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
}

charging_stations1 = {
    len(trips): {'name': 'CS1', 'type': 'cs', 'duration': 50,  'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
    len(trips)+1: {'name': 'CS1a', 'type': 'cs', 'duration': 50,  'dep_time': 0, 'arr_time': 1440, 'dep_term': 'Terminal A', 'arr_term': 'Terminal A'},
}

charging_stations2 = {
    len(trips): {'name': 'CS1', 'type': 'cs', 'duration': 50,  'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
    len(trips)+1: {'name': 'CS1a', 'type': 'cs', 'duration': 50,  'dep_time': 0, 'arr_time': 1440, 'dep_term': 'Terminal A', 'arr_term': 'Terminal A'},
    len(trips)+2: {'name': 'CS2b', 'type': 'cs', 'duration': 50,  'dep_time': 0, 'arr_time': 1440, 'dep_term': 'Terminal B', 'arr_term': 'Terminal B'},
}

# TERMINALS AND MINIMUM INTERVALS BETWEEN TRIPS
terminals = {
    'Terminal A': {'max_interval': 15},
    'Terminal B': {'max_interval': 15},
    'Terminal C': {'max_interval': 15},
    'Terminal D': {'max_interval': 15}
}
### Add extra parameter that allows e-Bus to complete schedule that is not continuous (aka the start node does not have to be the end node of the previous)
## Defining the deadhead parameter from Terminals
cs_deadheads = {
    'Terminal A': {'CS1': 40, 'CS1a': 10, 'CS2b': 75, 'CS3c': 75, 'CS4d': 75},
    'Terminal B': {'CS1': 40, 'CS1a': 75, 'CS2b': 10, 'CS3c': 75, 'CS4d': 75},
    'Terminal C': {'CS1': 40, 'CS1a': 75, 'CS2b': 75, 'CS3c': 10, 'CS4d': 75},
    'Terminal D': {'CS1': 40, 'CS1a': 75, 'CS2b': 75, 'CS3c': 75, 'CS4d': 10},
}

D_MAX = 350
CHARGING_TIME = 50
trips_df = pd.DataFrame.from_dict(trips, orient='index')
print(trips_df)
## Creating Gamma and Delta
arcs = feasible_pairs(trips_df, terminals)
print(charging_station)

######################################################## 1CS #####################################################################
recharge_1cs_arcs = feasible_recharge(trips_df, cs_deadheads, recharge=charging_station, terminals=terminals)
cs_ids = set(list(charging_station.keys()))
all_schedules_1cs = {**trips, **charging_station}
all_schs_1cs = pd.DataFrame.from_dict(all_schedules_1cs, orient='index')
all_schs_1cs['ID'] = range(len(all_schs_1cs))
durations_1cs = {idx: {'duration': all_schs_1cs.loc[idx, 'duration']} for idx in all_schs_1cs.index if idx != 0}
###################################################### CHS-SA ####################################################################

start_time = time.time()
print(f"{time.ctime()}")
schedules_tab, schedules_40_1 = constructiveScheduler(all_schs_1cs, arcs, recharge_1cs_arcs, cs_ids)
solution = vectorSchRepresentation(schedules_40_1)
end_time = time.time()
g_T40CS1_time = end_time - start_time
print(f"{time.ctime()}\nTime elapse to compute the solution = {(g_T40CS1_time)} seconds")
print(f"number of buses = {len(schedules_40_1)}")
print(schedules_40_1)
print('-'*100)
start_time = time.time()
print("starting simulated annealing....")
new_schedule_40_1, cost_40_1, cost_diffs_40_1, temp_40_1, it_40_1, costs_40_1, solution_spaces_40_1, best_costs_40_1 = annealing(solution, all_schs_1cs, arcs, recharge_1cs_arcs)
end_time = time.time()
T40CS1_time = (end_time-start_time)+g_T40CS1_time
print(f"{time.ctime()}\nTime elapse to compute the solution = {(T40CS1_time)} seconds")
print(f"prev_schedule = {solution} with number of buses = {len(solution)}... \nnext_schedule = {new_schedule_40_1} with number of buses = {len(new_schedule_40_1)}")

fig1, ax1 = plt.subplots()
ax1.plot(range(it_40_1), costs_40_1, label="Current_Solution Cost")
ax1.plot(range(it_40_1), best_costs_40_1, label="Best_Solution Cost")
# plt.gca().invert_xaxis()
ax1.set_xlabel("Iteration (it)")
# plt.ylabel("[weighted_sum] #ofBuses + Total Gap Time (cost)")
ax1.set_ylabel("Priority Normalized #ofBuses & Next Total Gaps(cost)")
ax1.set_title("40Trips1ChargingStation_CHS-SA")
ax1.legend(loc="upper right")
#plt.savefig("40Trips1CS.png")

FortyTrips_1cs_df = visualizeSolution(solution_spaces_40_1[1:], "40Trips1CSs-CHS-SA Pareto Front", all_schs_1cs, recharge_1cs_arcs)
newdf_40_1cs = visualizeResult(new_schedule_40_1, all_schs_1cs, "CHS_40Trips-1CS", cs_deadheads)

trips40_1_df = newdf_40_1cs.groupby('bus_id', group_keys=False).apply(apply_custom_shift)
trips40_1_df['next_dep'] = trips40_1_df['next_dep'].fillna(0)
trips40_1_df['difference'] = trips40_1_df['next_dep'] - trips40_1_df['arr_time']
trips40_1_df['difference'] = trips40_1_df['difference'].apply(lambda x: 0 if x < 0 else x)
chs_40Trips1cs_IDLE_soln = trips40_1_df.groupby(['bus_id']).agg(
    trips=('trip_id', concat_str),
    numRecharge=('trip_id',countRecharge),
    numTrips=('trip_id', countTrips),
    gapTime=('difference', 'sum')
)

chs_40Trips1cs_IDLE_soln.sort_values(['gapTime'], ascending=False)
print("Ending the process for 40Trips and 1 CS......")

################################################################### 3CS #############################################################################################
## Creating Phi and Delta
recharge_3cs_arcs = feasible_recharge(trips_df, cs_deadheads, recharge=charging_stations2, terminals=terminals)
print(recharge_3cs_arcs)
cs_ids = set(list(charging_stations2.keys()))
all_schedules_3cs = {**trips, **charging_stations2}
all_schs_3cs = pd.DataFrame.from_dict(all_schedules_3cs, orient='index')
all_schs_3cs['ID'] = range(len(all_schs_3cs))
durations_3cs = {idx: {'duration': all_schs_3cs.loc[idx, 'duration']} for idx in all_schs_3cs.index if idx != 0}

###### import time
start_time = time.time()
print(f"{time.ctime()}")
test_schedules_tab, schedules_40_3 = constructiveScheduler(all_schs_3cs, arcs, recharge_3cs_arcs, set(cs_ids))
test_solution = vectorSchRepresentation(schedules_40_3)
end_time = time.time()
g_T40CS3_time = end_time - start_time
print(f"{time.ctime()}\nTime elapse to compute the solution = {g_T40CS3_time} seconds")
print(f"number of buses = {len(schedules_40_3)}")
print(schedules_40_3)
print('-'*100)
print("starting simulated annealing....")
start_time = time.time()
new_schedule_40_3, cost_40_3, cost_diffs_40_3, temp_40_3, it_40_3, costs_40_3, solutionspaces_40_3, best_40_3 = annealing(test_solution, all_schs_3cs, arcs, recharge_3cs_arcs)
end_time = time.time()
T40CS3_time = (end_time - start_time) + g_T40CS3_time
print(f"{time.ctime()}\nTime elapse to compute the solution = {(T40CS3_time)} seconds")
print(f"prev_schedule = {test_solution} with number of buses = {len(test_solution)}... \nnext_schedule = {new_schedule_40_3} with number of buses = {len(new_schedule_40_3)}")

fig1, ax1 = plt.subplots()
ax1.plot(range(it_40_3), costs_40_3, label="Current_Solution Cost")
ax1.plot(range(it_40_3), best_40_3, label="Best_Solution Cost")
# plt.gca().invert_xaxis()
ax1.set_xlabel("Iteration (it)")
# plt.ylabel("[weighted_sum] #ofBuses + Total Gap Time (cost)")
ax1.set_ylabel("Priority Normalized #ofBuses & Next Total Gaps(cost)")
ax1.set_title("40Trips3ChargingStations_CHS-SA")
ax1.legend(loc="upper right")
FortyTrips_3cs_df = visualizeSolution(solutionspaces_40_3[1:], "40Trips3CSs-CHS-SA Pareto Front", all_schs_3cs, recharge_3cs_arcs)
newdf_40_3cs = visualizeResult(new_schedule_40_3, all_schs_3cs, "CHS_40Trips-3CS", cs_deadheads)

trips40_df_3cs = newdf_40_3cs.groupby('bus_id', group_keys=False).apply(apply_custom_shift)
trips40_df_3cs['next_dep'] = trips40_df_3cs['next_dep'].fillna(0)
trips40_df_3cs['difference'] = trips40_df_3cs['next_dep'] - trips40_df_3cs['arr_time']
trips40_df_3cs['difference'] = trips40_df_3cs['difference'].apply(lambda x: 0 if x < 0 else x)
chs_40Trips3cs_IDLE_soln = trips40_df_3cs.groupby(['bus_id']).agg(
    trips=('trip_id', concat_str),
    numRecharge=('trip_id',countRecharge),
    numTrips=('trip_id', countTrips),
    gapTime=('difference', 'sum')
)

chs_40Trips3cs_IDLE_soln.sort_values(['gapTime'], ascending=False)
# ################################################################################### 54 TRIPS ###############################################################################################
# trips = {
#     0: {'name': 'DEPOT', 'type': 'depot', 'duration': 0, 'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
#     1: {'name': 'AB1', 'type': 'trip', 'duration': 135, 'dep_time': 560, 'arr_time': 695, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
#     2: {'name': 'AB2', 'type': 'trip', 'duration': 135, 'dep_time': 740, 'arr_time': 875, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
#     3: {'name': 'AB3', 'type': 'trip', 'duration': 135, 'dep_time': 830, 'arr_time': 965, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
#     4: {'name': 'AB4', 'type': 'trip', 'duration': 135, 'dep_time': 920, 'arr_time': 1055, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
#     5: {'name': 'AB5', 'type': 'trip', 'duration': 135, 'dep_time': 980, 'arr_time': 1115, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
#     6: {'name': 'AB6', 'type': 'trip', 'duration': 135, 'dep_time': 1040, 'arr_time': 1175, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
#     7: {'name': 'AB7', 'type': 'trip', 'duration': 135, 'dep_time': 1100, 'arr_time': 1235, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
#     8: {'name': 'AB8', 'type': 'trip', 'duration': 135, 'dep_time': 1190, 'arr_time': 1325, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
#     9: {'name': 'AC1', 'type': 'trip', 'duration': 140, 'dep_time': 555, 'arr_time': 695, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
#     10: {'name': 'AC2', 'type': 'trip', 'duration': 140, 'dep_time': 675, 'arr_time': 815, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
#     11: {'name': 'AC3', 'type': 'trip', 'duration': 140, 'dep_time': 795, 'arr_time': 935, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
#     12: {'name': 'AC4', 'type': 'trip', 'duration': 140, 'dep_time': 1155, 'arr_time': 1295, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
#     13: {'name': 'AC5', 'type': 'trip', 'duration': 140, 'dep_time': 1215, 'arr_time': 1355, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
#     14: {'name': 'AD1', 'type': 'trip', 'duration': 140, 'dep_time': 560, 'arr_time': 700, 'dep_term': 'Terminal A', 'arr_term': 'Terminal D'},
#     15: {'name': 'AD2', 'type': 'trip', 'duration': 140, 'dep_time': 630, 'arr_time': 770, 'dep_term': 'Terminal A', 'arr_term': 'Terminal D'},
#     16: {'name': 'AD3', 'type': 'trip', 'duration': 135, 'dep_time': 790, 'arr_time': 925, 'dep_term': 'Terminal A', 'arr_term': 'Terminal D'},
#     17: {'name': 'AD4', 'type': 'trip', 'duration': 140, 'dep_time': 970, 'arr_time': 1110, 'dep_term': 'Terminal A', 'arr_term': 'Terminal D'},
#     18: {'name': 'AD5', 'type': 'trip', 'duration': 140, 'dep_time': 1120, 'arr_time': 1240, 'dep_term': 'Terminal A', 'arr_term': 'Terminal D'},
#     19: {'name': 'BA1', 'type': 'trip', 'duration': 135, 'dep_time': 540, 'arr_time': 675, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
#     20: {'name': 'BA2', 'type': 'trip', 'duration': 135, 'dep_time': 630, 'arr_time': 765, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
#     21: {'name': 'BA3', 'type': 'trip', 'duration': 135, 'dep_time': 720, 'arr_time': 855, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
#     22: {'name': 'BA4', 'type': 'trip', 'duration': 135, 'dep_time': 810, 'arr_time': 945, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
#     23: {'name': 'BA5', 'type': 'trip', 'duration': 135, 'dep_time': 900, 'arr_time': 1035, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
#     24: {'name': 'BA6', 'type': 'trip', 'duration': 135, 'dep_time': 990, 'arr_time': 1125, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
#     25: {'name': 'BA7', 'type': 'trip', 'duration': 135, 'dep_time': 1080, 'arr_time': 1215, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
#     26: {'name': 'BA8', 'type': 'trip', 'duration': 135, 'dep_time': 1140, 'arr_time': 1275, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
#     27: {'name': 'BC1', 'type': 'trip', 'duration': 140, 'dep_time': 435, 'arr_time': 570, 'dep_term': 'Terminal B', 'arr_term': 'Terminal C'},
#     28: {'name': 'BC2', 'type': 'trip', 'duration': 140, 'dep_time': 595, 'arr_time': 735, 'dep_term': 'Terminal B', 'arr_term': 'Terminal C'},
#     29: {'name': 'BC3', 'type': 'trip', 'duration': 120, 'dep_time': 800, 'arr_time': 920, 'dep_term': 'Terminal B', 'arr_term': 'Terminal C'},
#     30: {'name': 'BC4', 'type': 'trip', 'duration': 130, 'dep_time': 950, 'arr_time': 1130, 'dep_term': 'Terminal B', 'arr_term': 'Terminal C'},
#     31: {'name': 'CA1', 'type': 'trip', 'duration': 140, 'dep_time': 570, 'arr_time': 710, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
#     32: {'name': 'CA2', 'type': 'trip', 'duration': 140, 'dep_time': 740, 'arr_time': 880, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
#     33: {'name': 'CA3', 'type': 'trip', 'duration': 140, 'dep_time': 860, 'arr_time': 1000, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
#     34: {'name': 'CA4', 'type': 'trip', 'duration': 140, 'dep_time': 920, 'arr_time': 1060, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
#     35: {'name': 'CA5', 'type': 'trip', 'duration': 140, 'dep_time': 1040, 'arr_time': 1180, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
#     36: {'name': 'CB1', 'type': 'trip', 'duration': 120, 'dep_time': 300, 'arr_time': 420, 'dep_term': 'Terminal C', 'arr_term': 'Terminal B'},
#     37: {'name': 'CB2', 'type': 'trip', 'duration': 140, 'dep_time': 540, 'arr_time': 680, 'dep_term': 'Terminal C', 'arr_term': 'Terminal B'},
#     38: {'name': 'CB3', 'type': 'trip', 'duration': 140, 'dep_time': 645, 'arr_time': 785, 'dep_term': 'Terminal C', 'arr_term': 'Terminal B'},
#     39: {'name': 'CB4', 'type': 'trip', 'duration': 120, 'dep_time': 900, 'arr_time': 1020, 'dep_term': 'Terminal C', 'arr_term': 'Terminal B'},
#     40: {'name': 'CD1', 'type': 'trip', 'duration': 140, 'dep_time': 450, 'arr_time': 585, 'dep_term': 'Terminal C', 'arr_term': 'Terminal D'},
#     41: {'name': 'CD2', 'type': 'trip', 'duration': 135, 'dep_time': 550, 'arr_time': 685, 'dep_term': 'Terminal C', 'arr_term': 'Terminal D'},
#     42: {'name': 'CD3', 'type': 'trip', 'duration': 135, 'dep_time': 600, 'arr_time': 735, 'dep_term': 'Terminal C', 'arr_term': 'Terminal D'},
#     43: {'name': 'CD4', 'type': 'trip', 'duration': 135, 'dep_time': 765, 'arr_time': 900, 'dep_term': 'Terminal C', 'arr_term': 'Terminal D'},
#     44: {'name': 'CD5', 'type': 'trip', 'duration': 120, 'dep_time': 850, 'arr_time': 970, 'dep_term': 'Terminal C', 'arr_term': 'Terminal D'},
#     45: {'name': 'DA1', 'type': 'trip', 'duration': 140, 'dep_time': 720, 'arr_time': 860, 'dep_term': 'Terminal D', 'arr_term': 'Terminal A'},
#     46: {'name': 'DA2', 'type': 'trip', 'duration': 135, 'dep_time': 750, 'arr_time': 885, 'dep_term': 'Terminal D', 'arr_term': 'Terminal A'},
#     47: {'name': 'DA3', 'type': 'trip', 'duration': 140, 'dep_time': 800, 'arr_time': 940, 'dep_term': 'Terminal D', 'arr_term': 'Terminal A'},
#     48: {'name': 'DA4', 'type': 'trip', 'duration': 140, 'dep_time': 1000, 'arr_time': 1140, 'dep_term': 'Terminal D', 'arr_term': 'Terminal A'},
#     49: {'name': 'DA5', 'type': 'trip', 'duration': 140, 'dep_time': 1200, 'arr_time': 1340, 'dep_term': 'Terminal D', 'arr_term': 'Terminal A'},
#     50: {'name': 'DC1', 'type': 'trip', 'duration': 140, 'dep_time': 600, 'arr_time': 740, 'dep_term': 'Terminal D', 'arr_term': 'Terminal C'},
#     51: {'name': 'DC2', 'type': 'trip', 'duration': 120, 'dep_time': 765, 'arr_time': 885, 'dep_term': 'Terminal D', 'arr_term': 'Terminal C'},
#     52: {'name': 'DC3', 'type': 'trip', 'duration': 120, 'dep_time': 780, 'arr_time': 900, 'dep_term': 'Terminal D', 'arr_term': 'Terminal C'},
#     53: {'name': 'DC4', 'type': 'trip', 'duration': 135, 'dep_time': 965, 'arr_time': 1100, 'dep_term': 'Terminal D', 'arr_term': 'Terminal C'},
#     54: {'name': 'DC5', 'type': 'trip', 'duration': 135, 'dep_time': 975, 'arr_time': 1110, 'dep_term': 'Terminal D', 'arr_term': 'Terminal C'}
# }

# charging_station = {
#     len(trips): {'name': 'CS1', 'type': 'cs', 'duration': 50,  'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
# }

# charging_stations1 = {
#     len(trips): {'name': 'CS1', 'type': 'cs', 'duration': 50,  'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
#     len(trips)+1: {'name': 'CS1a', 'type': 'cs', 'duration': 50,  'dep_time': 0, 'arr_time': 1440, 'dep_term': 'Terminal A', 'arr_term': 'Terminal A'},
# }

# charging_stations2 = {
#     len(trips): {'name': 'CS1', 'type': 'cs', 'duration': 50,  'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
#     len(trips)+1: {'name': 'CS1a', 'type': 'cs', 'duration': 50,  'dep_time': 0, 'arr_time': 1440, 'dep_term': 'Terminal A', 'arr_term': 'Terminal A'},
#     len(trips)+2: {'name': 'CS2b', 'type': 'cs', 'duration': 50,  'dep_time': 0, 'arr_time': 1440, 'dep_term': 'Terminal B', 'arr_term': 'Terminal B'},
# }

# # TERMINALS AND MINIMUM INTERVALS BETWEEN TRIPS
# terminals = {
#     'Terminal A': {'max_interval': 15},
#     'Terminal B': {'max_interval': 15},
#     'Terminal C': {'max_interval': 15},
#     'Terminal D': {'max_interval': 15}
# }
# ### Add extra parameter that allows e-Bus to complete schedule that is not continuous (aka the start node does not have to be the end node of the previous)
# ## Defining the deadhead parameter from Terminals
# cs_deadheads = {
#     'Terminal A': {'CS1': 40, 'CS1a': 10, 'CS2b': 75, 'CS3c': 75, 'CS4d': 75},
#     'Terminal B': {'CS1': 40, 'CS1a': 75, 'CS2b': 10, 'CS3c': 75, 'CS4d': 75},
#     'Terminal C': {'CS1': 40, 'CS1a': 75, 'CS2b': 75, 'CS3c': 10, 'CS4d': 75},
#     'Terminal D': {'CS1': 40, 'CS1a': 75, 'CS2b': 75, 'CS3c': 75, 'CS4d': 10},
# }

# D_MAX = 350
# CHARGING_TIME = 50
# trips_df = pd.DataFrame.from_dict(trips, orient='index')
# print(trips_df)
# ## Creating Gamma and Delta
# arcs = feasible_pairs(trips_df, terminals)
# print(charging_station)

# ######################################################## 1CS #####################################################################
# recharge_1cs_arcs = feasible_recharge(trips_df, cs_deadheads, recharge=charging_station, terminals=terminals)
# cs_ids = set(list(charging_station.keys()))
# all_schedules_1cs = {**trips, **charging_station}
# all_schs_1cs = pd.DataFrame.from_dict(all_schedules_1cs, orient='index')
# all_schs_1cs['ID'] = range(len(all_schs_1cs))
# durations_1cs = {idx: {'duration': all_schs_1cs.loc[idx, 'duration']} for idx in all_schs_1cs.index if idx != 0}
# ###################################################### CHS-SA ####################################################################

# start_time = time.time()
# print(f"{time.ctime()}")
# schedules_tab, schedules_54_1 = constructiveScheduler(all_schs_1cs, arcs, recharge_1cs_arcs, cs_ids)
# solution = vectorSchRepresentation(schedules_54_1)
# end_time = time.time()
# g_T54CS1_time = end_time - start_time
# print(f"{time.ctime()}\nTime elapse to compute the solution = {(g_T54CS1_time)} seconds")
# print(f"number of buses = {len(schedules_54_1)}")
# print(schedules_54_1)
# print('-'*100)
# start_time = time.time()
# print("starting simulated annealing....")
# new_schedule_54_1, cost_54_1, cost_diffs_54_1, temp_54_1, it_54_1, costs_54_1, solution_spaces_54_1, best_costs_54_1 = annealing(solution, all_schs_1cs, arcs, recharge_1cs_arcs)
# end_time = time.time()
# T54CS1_time = (end_time-start_time)+g_T54CS1_time
# print(f"{time.ctime()}\nTime elapse to compute the solution = {(T54CS1_time)} seconds")
# print(f"prev_schedule = {solution} with number of buses = {len(solution)}... \nnext_schedule = {new_schedule_54_1} with number of buses = {len(new_schedule_54_1)}")

# fig1, ax1 = plt.subplots()
# ax1.plot(range(it_54_1), costs_54_1, label="Current_Solution Cost")
# ax1.plot(range(it_54_1), best_costs_54_1, label="Best_Solution Cost")
# # plt.gca().invert_xaxis()
# ax1.set_xlabel("Iteration (it)")
# # plt.ylabel("[weighted_sum] #ofBuses + Total Gap Time (cost)")
# ax1.set_ylabel("Priority Normalized #ofBuses & Next Total Gaps(cost)")
# ax1.set_title("54Trips1ChargingStation_CHS-SA")
# ax1.legend(loc="upper right")
# #plt.savefig("54Trips1CS.png")

# FiftyTrips_1cs_df = visualizeSolution(solution_spaces_54_1[1:], "54Trips1CSs-CHS-SA Pareto Front", all_schs_1cs, recharge_1cs_arcs)
# newdf_54_1cs = visualizeResult(new_schedule_54_1, all_schs_1cs, "CHS_54Trips-1CS", cs_deadheads)

# trips54_1_df = newdf_54_1cs.groupby('bus_id', group_keys=False).apply(apply_custom_shift)
# trips54_1_df['next_dep'] = trips54_1_df['next_dep'].fillna(0)
# trips54_1_df['difference'] = trips54_1_df['next_dep'] - trips54_1_df['arr_time']
# trips54_1_df['difference'] = trips54_1_df['difference'].apply(lambda x: 0 if x < 0 else x)
# chs_54Trips1cs_IDLE_soln = trips54_1_df.groupby(['bus_id']).agg(
#     trips=('trip_id', concat_str),
#     numRecharge=('trip_id',countRecharge),
#     numTrips=('trip_id', countTrips),
#     gapTime=('difference', 'sum')
# )

# chs_54Trips1cs_IDLE_soln.sort_values(['gapTime'], ascending=False)
# print("Ending the process for 54Trips and 1 CS......")
# # ################################################################ 2 CS ##################################################################################
# # ## Creating Phi and Delta
# # recharge_2cs_arcs = feasible_recharge(trips_df, cs_deadheads, recharge=charging_stations1, terminals=terminals)
# # print(recharge_2cs_arcs)
# # cs_ids = set(list(charging_stations1.keys()))
# # all_schedules_2cs = {**trips, **charging_stations1}
# # all_schs_2cs = pd.DataFrame.from_dict(all_schedules_2cs, orient='index')
# # all_schs_2cs['ID'] = range(len(all_schs_2cs))
# # durations_2cs = {idx: {'duration': all_schs_2cs.loc[idx, 'duration']} for idx in all_schs_2cs.index if idx != 0}

# # start_time = time.time()
# # print(f"{time.ctime()}")
# # test_schedules_tab, schedules_54_2 = constructiveScheduler(all_schs_2cs, arcs, recharge_2cs_arcs, set(cs_ids))
# # test_solution = vectorSchRepresentation(schedules_54_2)
# # end_time = time.time()
# # g_T54CS2_time = end_time - start_time
# # print(f"{time.ctime()}\nTime elapse to compute the solution = {(g_T54CS2_time)} seconds")
# # print(f"number of buses = {len(schedules_54_2)}")
# # print(schedules_54_2)
# # print('-'*100)
# # print("starting simulated annealing....")
# # start_time = time.time()
# # new_schedule_54_2, cost_54_2, cost_diffs_54_2, temp_54_2, it_54_2, costs_54_2, solutionspaces_54_2, best_54_2 = annealing(test_solution, all_schs_2cs, arcs, recharge_2cs_arcs)
# # end_time = time.time()
# # T54CS2_time = (end_time - start_time) + g_T54CS2_time
# # print(f"{time.ctime()}\nTime elapse to compute the solution = {(T54CS2_time)} seconds")
# # print(f"prev_schedule = {test_solution} with number of buses = {len(test_solution)}... \nnext_schedule = {new_schedule_54_2} with number of buses = {len(new_schedule_54_2)}")

# # plt.plot(range(it_54_2), costs_54_2, label="Current_Solution Cost")
# # plt.plot(range(it_54_2), best_54_2, label="Best_Solution Cost")
# # # plt.gca().invert_xaxis()
# # plt.xlabel("Iteration (it)")
# # # plt.ylabel("[weighted_sum] #ofBuses + Total Gap Time (cost)")
# # plt.ylabel("Priority Normalized #ofBuses & Next Total Gaps(cost)")
# # plt.title("54Trips2ChargingStations_CHS-SA")
# # plt.legend(loc="upper right")
# # plt.savefig("54Trips2CSs.png")

# # FiftyTrips_2cs_df = visualizeSolution(solutionspaces_54_2[1:], "54Trips2CSs-CHS-SA Pareto Front", all_schs_2cs, recharge_2cs_arcs)
# # newdf_54_2cs = visualizeResult(new_schedule_54_2, all_schs_2cs, "CHS_54Trips-2CS")

# # trips54_df_2cs = newdf_54_2cs.copy(deep=True)
# # trips54_df_2cs['next_dep'] = trips54_df_2cs.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
# # trips54_df_2cs['difference'] = trips54_df_2cs['next_dep'] - trips54_df_2cs['arr_time']
# # trips54_df_2cs['difference'] = trips54_df_2cs['difference'].apply(lambda x: 0 if x < 0 else x)
# # trips54_soln_2cs = trips54_df_2cs.groupby(['bus_id'])['difference'].sum()

# # chs_54Trips2cs_IDLE_soln = trips54_df_2cs.groupby(['bus_id']).agg(
# #     trips=('trip_id', concat_str),
# #     numRecharge=('trip_id',countRecharge),
# #     numTrips=('trip_id', countTrips),
# #     gapTime=('difference', 'sum')
# # )
# # chs_54Trips2cs_IDLE_soln.sort_values(['gapTime'], ascending=False)

# ################################################################### 3CS #############################################################################################
# ## Creating Phi and Delta
# recharge_3cs_arcs = feasible_recharge(trips_df, cs_deadheads, recharge=charging_stations2, terminals=terminals)
# print(recharge_3cs_arcs)
# cs_ids = set(list(charging_stations2.keys()))
# all_schedules_3cs = {**trips, **charging_stations2}
# all_schs_3cs = pd.DataFrame.from_dict(all_schedules_3cs, orient='index')
# all_schs_3cs['ID'] = range(len(all_schs_3cs))
# durations_3cs = {idx: {'duration': all_schs_3cs.loc[idx, 'duration']} for idx in all_schs_3cs.index if idx != 0}

# ###### import time
# start_time = time.time()
# print(f"{time.ctime()}")
# test_schedules_tab, schedules_54_3 = constructiveScheduler(all_schs_3cs, arcs, recharge_3cs_arcs, set(cs_ids))
# test_solution = vectorSchRepresentation(schedules_54_3)
# end_time = time.time()
# g_T54CS3_time = end_time - start_time
# print(f"{time.ctime()}\nTime elapse to compute the solution = {g_T54CS3_time} seconds")
# print(f"number of buses = {len(schedules_54_3)}")
# print(schedules_54_3)
# print('-'*100)
# print("starting simulated annealing....")
# start_time = time.time()
# new_schedule_54_3, cost_54_3, cost_diffs_54_3, temp_54_3, it_54_3, costs_54_3, solutionspaces_54_3, best_54_3 = annealing(test_solution, all_schs_3cs, arcs, recharge_3cs_arcs)
# end_time = time.time()
# T54CS3_time = (end_time - start_time) + g_T54CS3_time
# print(f"{time.ctime()}\nTime elapse to compute the solution = {(T54CS3_time)} seconds")
# print(f"prev_schedule = {test_solution} with number of buses = {len(test_solution)}... \nnext_schedule = {new_schedule_54_3} with number of buses = {len(new_schedule_54_3)}")

# fig1, ax1 = plt.subplots()
# ax1.plot(range(it_54_3), costs_54_3, label="Current_Solution Cost")
# ax1.plot(range(it_54_3), best_54_3, label="Best_Solution Cost")
# # plt.gca().invert_xaxis()
# ax1.set_xlabel("Iteration (it)")
# # plt.ylabel("[weighted_sum] #ofBuses + Total Gap Time (cost)")
# ax1.set_ylabel("Priority Normalized #ofBuses & Next Total Gaps(cost)")
# ax1.set_title("54Trips3ChargingStations_CHS-SA")
# ax1.legend(loc="upper right")
# FiftyTrips_3cs_df = visualizeSolution(solutionspaces_54_3[1:], "54Trips3CSs-CHS-SA Pareto Front", all_schs_3cs, recharge_3cs_arcs)
# newdf_54_3cs = visualizeResult(new_schedule_54_3, all_schs_3cs, "CHS_54Trips-3CS", cs_deadheads)

# trips54_df_3cs = newdf_54_3cs.groupby('bus_id', group_keys=False).apply(apply_custom_shift)
# trips54_df_3cs['next_dep'] = trips54_df_3cs['next_dep'].fillna(0)
# trips54_df_3cs['difference'] = trips54_df_3cs['next_dep'] - trips54_df_3cs['arr_time']
# trips54_df_3cs['difference'] = trips54_df_3cs['difference'].apply(lambda x: 0 if x < 0 else x)
# chs_54Trips3cs_IDLE_soln = trips54_df_3cs.groupby(['bus_id']).agg(
#     trips=('trip_id', concat_str),
#     numRecharge=('trip_id',countRecharge),
#     numTrips=('trip_id', countTrips),
#     gapTime=('difference', 'sum')
# )

# chs_54Trips3cs_IDLE_soln.sort_values(['gapTime'], ascending=False)

# cs_ids = ["CS1", "CS2", "CS3"]
# def countRecharge(x):
#     trips = x.split(",")
#     return len(list(filter(lambda x: x in cs_ids, trips)))
# chs_10Trips1cs_IDLE_soln['numRecharge'] = chs_10Trips1cs_IDLE_soln.trips.apply(countRecharge)
# chs_10Trips2cs_IDLE_soln['numRecharge'] = chs_10Trips2cs_IDLE_soln.trips.apply(countRecharge)

# chs_30Trips1cs_IDLE_soln['numRecharge'] = chs_30Trips1cs_IDLE_soln.trips.apply(countRecharge)
# chs_30Trips2cs_IDLE_soln['numRecharge'] = chs_30Trips2cs_IDLE_soln.trips.apply(countRecharge)

# chs_54Trips1cs_IDLE_soln['numRecharge'] = chs_54Trips1cs_IDLE_soln.trips.apply(countRecharge)
# chs_54Trips3cs_IDLE_soln['numRecharge'] = chs_54Trips3cs_IDLE_soln.trips.apply(countRecharge)

# test = chs_10Trips1cs_IDLE_soln.describe().loc['mean']
# test['numBuses'] = chs_10Trips1cs_IDLE_soln.describe().loc['count','gapTime']
# test['time_to_best_soln'] = T10CS1_time
# test = test.to_frame().rename(columns={"mean":"CHS_10Trips1CS"})

# test2 = chs_10Trips3cs_IDLE_soln.describe().loc['mean']
# test2['numBuses'] = chs_10Trips3cs_IDLE_soln.describe().loc['count','gapTime']
# test2['time_to_best_soln'] = T10CS3_time
# test2 = test2.to_frame().rename(columns={"mean":"CHS_10Trips3CS"})

# result_10Trips = pd.concat([test, test2], axis=1)
# print(result_10Trips)

# test = chs_20Trips1cs_IDLE_soln.describe().loc['mean']
# test['numBuses'] = chs_20Trips1cs_IDLE_soln.describe().loc['count','gapTime']
# test['time_to_best_soln'] = T20CS1_time
# test = test.to_frame().rename(columns={"mean":"CHS_20Trips1CS"})

# test2 = chs_20Trips3cs_IDLE_soln.describe().loc['mean']
# test2['numBuses'] = chs_20Trips3cs_IDLE_soln.describe().loc['count','gapTime']
# test2['time_to_best_soln'] = T20CS3_time
# test2 = test2.to_frame().rename(columns={"mean":"CHS_20Trips3CS"})

# result_20Trips = pd.concat([test, test2], axis=1)
# print(result_20Trips)

# test = chs_30Trips1cs_IDLE_soln.describe().loc['mean']
# test['numBuses'] = chs_30Trips1cs_IDLE_soln.describe().loc['count','gapTime']
# test['time_to_best_soln'] = T30CS1_time
# test = test.to_frame().rename(columns={"mean":"CHS_30Trips1CS"})

# test2 = chs_30Trips3cs_IDLE_soln.describe().loc['mean']
# test2['numBuses'] = chs_30Trips3cs_IDLE_soln.describe().loc['count','gapTime']
# test2['time_to_best_soln'] = T30CS3_time
# test2 = test2.to_frame().rename(columns={"mean":"CHS_30Trips3CS"})

# result_30Trips = pd.concat([test, test2], axis=1)
# print(result_30Trips)

# test = chs_40Trips1cs_IDLE_soln.describe().loc['mean']
# test['numBuses'] = chs_40Trips1cs_IDLE_soln.describe().loc['count','gapTime']
# test['time_to_best_soln'] = T40CS1_time
# test = test.to_frame().rename(columns={"mean":"CHS_40Trips1CS"})

# test2 = chs_40Trips3cs_IDLE_soln.describe().loc['mean']
# test2['numBuses'] = chs_40Trips3cs_IDLE_soln.describe().loc['count','gapTime']
# test2['time_to_best_soln'] = T40CS3_time
# test2 = test2.to_frame().rename(columns={"mean":"CHS_40Trips3CS"})

# result_40Trips = pd.concat([test, test2], axis=1)
# print(result_40Trips)

# test = chs_54Trips1cs_IDLE_soln.describe().loc['mean']
# test['numBuses'] = chs_54Trips1cs_IDLE_soln.describe().loc['count','gapTime']
# test['time_to_best_soln'] = T54CS1_time
# test = test.to_frame().rename(columns={"mean":"CHS_54Trips1CS"})

# test2 = chs_54Trips3cs_IDLE_soln.describe().loc['mean']
# test2['numBuses'] = chs_54Trips3cs_IDLE_soln.describe().loc['count','gapTime']
# test2['time_to_best_soln'] = T54CS3_time
# test2 = test2.to_frame().rename(columns={"mean":"CHS_54Trips3CS"})

# result_54Trips = pd.concat([test, test2], axis=1)
# print(result_54Trips)

# test.columns = pd.MultiIndex.from_product([["CHS_30Trips2CS"], test.columns])
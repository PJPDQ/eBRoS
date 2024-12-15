from itertools import chain
import pandas as pd
import time
import matplotlib.pyplot as plt
max_it = 10
from misc import vectorSchRepresentation, feasible_pairs, feasible_recharge, visualizeResult, visualizeSolution, drawBusTrips
from chs import constructiveScheduler
from sa import annealing, get_total_gap
cs_ids = ["CS1", "CS2", "CS3"]

def draw_busTrips(data):
    min_val = data.dep_time.min() - 30
    max_val = data.arr_time.max() + 10
    bar_style = {"alpha": 1.0, "lw": 10, "solid_capstyle": "butt"}
    text_style = {
        "fontsize": 5,
        "color": "red",
        "weight": "bold",
        "ha": "center",
        "va": "center",
    }
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, xlabel="Departure/Arrival Time", title="Bus Schedule")
    ax.get_yaxis().set_visible(False)
    idx = 1
    print(data.tail())
    for id, row in data.iterrows():
        name, dep_time, arr_time, dep_city, arr_city = row
        ax.plot([dep_time, arr_time], [idx] * 2, "gray", **bar_style)
        ax.text((dep_time + arr_time) / 2, idx, f"{name}", **text_style)
        ax.text(dep_time+2, idx, f"{dep_time}", {**text_style, 'fontsize': 5, "color": "black"})
        ax.text(arr_time-2, idx, f"{arr_time}", {**text_style, 'fontsize': 5, "color": "black"})
        idx += 1
    for hr in range(0, 1440, 50):
        ax.axvline(hr, alpha=0.1)
    ax.set_xlim(min_val, max_val)
    ax.set_xticks([200 + 100*i for i in range(1, 13)])
    return ax

def countRecharge(x):
    trips = x.to_list()
    return len(list(filter(lambda x: x in cs_ids,trips)))

def countTrips(x):
    trips = x.to_list()
    return len(set(trips) - set(cs_ids))

def concat_str(x):
    return ','.join(x)

# EXAMPLE DATA SET, DEPARTURE AND ARRIVAL TIMES IN MINUTES
trips = {
    0: {'name': 'DEPOT', 'type': 'depot', 'duration': 0, 'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
    1: {'name': 'AC1', 'type': 'trip', 'duration': 140,  'dep_time': 555, 'arr_time': 695, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
    2: {'name': 'AC2', 'type': 'trip', 'duration': 140, 'dep_time': 675, 'arr_time': 815, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
    3: {'name': 'AC3', 'type': 'trip', 'duration': 140,  'dep_time': 795, 'arr_time': 935, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
    4: {'name': 'AC4', 'type': 'trip', 'duration': 140,  'dep_time': 1155, 'arr_time': 1295, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
    5: {'name': 'AC5', 'type': 'trip', 'duration': 140,  'dep_time': 1215, 'arr_time': 1355, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
    6: {'name': 'CA1', 'type': 'trip', 'duration': 140,  'dep_time': 570, 'arr_time': 710, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
    7: {'name': 'CA2', 'type': 'trip', 'duration': 140,  'dep_time': 740, 'arr_time': 880, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
    8: {'name': 'CA3', 'type': 'trip', 'duration': 140,  'dep_time': 860, 'arr_time': 1000, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
    9: {'name': 'CA4', 'type': 'trip', 'duration': 140,  'dep_time': 920, 'arr_time': 1060, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
    10: {'name': 'CA5', 'type': 'trip', 'duration': 140,  'dep_time': 1040, 'arr_time': 1180, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
}

charging_station = {
    11: {'name': 'CS1', 'type': 'cs', 'duration': 60,  'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
}

charging_stations = {
    11: {'name': 'CS1', 'type': 'cs', 'duration': 60,  'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
    12: {'name': 'CS2', 'type': 'cs', 'duration': 70,  'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
}
# TERMINALS AND MINIMUM INTERVALS BETWEEN TRIPS
terminals = {
    'Terminal A': {'max_interval': 15},
    'Terminal C': {'max_interval': 15},
}

### Add extra parameter that allows e-Bus to complete schedule that is not continuous (aka the start node does not have to be the end node of the previous)
## Defining the deadhead parameter from Terminals
deadheads = {
    'Terminal A': {'deadhead_time': 10},
    'Terminal C': {'deadhead_time': 5},
}
D_MAX = 350
CHARGING_TIME = 100

trips_df = pd.DataFrame.from_dict(trips, orient='index')
print(trips_df)
## Creating Gamma and Delta
print(feasible_pairs(trips_df, terminals=terminals))
arcs = feasible_pairs(trips_df, terminals=terminals)

print(charging_stations)
## Creating Phi and Delta
print(feasible_recharge(trips_df, arcs, recharge=charging_stations))

################################################################### 1CS #############################################################################################
recharge_1cs_arcs = feasible_recharge(trips_df, arcs, recharge=charging_station)
cs_ids = set(list(charging_station.keys()))
all_schedules_1cs = {**trips, **charging_station}
all_schs_1cs = pd.DataFrame.from_dict(all_schedules_1cs, orient='index')
all_schs_1cs['ID'] = range(len(all_schs_1cs))
durations_1cs = {idx: {'duration': all_schs_1cs.loc[idx, 'duration']} for idx in all_schs_1cs.index if idx != 0}

################################################################### CHS-SA ###########################################################################################
####### CHS
import time
start_time = time.time()
print(f"{time.ctime()}")
schedules_tab_1cs, schedules_1cs = constructiveScheduler(all_schs_1cs, arcs, recharge_1cs_arcs, cs_ids)
solution = vectorSchRepresentation(schedules_1cs)
end_time = time.time()
g_T10CS1_time = end_time-start_time
print(f"{time.ctime()}\nTime elapse to compute the solution = {(end_time - start_time)} seconds")
print(f"number of buses = {len(schedules_1cs)}")
print(schedules_1cs)

####### SA
print('-'*100)
print("starting simulated annealing....")
start_time = time.time()
# test_new_schedule, test_cost, test_cost_diffs, test_temp, test_it, test_costs, test_solutionspaces, test_best
new_schedule_1cs, cost_1cs, cost_diffs_1cs, temp_1cs, it_1cs, test_costs_1cs, solution_spaces_1cs, best_costs_1cs = annealing(solution, all_schs_1cs, arcs, recharge_1cs_arcs)
end_time = time.time()
T10CS1_time = (end_time-start_time) + g_T10CS1_time
print(f"{time.ctime()}\nTime elapse to compute the solution = {(T10CS1_time)} seconds")
print(f"prev_schedule = {solution} with number of buses = {len(solution)}... \nnext_schedule = {new_schedule_1cs} with number of buses = {len(new_schedule_1cs)}")

print(f"Number of buses previous = {len(schedules_1cs)}... new = {len(new_schedule_1cs)} ")
print(f"Prev: {vectorSchRepresentation(schedules_1cs)}\nNew: {new_schedule_1cs}")
print(f"Total Gap among buses=> prev = {get_total_gap(solution, all_schs_1cs, recharge_1cs_arcs)}... new = {get_total_gap(new_schedule_1cs, all_schs_1cs, recharge_1cs_arcs)}")

plt.plot(range(it_1cs), test_costs_1cs, label="current_solution")
plt.plot(range(it_1cs), best_costs_1cs, label="best_solution")
plt.xlabel("Iteration (it)")
plt.ylabel("#ofBuses(cost) + Total Gaps between Schedules")
plt.title("10Trips1ChargingStation_CHS-SA")
plt.legend(loc="upper right")
# plt.savefig("10Trips1CS.png")

TenTrips_df = visualizeSolution(solution_spaces_1cs[1:], "10Trips1CS-CHS-SA Pareto Front", all_schs_1cs, recharge_1cs_arcs)
newdf = visualizeResult(new_schedule_1cs, all_schs_1cs, "CHS_10Trips-1CS")
trips10_df = newdf.copy(deep=True)
trips10_df['next_dep'] = trips10_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
trips10_df['difference'] = trips10_df['next_dep'] - trips10_df['arr_time']
trips10_df['difference'] = trips10_df['difference'].apply(lambda x: 0 if x < 0 else x)
trips10_soln = trips10_df.groupby(['bus_id'])['difference'].sum()
# trips10_df.to_csv("10Trips1CS.csv")
chs_10Trips1cs_IDLE_soln = trips10_df.groupby(['bus_id']).agg({'trip_id':concat_str, 'dep_time': 'first', 'arr_time': 'last', 'dep_terminal': 'first', 'arr_terminal': 'last', 'difference': 'sum'})
chs_10Trips1cs_IDLE_soln = trips10_df.groupby(['bus_id']).agg(
    trips=('trip_id', concat_str),
    numRecharge=('trip_id',countRecharge),
    numTrips=('trip_id', countTrips),
    gapTime=('difference', 'sum')
)
################################################################### 2CS #############################################################################################
recharge_2cs_arcs = feasible_recharge(trips_df, arcs, recharge=charging_stations)
cs_ids = set(list(charging_stations.keys()))
all_schedules_2cs = {**trips, **charging_stations}
all_schs_2cs = pd.DataFrame.from_dict(all_schedules_2cs, orient='index')
all_schs_2cs['ID'] = range(len(all_schs_2cs))
durations_2cs = {idx: {'duration': all_schs_2cs.loc[idx, 'duration']} for idx in all_schs_2cs.index if idx != 0}

####### CHS
import time
start_time = time.time()
print(f"{time.ctime()}")
schedules_tab_2cs, schedules_2cs = constructiveScheduler(all_schs_2cs, arcs, recharge_2cs_arcs, cs_ids)
solution = vectorSchRepresentation(schedules_2cs)
end_time = time.time()
g_T10CS2_time = end_time - start_time
print(f"{time.ctime()}\nTime elapse to compute the solution = {(g_T10CS2_time)} seconds")
print(f"number of buses = {len(schedules_2cs)}")
print(schedules_2cs)
####### SA
print('-'*100)
print("starting simulated annealing....")
start_time = time.time()
new_schedule_2cs, cost_2cs, cost_diffs_2cs, temp_2cs, it_2cs, test_costs_2cs, solution_spaces_2cs, best_costs_2cs = annealing(solution, all_schs_2cs, arcs, recharge_2cs_arcs)
end_time = time.time()
T10CS2_time = (end_time-start_time)+g_T10CS2_time
print(f"{time.ctime()}\nTime elapse to compute the solution = {(T10CS2_time)} seconds")
print(f"prev_schedule = {solution} with number of buses = {len(solution)}... \nnext_schedule = {new_schedule_2cs} with number of buses = {len(new_schedule_2cs)}")

print(f"Number of buses previous = {len(schedules_2cs)}... new = {len(new_schedule_2cs)} ")
print(f"Prev: {vectorSchRepresentation(schedules_2cs)}\nNew: {new_schedule_2cs}")
print(f"Total Gap among buses=> prev = {get_total_gap(solution, all_schs_2cs, recharge_2cs_arcs)}... new = {get_total_gap(new_schedule_2cs, all_schs_2cs, recharge_2cs_arcs)}")

plt.plot(range(it_2cs), test_costs_2cs, label="current_solution")
plt.plot(range(it_2cs), best_costs_2cs, label="best_solution")
plt.xlabel("Iteration (it)")
plt.ylabel("#ofBuses(cost) + Total Gaps between Schedules")
plt.title("10Trips2ChargingStations_CHS-SA")
plt.legend(loc="upper right")
plt.savefig("10Trips2CSs.png")

TenTrips_df = visualizeSolution(solution_spaces_2cs[1:], "10Trips2CSs-CHS-SA Pareto Front", all_schs_2cs, recharge_2cs_arcs)
newdf = visualizeResult(new_schedule_2cs, all_schs_2cs, "CHS_10Trips-2CS")
trips10_df = newdf.copy(deep=True)
trips10_df['next_dep'] = trips10_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
trips10_df['difference'] = trips10_df['next_dep'] - trips10_df['arr_time']
trips10_df['difference'] = trips10_df['difference'].apply(lambda x: 0 if x < 0 else x)
trips10_soln = trips10_df.groupby(['bus_id'])['difference'].sum()
# trips10_df.to_csv("10Trips2CS.csv")
chs_10Trips2cs_IDLE_soln = trips10_df.groupby(['bus_id']).agg({'trip_id':concat_str, 'dep_time': 'first', 'arr_time': 'last', 'dep_terminal': 'first', 'arr_terminal': 'last', 'difference': 'sum'})
chs_10Trips2cs_IDLE_soln = trips10_df.groupby(['bus_id']).agg(
    trips=('trip_id', concat_str),
    numRecharge=('trip_id',countRecharge),
    numTrips=('trip_id', countTrips),
    gapTime=('difference', 'sum')
)
# ################################################################################# 30 TRIPS ##################################################################################
trips = {
    0: {'name': 'DEPOT', 'type': 'depot', 'duration': 0, 'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
    1: {'name': 'AB1', 'type': 'trip', 'duration': 135, 'dep_time': 560, 'arr_time': 695, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
    2: {'name': 'AB2', 'type': 'trip', 'duration': 135, 'dep_time': 740, 'arr_time': 875, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
    3: {'name': 'AB3', 'type': 'trip', 'duration': 135, 'dep_time': 830, 'arr_time': 965, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
    4: {'name': 'AB4', 'type': 'trip', 'duration': 135, 'dep_time': 920, 'arr_time': 1055, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
    5: {'name': 'AB5', 'type': 'trip', 'duration': 135, 'dep_time': 980, 'arr_time': 1115, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
    6: {'name': 'AB6', 'type': 'trip', 'duration': 135, 'dep_time': 1040, 'arr_time': 1175, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
    7: {'name': 'AB7', 'type': 'trip', 'duration': 135, 'dep_time': 1100, 'arr_time': 1235, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
    8: {'name': 'AB8', 'type': 'trip', 'duration': 135, 'dep_time': 1190, 'arr_time': 1325, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
    9: {'name': 'AC1', 'type': 'trip', 'duration': 140, 'dep_time': 555, 'arr_time': 695, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
    10: {'name': 'AC2', 'type': 'trip', 'duration': 140, 'dep_time': 675, 'arr_time': 815, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
    11: {'name': 'AC3', 'type': 'trip', 'duration': 140, 'dep_time': 795, 'arr_time': 935, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
    12: {'name': 'AC4', 'type': 'trip', 'duration': 140, 'dep_time': 1155, 'arr_time': 1295, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
    13: {'name': 'AC5', 'type': 'trip', 'duration': 140, 'dep_time': 1215, 'arr_time': 1355, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
    14: {'name': 'AD1', 'type': 'trip', 'duration': 210, 'dep_time': 560, 'arr_time': 770, 'dep_term': 'Terminal A', 'arr_term': 'Terminal D'},
    15: {'name': 'BA1', 'type': 'trip', 'duration': 135, 'dep_time': 540, 'arr_time': 675, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
    16: {'name': 'BA2', 'type': 'trip', 'duration': 135, 'dep_time': 630, 'arr_time': 765, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
    17: {'name': 'BA3', 'type': 'trip', 'duration': 135, 'dep_time': 720, 'arr_time': 855, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
    18: {'name': 'BA4', 'type': 'trip', 'duration': 135, 'dep_time': 810, 'arr_time': 945, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
    19: {'name': 'BA5', 'type': 'trip', 'duration': 135, 'dep_time': 900, 'arr_time': 1035, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
    20: {'name': 'BA6', 'type': 'trip', 'duration': 135, 'dep_time': 990, 'arr_time': 1125, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
    21: {'name': 'BA7', 'type': 'trip', 'duration': 135, 'dep_time': 1080, 'arr_time': 1215, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
    22: {'name': 'BA8', 'type': 'trip', 'duration': 135, 'dep_time': 1140, 'arr_time': 1275, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
    23: {'name': 'CA1', 'type': 'trip', 'duration': 140, 'dep_time': 570, 'arr_time': 710, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
    24: {'name': 'CA2', 'type': 'trip', 'duration': 140, 'dep_time': 740, 'arr_time': 880, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
    25: {'name': 'CA3', 'type': 'trip', 'duration': 140, 'dep_time': 860, 'arr_time': 1000, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
    26: {'name': 'CA4', 'type': 'trip', 'duration': 140, 'dep_time': 920, 'arr_time': 1060, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
    27: {'name': 'CA5', 'type': 'trip', 'duration': 140, 'dep_time': 1040, 'arr_time': 1180, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
    28: {'name': 'CD1', 'type': 'trip', 'duration': 270, 'dep_time': 450, 'arr_time': 720, 'dep_term': 'Terminal C', 'arr_term': 'Terminal D'},
    29: {'name': 'DA1', 'type': 'trip', 'duration': 210, 'dep_time': 860, 'arr_time': 1070, 'dep_term': 'Terminal D', 'arr_term': 'Terminal A'},
    30: {'name': 'DC1', 'type': 'trip', 'duration': 270, 'dep_time': 810, 'arr_time': 1080, 'dep_term': 'Terminal D', 'arr_term': 'Terminal C'}
}
charging_station = {
    31: {'name': 'CS1', 'type': 'cs', 'duration': 60,  'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
}

charging_stations = {
    31: {'name': 'CS1', 'type': 'cs', 'duration': 60,  'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
    32: {'name': 'CS2', 'type': 'cs', 'duration': 70,  'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
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
deadheads = {
    'Terminal A': {'Terminal B': 5, 'Terminal C': 10, 'Terminal D': 15},
    'Terminal B': {'Terminal A': 5, 'Terminal C': 15, 'Terminal D': 20},
    'Terminal C': {'Terminal A': 10, 'Terminal B': 15, 'Terminal D': 3},
    'Terminal D': {'Terminal A': 15, 'Terminal B': 20, 'Terminal C': 3},
}
D_MAX = 350
CHARGING_TIME = 100
trips_df = pd.DataFrame.from_dict(trips, orient='index')
print(trips_df)
## Creating Gamma and Delta
arcs = feasible_pairs(trips_df, terminals)
print(charging_stations)

######################################################## 1CS #####################################################################
recharge_1cs_arcs = feasible_recharge(trips_df, arcs, recharge=charging_station)
cs_ids = set(list(charging_station.keys()))
all_schedules_1cs = {**trips, **charging_station}
all_schs_1cs = pd.DataFrame.from_dict(all_schedules_1cs, orient='index')
all_schs_1cs['ID'] = range(len(all_schs_1cs))
durations_1cs = {idx: {'duration': all_schs_1cs.loc[idx, 'duration']} for idx in all_schs_1cs.index if idx != 0}
###################################################### CHS-SA ####################################################################
import time
start_time = time.time()
print(f"{time.ctime()}")
schedules_tab_1cs, schedules_1cs = constructiveScheduler(all_schs_1cs, arcs, recharge_1cs_arcs, cs_ids)
solution = vectorSchRepresentation(schedules_1cs)
end_time = time.time()
g_T30CS1_time = end_time - start_time
print(f"{time.ctime()}\nTime elapse to compute the solution = {(g_T30CS1_time)} seconds")
print(f"number of buses = {len(schedules_1cs)}")
print(schedules_1cs)

####### SA
print('-'*100)
print("starting simulated annealing....")
start_time = time.time()
# test_new_schedule, test_cost, test_cost_diffs, test_temp, test_it, test_costs, test_solutionspaces, test_best
new_schedule_1cs, cost_1cs, cost_diffs_1cs, temp_1cs, it_1cs, test_costs_1cs, solution_spaces_1cs, best_costs_1cs = annealing(solution, all_schs_1cs, arcs, recharge_1cs_arcs)
end_time = time.time()
T30CS1_time = (end_time-start_time) + g_T30CS1_time
print(f"{time.ctime()}\nTime elapse to compute the solution = {(T30CS1_time)} seconds")
print(f"prev_schedule = {solution} with number of buses = {len(solution)}... \nnext_schedule = {new_schedule_1cs} with number of buses = {len(new_schedule_1cs)}")

print(f"Number of buses previous = {len(schedules_1cs)}... new = {len(new_schedule_1cs)} ")
print(f"Prev: {vectorSchRepresentation(schedules_1cs)}\nNew: {new_schedule_1cs}")
print(f"Total Gap among buses=> prev = {get_total_gap(solution, all_schs_1cs, recharge_1cs_arcs)}... new = {get_total_gap(new_schedule_1cs, all_schs_1cs, recharge_1cs_arcs)}")

plt.plot(range(it_1cs), test_costs_1cs, label="current_solution")
plt.plot(range(it_1cs), best_costs_1cs, label="best_solution")
plt.xlabel("Iteration (it)")
plt.ylabel("#ofBuses(cost) + Total Gaps between Schedules")
plt.title("10Trips1ChargingStation_CHS-SA")
plt.legend(loc="upper right")
plt.savefig("30Trips1CS.png")

thirtyTrips_df = visualizeSolution(solution_spaces_1cs[1:], "30Trips1CS-CHS-SA Pareto Front", all_schs_1cs, recharge_1cs_arcs)
newdf = visualizeResult(new_schedule_1cs, all_schs_1cs, "CHS_30Trips-1CS")
trips30_df = newdf.copy(deep=True)
trips30_df['next_dep'] = trips30_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
trips30_df['difference'] = trips30_df['next_dep'] - trips30_df['arr_time']
trips30_df['difference'] = trips30_df['difference'].apply(lambda x: 0 if x < 0 else x)
trips30_soln = trips30_df.groupby(['bus_id'])['difference'].sum()
# trips30_df.to_csv("30Trips1CS.csv")
chs_30Trips1cs_IDLE_soln = trips30_df.groupby(['bus_id']).agg({'trip_id':concat_str, 'dep_time': 'first', 'arr_time': 'last', 'dep_terminal': 'first', 'arr_terminal': 'last', 'difference': 'sum'})
chs_30Trips1cs_IDLE_soln = trips30_df.groupby(['bus_id']).agg(
    trips=('trip_id', concat_str),
    numRecharge=('trip_id',countRecharge),
    numTrips=('trip_id', countTrips),
    gapTime=('difference', 'sum')
)
################################################################### 2CS #############################################################################################

## Creating Phi and Delta
recharge_arcs = feasible_recharge(trips_df, arcs, recharge=charging_stations)
cs_ids = list(charging_stations.keys())
all_schedules_2cs = {**trips, **charging_stations}
all_schs = pd.DataFrame.from_dict(all_schedules_2cs, orient='index')
all_schs['ID'] = range(len(all_schs))
durations = {idx: {'duration': all_schs.loc[idx, 'duration']} for idx in all_schs.index if idx != 0}

#### Initiate CHS-SA
start_time = time.time()
print(f"{time.ctime()}")
test_schedules_tab, test_schedules = constructiveScheduler(all_schs, arcs, recharge_arcs, set(cs_ids))
test_solution = vectorSchRepresentation(test_schedules)
end_time = time.time()
g_T30CS2_time = end_time - start_time
print(f"{time.ctime()}\nTime elapse to compute the solution = {(g_T30CS2_time)} seconds")
print(f"number of buses = {len(test_schedules)}")
print(test_schedules)
print('-'*100)
print("starting simulated annealing....")
start_time = time.time()
test_new_schedule, test_cost, test_cost_diffs, test_temp, test_it, test_costs, test_solutionspaces, test_best = annealing(test_solution, all_schs, arcs, recharge_arcs)
end_time = time.time()
T30CS2_time = (end_time-start_time) + g_T30CS2_time
print(f"{time.ctime()}\nTime elapse to compute the solution = {(T30CS2_time)} seconds")
print(f"{time.ctime()}\nTime elapse to compute the solution = {(end_time - start_time)} seconds")
print(f"prev_schedule = {test_solution} with number of buses = {len(test_solution)}... \nnext_schedule = {test_new_schedule} with number of buses = {len(test_new_schedule)}")

#### PLOTTING
plt.plot(range(test_it), test_costs, label="Current_Solution Cost")
plt.plot(range(test_it), test_best, label="Best_Solution Cost")
plt.xlabel("Iteration (it)")
plt.ylabel("#ofBuses(cost) + Total Gaps between Schedules")
plt.title("30Trips2ChargingStations_CHS-SA")
plt.legend(loc="upper right")
plt.savefig("30Trips2CSs.png")


test_ThirtyTrips_df = visualizeSolution(test_solutionspaces[1:], "30Trips2CSs-CHS-SA Pareto Front", all_schs, recharge_arcs)
newdf = visualizeResult(test_new_schedule, all_schs, "CHS_30Trips-2CS")
trips30_df = newdf.copy(deep=True)
trips30_df['next_dep'] = trips30_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
trips30_df['difference'] = trips30_df['next_dep'] - trips30_df['arr_time']
trips30_df['difference'] = trips30_df['difference'].apply(lambda x: 0 if x < 0 else x)
trips30_soln = trips30_df.groupby(['bus_id'])['difference'].sum()
trips30_df.to_csv("30Trips2CS.csv")
chs_30Trips2cs_IDLE_soln = trips30_df.groupby(['bus_id']).agg({'trip_id':concat_str, 'dep_time': 'first', 'arr_time': 'last', 'dep_terminal': 'first', 'arr_terminal': 'last', 'difference': 'sum'})
chs_30Trips2cs_IDLE_soln = trips30_df.groupby(['bus_id']).agg(
    trips=('trip_id', concat_str),
    numRecharge=('trip_id',countRecharge),
    numTrips=('trip_id', countTrips),
    gapTime=('difference', 'sum')
)
################################################################################### 54 TRIPS ###############################################################################################
trips = {
    0: {'name': 'DEPOT', 'type': 'depot', 'duration': 0, 'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
    1: {'name': 'AB1', 'type': 'trip', 'duration': 135, 'dep_time': 560, 'arr_time': 695, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
    2: {'name': 'AB2', 'type': 'trip', 'duration': 135, 'dep_time': 740, 'arr_time': 875, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
    3: {'name': 'AB3', 'type': 'trip', 'duration': 135, 'dep_time': 830, 'arr_time': 965, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
    4: {'name': 'AB4', 'type': 'trip', 'duration': 135, 'dep_time': 920, 'arr_time': 1055, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
    5: {'name': 'AB5', 'type': 'trip', 'duration': 135, 'dep_time': 980, 'arr_time': 1115, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
    6: {'name': 'AB6', 'type': 'trip', 'duration': 135, 'dep_time': 1040, 'arr_time': 1175, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
    7: {'name': 'AB7', 'type': 'trip', 'duration': 135, 'dep_time': 1100, 'arr_time': 1235, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
    8: {'name': 'AB8', 'type': 'trip', 'duration': 135, 'dep_time': 1190, 'arr_time': 1325, 'dep_term': 'Terminal A', 'arr_term': 'Terminal B'},
    9: {'name': 'AC1', 'type': 'trip', 'duration': 140, 'dep_time': 555, 'arr_time': 695, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
    10: {'name': 'AC2', 'type': 'trip', 'duration': 140, 'dep_time': 675, 'arr_time': 815, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
    11: {'name': 'AC3', 'type': 'trip', 'duration': 140, 'dep_time': 795, 'arr_time': 935, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
    12: {'name': 'AC4', 'type': 'trip', 'duration': 140, 'dep_time': 1155, 'arr_time': 1295, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
    13: {'name': 'AC5', 'type': 'trip', 'duration': 140, 'dep_time': 1215, 'arr_time': 1355, 'dep_term': 'Terminal A', 'arr_term': 'Terminal C'},
    14: {'name': 'AD1', 'type': 'trip', 'duration': 140, 'dep_time': 560, 'arr_time': 700, 'dep_term': 'Terminal A', 'arr_term': 'Terminal D'},
    15: {'name': 'AD2', 'type': 'trip', 'duration': 140, 'dep_time': 630, 'arr_time': 770, 'dep_term': 'Terminal A', 'arr_term': 'Terminal D'},
    16: {'name': 'AD3', 'type': 'trip', 'duration': 135, 'dep_time': 790, 'arr_time': 925, 'dep_term': 'Terminal A', 'arr_term': 'Terminal D'},
    17: {'name': 'AD4', 'type': 'trip', 'duration': 140, 'dep_time': 970, 'arr_time': 1110, 'dep_term': 'Terminal A', 'arr_term': 'Terminal D'},
    18: {'name': 'AD5', 'type': 'trip', 'duration': 140, 'dep_time': 1120, 'arr_time': 1240, 'dep_term': 'Terminal A', 'arr_term': 'Terminal D'},
    19: {'name': 'BA1', 'type': 'trip', 'duration': 135, 'dep_time': 540, 'arr_time': 675, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
    20: {'name': 'BA2', 'type': 'trip', 'duration': 135, 'dep_time': 630, 'arr_time': 765, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
    21: {'name': 'BA3', 'type': 'trip', 'duration': 135, 'dep_time': 720, 'arr_time': 855, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
    22: {'name': 'BA4', 'type': 'trip', 'duration': 135, 'dep_time': 810, 'arr_time': 945, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
    23: {'name': 'BA5', 'type': 'trip', 'duration': 135, 'dep_time': 900, 'arr_time': 1035, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
    24: {'name': 'BA6', 'type': 'trip', 'duration': 135, 'dep_time': 990, 'arr_time': 1125, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
    25: {'name': 'BA7', 'type': 'trip', 'duration': 135, 'dep_time': 1080, 'arr_time': 1215, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
    26: {'name': 'BA8', 'type': 'trip', 'duration': 135, 'dep_time': 1140, 'arr_time': 1275, 'dep_term': 'Terminal B', 'arr_term': 'Terminal A'},
    27: {'name': 'BC1', 'type': 'trip', 'duration': 140, 'dep_time': 435, 'arr_time': 570, 'dep_term': 'Terminal B', 'arr_term': 'Terminal C'},
    28: {'name': 'BC2', 'type': 'trip', 'duration': 140, 'dep_time': 595, 'arr_time': 735, 'dep_term': 'Terminal B', 'arr_term': 'Terminal C'},
    29: {'name': 'BC3', 'type': 'trip', 'duration': 120, 'dep_time': 800, 'arr_time': 920, 'dep_term': 'Terminal B', 'arr_term': 'Terminal C'},
    30: {'name': 'BC4', 'type': 'trip', 'duration': 130, 'dep_time': 950, 'arr_time': 1130, 'dep_term': 'Terminal B', 'arr_term': 'Terminal C'},
    31: {'name': 'CA1', 'type': 'trip', 'duration': 140, 'dep_time': 570, 'arr_time': 710, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
    32: {'name': 'CA2', 'type': 'trip', 'duration': 140, 'dep_time': 740, 'arr_time': 880, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
    33: {'name': 'CA3', 'type': 'trip', 'duration': 140, 'dep_time': 860, 'arr_time': 1000, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
    34: {'name': 'CA4', 'type': 'trip', 'duration': 140, 'dep_time': 920, 'arr_time': 1060, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
    35: {'name': 'CA5', 'type': 'trip', 'duration': 140, 'dep_time': 1040, 'arr_time': 1180, 'dep_term': 'Terminal C', 'arr_term': 'Terminal A'},
    36: {'name': 'CB1', 'type': 'trip', 'duration': 140, 'dep_time': 540, 'arr_time': 680, 'dep_term': 'Terminal C', 'arr_term': 'Terminal B'},
    37: {'name': 'CB2', 'type': 'trip', 'duration': 140, 'dep_time': 645, 'arr_time': 785, 'dep_term': 'Terminal C', 'arr_term': 'Terminal B'},
    38: {'name': 'CB3', 'type': 'trip', 'duration': 120, 'dep_time': 900, 'arr_time': 1020, 'dep_term': 'Terminal C', 'arr_term': 'Terminal B'},
    39: {'name': 'CB4', 'type': 'trip', 'duration': 130, 'dep_time': 950, 'arr_time': 1080, 'dep_term': 'Terminal C', 'arr_term': 'Terminal B'},
    40: {'name': 'CD1', 'type': 'trip', 'duration': 140, 'dep_time': 450, 'arr_time': 585, 'dep_term': 'Terminal C', 'arr_term': 'Terminal D'},
    41: {'name': 'CD2', 'type': 'trip', 'duration': 135, 'dep_time': 550, 'arr_time': 685, 'dep_term': 'Terminal C', 'arr_term': 'Terminal D'},
    42: {'name': 'CD3', 'type': 'trip', 'duration': 135, 'dep_time': 600, 'arr_time': 735, 'dep_term': 'Terminal C', 'arr_term': 'Terminal D'},
    43: {'name': 'CD4', 'type': 'trip', 'duration': 135, 'dep_time': 765, 'arr_time': 900, 'dep_term': 'Terminal C', 'arr_term': 'Terminal D'},
    44: {'name': 'CD5', 'type': 'trip', 'duration': 120, 'dep_time': 850, 'arr_time': 970, 'dep_term': 'Terminal C', 'arr_term': 'Terminal D'},
    45: {'name': 'DA1', 'type': 'trip', 'duration': 140, 'dep_time': 720, 'arr_time': 860, 'dep_term': 'Terminal D', 'arr_term': 'Terminal A'},
    46: {'name': 'DA2', 'type': 'trip', 'duration': 135, 'dep_time': 750, 'arr_time': 885, 'dep_term': 'Terminal D', 'arr_term': 'Terminal A'},
    47: {'name': 'DA3', 'type': 'trip', 'duration': 140, 'dep_time': 800, 'arr_time': 940, 'dep_term': 'Terminal D', 'arr_term': 'Terminal A'},
    48: {'name': 'DA4', 'type': 'trip', 'duration': 140, 'dep_time': 1000, 'arr_time': 1140, 'dep_term': 'Terminal D', 'arr_term': 'Terminal A'},
    49: {'name': 'DA5', 'type': 'trip', 'duration': 140, 'dep_time': 1200, 'arr_time': 1340, 'dep_term': 'Terminal D', 'arr_term': 'Terminal A'},
    50: {'name': 'DC1', 'type': 'trip', 'duration': 140, 'dep_time': 600, 'arr_time': 740, 'dep_term': 'Terminal D', 'arr_term': 'Terminal C'},
    51: {'name': 'DC2', 'type': 'trip', 'duration': 120, 'dep_time': 765, 'arr_time': 885, 'dep_term': 'Terminal D', 'arr_term': 'Terminal C'},
    52: {'name': 'DC3', 'type': 'trip', 'duration': 120, 'dep_time': 780, 'arr_time': 900, 'dep_term': 'Terminal D', 'arr_term': 'Terminal C'},
    53: {'name': 'DC4', 'type': 'trip', 'duration': 135, 'dep_time': 965, 'arr_time': 1100, 'dep_term': 'Terminal D', 'arr_term': 'Terminal C'},
    54: {'name': 'DC5', 'type': 'trip', 'duration': 135, 'dep_time': 975, 'arr_time': 1110, 'dep_term': 'Terminal D', 'arr_term': 'Terminal C'}
}
charging_station = {
    55: {'name': 'CS1', 'type': 'cs', 'duration': 60,  'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
}

charging_stations = {
    55: {'name': 'CS1', 'type': 'cs', 'duration': 60,  'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
    56: {'name': 'CS2', 'type': 'cs', 'duration': 75,  'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
    57: {'name': 'CS3', 'type': 'cs', 'duration': 70,  'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
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
deadheads = {
    'Terminal A': {'Terminal B': 25, 'Terminal C': 30, 'Terminal D': 35},
    'Terminal B': {'Terminal A': 25, 'Terminal C': 28, 'Terminal D': 40},
    'Terminal C': {'Terminal A': 30, 'Terminal B': 28, 'Terminal D': 33},
    'Terminal D': {'Terminal A': 35, 'Terminal B': 40, 'Terminal C': 33},
}
D_MAX = 350
CHARGING_TIME = 100

############################################### 1 CS #####################################################################################################
print("Starting the process for 54Trips at Single Charging Stations......")
trips_df = pd.DataFrame.from_dict(trips, orient='index')
arcs = feasible_pairs(trips_df, terminals)
## Creating Phi and Delta
recharge_arcs = feasible_recharge(trips_df, arcs, recharge=charging_station)
cs_ids = list(charging_station.keys())
all_schedules = {**trips, **charging_station}
all_schs = pd.DataFrame.from_dict(all_schedules, orient='index')
all_schs['ID'] = range(len(all_schs))
durations = {idx: {'duration': all_schs.loc[idx, 'duration']} for idx in all_schs.index if idx != 0}

ax = drawBusTrips(all_schs[1:55][['name','dep_time', 'arr_time', 'dep_term', 'arr_term']])


start_time = time.time()
print(f"{time.ctime()}")
# max_it = 100000
schedules_54Trips1cs_tab, schedules_54trips1cs = constructiveScheduler(all_schs, arcs, recharge_arcs, set(cs_ids))
test_solution = vectorSchRepresentation(schedules_54trips1cs)
end_time = time.time()
g_T54CS1_time = end_time - start_time
print(f"{time.ctime()}\nTime elapse to compute the solution = {(g_T54CS1_time)} seconds")
print(f"number of buses = {len(test_solution)}")
print(schedules_54trips1cs)
print('-'*100)

print("starting simulated annealing....")
start_time = time.time()
new_54_1_schedule, new_54_cost, new_54_1_cost_diffs, new_54_1_temp, new_54_1_it, new_54_1_costs, new_54_1_solutionspaces, new_54_1_best = annealing(test_solution, all_schs, arcs, recharge_arcs)
end_time = time.time()
T54CS1_time = (end_time-start_time)+g_T54CS1_time
print(f"{time.ctime()}\nTime elapse to compute the solution = {(T54CS1_time)} seconds")
print(f"prev_schedule = {test_solution} with number of buses = {len(test_solution)}... \nnext_schedule = {new_54_1_schedule} with number of buses = {len(new_54_1_schedule)}")

plt.plot(range(new_54_1_it), new_54_1_costs, label="Current_Solution Cost")
plt.plot(range(new_54_1_it), new_54_1_best, label="Best_Solution Cost")
# plt.gca().invert_xaxis()
plt.xlabel("Iteration (it)")
# plt.ylabel("[weighted_sum] #ofBuses + Total Gap Time (cost)")
plt.ylabel("Priority Normalized #ofBuses & Next Total Gaps(cost)")
plt.title("54Trips1ChargingStation_CHS-SA")
plt.legend(loc="upper right")
plt.savefig("54Trips1CS.png")

fiftyfourTrips_df = visualizeSolution(new_54_1_solutionspaces[1:], "54Trips1CS-CHS-SA Pareto Front", all_schs, recharge_arcs)

newdf = visualizeResult(new_54_1_schedule, all_schs, "CHS_54Trips-1CS")

trips54_1_df = newdf.copy(deep=True)
trips54_1_df['next_dep'] = trips54_1_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
trips54_1_df['difference'] = trips54_1_df['next_dep'] - trips54_1_df['arr_time']
trips54_1_df['difference'] = trips54_1_df['difference'].apply(lambda x: 0 if x < 0 else x)
trips54_1_soln = trips54_1_df.groupby(['bus_id'])['difference'].sum()

chs_54Trips1cs_IDLE_soln = trips54_1_df.groupby(['bus_id']).agg(
    trips=('trip_id', concat_str),
    numRecharge=('trip_id',countRecharge),
    numTrips=('trip_id', countTrips),
    gapTime=('difference', 'sum')
)
chs_54Trips1cs_IDLE_soln.sort_values(['gapTime'], ascending=False)
print("Ending the process for 54Trips and 1 CS......")
################################################################ 2 CS ##################################################################################
trips_df = pd.DataFrame.from_dict(trips, orient='index')
arcs = feasible_pairs(trips_df, terminals)
## Creating Phi and Delta
recharge_arcs = feasible_recharge(trips_df, arcs, recharge=charging_stations)
cs_ids = list(charging_stations.keys())
all_schedules = {**trips, **charging_stations}
all_schs = pd.DataFrame.from_dict(all_schedules, orient='index')
all_schs['ID'] = range(len(all_schs))
durations = {idx: {'duration': all_schs.loc[idx, 'duration']} for idx in all_schs.index if idx != 0}

ax = drawBusTrips(all_schs[1:55][['name','dep_time', 'arr_time', 'dep_term', 'arr_term']])


start_time = time.time()
print(f"{time.ctime()}")
# max_it = 100000
test_schedules_tab, test_schedules = constructiveScheduler(all_schs, arcs, recharge_arcs, set(cs_ids))
test_solution = vectorSchRepresentation(test_schedules)
end_time = time.time()
g_T54CS3_time = end_time - start_time
print(f"{time.ctime()}\nTime elapse to compute the solution = {(g_T54CS3_time)} seconds")
print(f"number of buses = {len(test_schedules)}")
print(test_schedules)
print('-'*100)

print("starting simulated annealing....")
start_time = time.time()
new_54_3_schedule, new_54_3_cost, new_54_3_cost_diffs, new_54_3_temp, new_54_3_it, new_54_3_costs, new_54_3_solutionspaces, new_54_3_best = annealing(test_solution, all_schs, arcs, recharge_arcs)
end_time = time.time()
T54CS3_time = (end_time - start_time) + g_T54CS3_time
print(f"{time.ctime()}\nTime elapse to compute the solution = {(T54CS3_time)} seconds")
print(f"prev_schedule = {test_solution} with number of buses = {len(test_solution)}... \nnext_schedule = {new_54_3_schedule} with number of buses = {len(new_54_3_schedule)}")

plt.plot(range(new_54_3_it), new_54_3_costs, label="Current_Solution Cost")
plt.plot(range(new_54_3_it), new_54_3_best, label="Best_Solution Cost")
# plt.gca().invert_xaxis()
plt.xlabel("Iteration (it)")
# plt.ylabel("[weighted_sum] #ofBuses + Total Gap Time (cost)")
plt.ylabel("Priority Normalized #ofBuses & Next Total Gaps(cost)")
plt.title("54Trips3ChargingStations_CHS-SA")
plt.legend(loc="upper right")
plt.savefig("54Trips3CSs.png")

fiftyfourTrips_df = visualizeSolution(new_54_3_solutionspaces[1:], "54Trips3CSs-CHS-SA Pareto Front", all_schs, recharge_arcs)

newdf = visualizeResult(new_54_3_schedule, all_schs, "CHS_54Trips-3CS")

trips54_3_df = newdf.copy(deep=True)
trips54_3_df['next_dep'] = trips54_3_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
trips54_3_df['difference'] = trips54_3_df['next_dep'] - trips54_3_df['arr_time']
trips54_3_df['difference'] = trips54_3_df['difference'].apply(lambda x: 0 if x < 0 else x)
trips54_3_soln = trips54_3_df.groupby(['bus_id'])['difference'].sum()

chs_54Trips3cs_IDLE_soln = trips54_3_df.groupby(['bus_id']).agg(
    trips=('trip_id', concat_str),
    numRecharge=('trip_id',countRecharge),
    numTrips=('trip_id', countTrips),
    gapTime=('difference', 'sum')
)
chs_54Trips3cs_IDLE_soln.sort_values(['gapTime'], ascending=False)


cs_ids = ["CS1", "CS2", "CS3"]
def countRecharge(x):
    trips = x.split(",")
    return len(list(filter(lambda x: x in cs_ids, trips)))
chs_10Trips1cs_IDLE_soln['numRecharge'] = chs_10Trips1cs_IDLE_soln.trips.apply(countRecharge)
chs_10Trips2cs_IDLE_soln['numRecharge'] = chs_10Trips2cs_IDLE_soln.trips.apply(countRecharge)

chs_30Trips1cs_IDLE_soln['numRecharge'] = chs_30Trips1cs_IDLE_soln.trips.apply(countRecharge)
chs_30Trips2cs_IDLE_soln['numRecharge'] = chs_30Trips2cs_IDLE_soln.trips.apply(countRecharge)

chs_54Trips1cs_IDLE_soln['numRecharge'] = chs_54Trips1cs_IDLE_soln.trips.apply(countRecharge)
chs_54Trips3cs_IDLE_soln['numRecharge'] = chs_54Trips3cs_IDLE_soln.trips.apply(countRecharge)

test = chs_10Trips1cs_IDLE_soln.describe().loc['mean']
test['time_to_best_soln'] = T10CS1_time
test = test.to_frame().rename(columns={"mean":"CHS_10Trips1CS"})

test2 = chs_10Trips2cs_IDLE_soln.describe().loc['mean']
test2['numBuses'] = chs_10Trips2cs_IDLE_soln.describe().loc['count','gapTime']
test2['time_to_best_soln'] = T10CS2_time
test2 = test2.to_frame().rename(columns={"mean":"CHS_10Trips2CS"})

result_10Trips = pd.concat([test, test2], axis=1)

test = chs_30Trips1cs_IDLE_soln.describe().loc['mean']
test['numBuses'] = chs_30Trips1cs_IDLE_soln.describe().loc['count','gapTime']
test['time_to_best_soln'] = T30CS1_time
test = test.to_frame().rename(columns={"mean":"CHS_30Trips1CS"})

test2 = chs_30Trips2cs_IDLE_soln.describe().loc['mean']
test2['numBuses'] = chs_30Trips2cs_IDLE_soln.describe().loc['count','gapTime']
test2['time_to_best_soln'] = T30CS2_time
test2 = test2.to_frame().rename(columns={"mean":"CHS_30Trips2CS"})

result_30Trips = pd.concat([test, test2], axis=1)

test = chs_54Trips1cs_IDLE_soln.describe().loc['mean']
test['numBuses'] = chs_54Trips1cs_IDLE_soln.describe().loc['count','gapTime']
test['time_to_best_soln'] = T54CS1_time
test = test.to_frame().rename(columns={"mean":"CHS_54Trips1CS"})

test2 = chs_54Trips3cs_IDLE_soln.describe().loc['mean']
test2['numBuses'] = chs_54Trips3cs_IDLE_soln.describe().loc['count','gapTime']
test2['time_to_best_soln'] = T54CS3_time
test2 = test2.to_frame().rename(columns={"mean":"CHS_54Trips3CS"})

result_54Trips = pd.concat([test, test2], axis=1)


# test.columns = pd.MultiIndex.from_product([["CHS_30Trips2CS"], test.columns])
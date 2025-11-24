# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 01:05:10 2025

@author: gozalid
"""
from collections import Counter 
def count_energy(x, temp):
    trips = x.to_list()
    energy = 0
    #for idx, x in temp.iterrows():
    for trip in trips:
        if temp.loc[temp['ID'] == trip].iloc[0]['type'] == 'trip':
            energy += temp.loc[temp['ID'] == trip].iloc[0]['duration']
        else:
            energy = 0
    return energy

import datetime
path = ".\\"
def feasible_recharge2(data, deadheads, recharge, terminals):
    bus_data = data.copy(deep=True)
    bus_data = bus_data.reset_index()
    bus_data.rename(columns={'index': 'trip_id'}, inplace=True)
    bus_data.index = np.arange(0, len(data))

    connected_trips = (
        lambda pair: bus_data.loc[pair[1], "dep_time"] - bus_data.loc[pair[0], "arr_time"]
    )
    # Recharging Feasibility becomes a spatial dependent variable with the sum of the deadheads and the duration to recharge.
    recharging_costs = (
        lambda pair: (recharge[pair[2]]['dep_term'] == "-" and 
                      connected_trips(pair)>=(CHARGING_TIME + int(deadheads[bus_data.loc[pair[0], "arr_term"]]["CS1"]))) or 
                      (recharge[pair[2]]['dep_term'] == bus_data.loc[pair[1], "dep_term"] and bus_data.loc[pair[0], "arr_term"] == bus_data.loc[pair[1], "dep_term"] and
                       connected_trips(pair) >= (CHARGING_TIME + int(deadheads[bus_data.loc[pair[1],"dep_term"]][recharge[pair[2]]['name']]))) 
                       or (recharge[pair[2]]['dep_term'] == bus_data.loc[pair[1], "dep_term"] and bus_data.loc[pair[0], "arr_term"] != bus_data.loc[pair[1], "dep_term"] and
                       connected_trips(pair) >= (CHARGING_TIME + int(deadheads[bus_data.loc[pair[1],"dep_term"]][recharge[pair[2]]['name']])))
    )
    ### duration (delta)
    duration_trips = (
        lambda pair: deadheads[bus_data.loc[pair[1], "dep_term"]][recharge[pair[2]]['name']] + CHARGING_TIME
    )
    ## feasible (phi)
    pairs = filter(
        lambda pair: (bus_data.loc[pair[0], 'type'] != 'depot' and bus_data.loc[pair[1], "type"] != 'depot') and (connected_trips(pair) >= terminals[bus_data.loc[pair[1], "arr_term"]]["max_interval"]) and (recharging_costs(pair)),
        [(i,j,k) for k in recharge.keys() for i in bus_data.index for j in bus_data.index if i != j],
    )
    return {pair: {'duration': duration_trips(pair)} for pair in pairs}

      
SELECTED_DATE = datetime.date(2024, 7, 26)
date_name = str(SELECTED_DATE.day)+"-"+str(SELECTED_DATE.month)+"-"+str(SELECTED_DATE.year)
month_name = SELECTED_DATE.strftime("%B") + "_" +str(SELECTED_DATE.year)
print(date_name)
month_name = month_name.lower()


from zipfile import ZipFile
import pandas as pd

filename = f"Y:\\Data\\GTFS_NEW\\GTFS Static\\GTFS Static {date_name}.zip"
# date_name = filename.split(" ")[-1]
print(filename)
zip_file = ZipFile(filename)
trips = pd.read_csv(zip_file.open('trips.txt'))
stop_times = pd.read_csv(zip_file.open('stop_times.txt'))
def date_parser(x, date):
    """
    Convert date and time in string to datetime format.
    """
    return pd.to_datetime(date) + pd.to_timedelta(x)
for name in ['arrival_time', 'departure_time']:
    stop_times[name] = stop_times[name].apply(date_parser, date=SELECTED_DATE)


static_schedule = pd.read_csv(f"{path}Static_Timetable_{date_name}.csv")
static_schedule.head()
static_schedule = static_schedule.drop(['Unnamed: 0'], axis=1)
def hasNumber(x):
    return any(char.isdigit() for char in x)

def categorizeModes(x):
    ferries = ['NHAM', 'UQSL', 'TNRF', 'NTHQ', 'SYDS', 'SMBI']
    if x in ferries:
        return "FERRY"
    elif hasNumber(x):
        return "BUS"
    else:
        return "RAIL"
static_schedule['modes'] = static_schedule.pt_num.apply(categorizeModes)
static_schedule.head()
high_frequency_buses = ["555"]#['60', '61', '100', '111' ,'120', '130' ,'140', '150', '180', '196' , '199' ,'200' ,'222', '330', '333' ,'340', '345', '385', '412', '444', '555']
electric_buses = ['40', '50']
bus_schedules = static_schedule.loc[(static_schedule.modes == 'BUS') & (static_schedule.pt_num.isin(high_frequency_buses))]
bus_schedules.info()
trips = bus_schedules[['trip_id', 'pt_num', 'shape_id', 'direction_id', 'start_minutes_day', 'end_minutes_day', 'sch_duration_min', 'stop_sequence']]
# trips.drop(['index'])
trips = trips.reset_index()
static_gtfs = trips.copy(deep=True)
static_gtfs['dep_term'] = static_gtfs[['stop_sequence', 'direction_id']].apply(lambda x: f"Terminal_{x['stop_sequence']}" if x['direction_id'] else "Terminal_1", axis=1)
static_gtfs['arr_term'] = static_gtfs[['stop_sequence', 'direction_id']].apply(lambda x: "Terminal_1" if x['direction_id'] else f"Terminal_{x['stop_sequence']}", axis=1)
static_gtfs['type'] = 'trip'
static_gtfs['dep_time'] = static_gtfs['start_minutes_day']
static_gtfs['arr_time'] = static_gtfs['end_minutes_day']
static_gtfs['duration'] = static_gtfs.apply(lambda x: x['arr_time'] - x['dep_time'], axis=1)
# static_gtfs = static_gtfs.loc[(static_gtfs.dep_time >= 360) & (static_gtfs.arr_time < 601)]
terms = list(static_gtfs['dep_term'].unique())+list(static_gtfs['arr_term'].unique())
print(terms)
terminals = Counter(terms)
max_term = max(terminals, key=terminals.get)
min_term = min(terminals, key=terminals.get)
NTrips = len(static_gtfs) #trips
NTerms = len(terminals)#terminals
NCS = 2 #recharging station
CHARGING_TIME = 30 #minutes
NRechargeCycle = 3 #cycle
D_MAX = 120 #minutes of operation
DEPOT = 1 #depot
len_df = NTrips
columns = ['trip_id', 'type', 'shape_id', 'direction_id', 'duration', 'dep_time', 'arr_time', 'dep_term', 'arr_term']
cs_ids = ["CS1", "CS2", "CS3"]
len_df = NTrips
columns = ['trip_id', 'type', 'shape_id', 'direction_id', 'duration', 'dep_time', 'arr_time', 'dep_term', 'arr_term']
depot = {
    0: {'trip_id': 'DEPOT', 'type': 'depot', 'shape_id': '-', 'duration': 0,  'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
}
charging_station = {
    (len_df)+1: {'trip_id': cs_ids[0], 'type': 'cs', 'shape_id': '-', 'duration': 30,  'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
}
charging_stations = {
    (len_df)+1: {'trip_id': cs_ids[0], 'type': 'cs', 'shape_id': '-', 'duration': 30,  'dep_time': 0, 'arr_time': 1440, 'dep_term': '-', 'arr_term': '-'},
    (len_df)+2: {'trip_id': cs_ids[1], 'type': 'cs', 'shape_id': '-', 'duration': 30,  'dep_time': 0, 'arr_time': 1440, 'dep_term': f'{max_term}', 'arr_term': f'{max_term}'},
    (len_df)+3: {'trip_id': cs_ids[2], 'type': 'cs', 'shape_id': '-', 'duration': 30,  'dep_time': 0, 'arr_time': 1440, 'dep_term': f'{min_term}', 'arr_term': f'{min_term}'},
}
static_gtfs = static_gtfs[columns]
depot_df = pd.DataFrame.from_dict(depot, orient='index')
cs_df = pd.DataFrame.from_dict(charging_stations, orient='index')

num_cs = len(cs_df)
dur_terminals = {}
for i in terminals.keys():
    dur_terminals[i] = {'max_interval': 15}
deadheads = {}
for i in terminals.keys():
    temp = {}
    for j, val in charging_stations.items():
        name_term = val['dep_term']
        name_cs = val['trip_id']
        if name_term != "-":
            if i != name_term:
                print(f"Name {i} {(int((name_term.split('Terminal_')[1]))%10)}")
                temp[name_cs] = 5 * (int((name_term.split("Terminal_")[1]))%10)
            else:
                temp[name_cs] = 5
        else:
            temp[name_cs] = 15
    print(f"{i} name {temp}")
    deadheads[i] = temp
print(f"ebris - {terminals}")

arcs = feasible_pairs(static_gtfs, dur_terminals)
temp = pd.concat([depot_df, static_gtfs], ignore_index=True)
temp = pd.concat([temp, cs_df], ignore_index=True)
temp['ID'] = [x for x in range(len(temp))]
full_recharge_arcs = feasible_recharge2(temp.loc[temp.type != "cs"], deadheads, recharge=charging_stations, terminals=dur_terminals)

# ######
# D_MAX=180
# scheduler = ImprovedEVScheduler(max_iterations=100, d_max=D_MAX)
# ###################################################### 1CS #########################################################################
# recharge_arcs = {key: val for key, val in full_recharge_arcs.items() if key[2] == list(cs_ids)[0]}
# eb_gtfs = temp.loc[(temp.type != "cs") | (temp.ID == list(cs_ids)[0])]
# start_time = time.time()
# print(f"{time.ctime()}")
# schedules_tab, schedules_1 = scheduler.constructive_scheduler_improved(eb_gtfs, arcs, recharge_arcs, cs_ids)
# solution = vectorSchRepresentation(schedules_1)
# end_time = time.time()
# g_TCS3_time = end_time - start_time
# print(f"{time.ctime()}\nTime elapse to compute the solution = {(g_TCS3_time)} seconds")
# print(f"number of buses = {len(schedules_1)}")
# print(schedules_1)
# print('-'*100)
# print("starting simulated annealing....")
# start_time = time.time()
# # test_new_schedule, test_cost, test_cost_diffs, test_temp, test_it, test_costs, test_solutionspaces, test_best
# new_schedule_1, cost_1, cost_diffs_1, temp_1, it_1, costs_1, solution_spaces_1, best_costs_1 = annealing(solution, eb_gtfs, arcs, recharge_arcs, D_MAX=D_MAX)
# end_time = time.time()
# T100CS3_time = (end_time-start_time) + g_TCS3_time
# print(f"{time.ctime()}\nTime elapse to compute the solution = {T100CS3_time} seconds")
# print(f"prev_schedule = {solution} with number of buses = {len(solution)}... \nnext_schedule = {new_schedule_1} with number of buses = {len(new_schedule_1)}")
# print(f"Number of buses previous = {len(solution)}... new = {len(new_schedule_1)} ")
# print(f"Prev: {solution}\nNew: {new_schedule_1}")
# print(f"Total Gap among buses=> prev = {get_total_gap(solution, eb_gtfs, recharge_arcs)}... new = {get_total_gap(new_schedule_1, eb_gtfs, recharge_arcs)}")

# fig1, ax1 = plt.subplots()
# ax1.plot(range(len(costs_1)), costs_1, label="Current_Solution Cost")
# ax1.plot(range(len(best_costs_1)), best_costs_1, label="Best_Solution Cost")
# ax1.set_xlabel("Iteration (it)")
# ax1.set_ylabel("Priority Normalized #ofBuses & Next Total Gaps(cost)")
# ax1.set_title("90Trips1ChargingStation_CHS-SA")
# ax1.legend(loc="upper right")
# fig1.savefig("HighFrequencyBuses26Aug24Trips1cs1Depot.png")

# HundredTrips_1cs_df = visualizeSolution(solution_spaces_1[1:], "HighFrequencyBuses26Aug241CS1D-CHS-SA Pareto Front", eb_gtfs, recharge_arcs)
# newdf_1cs = visualizeResult(new_schedule_1, eb_gtfs, "HighFrequencyBuses26Aug24-1CS")

# tripsdf_1cs = newdf_1cs.groupby('bus_id', group_keys=False).apply(apply_custom_shift)
# tripsdf_1cs['next_dep'] = tripsdf_1cs.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
# tripsdf_1cs['difference'] = tripsdf_1cs['next_dep'] - tripsdf_1cs['arr_time']
# tripsdf_1cs['difference'] = tripsdf_1cs['difference'].apply(lambda x: 0 if x < 0 else x)
# tripssoln_1cs = tripsdf_1cs.groupby(['bus_id'])['difference'].sum()

# from functools import partial
# # Create a partial function with fixed arguments
# energy_agg_func = partial(count_energy, temp=eb_gtfs)

# chs_300Trips1cs_IDLE_soln = tripsdf_1cs.groupby(['bus_id']).agg(
#     energy=('trip_id', energy_agg_func),
#     trips=('trip_id', concat_str),
#     numRecharge=('trip_id',countRecharge),
#     numTrips=('trip_id', countTrips),
#     gapTime=('difference', 'sum')
# )
# chs_300Trips1cs_IDLE_soln.sort_values(['gapTime'], ascending=False)
# chs_300Trips1cs_IDLE_soln.to_csv(f"HighFrequencyBuses26Aug24_{date_name}_1cs.csv")

# ###################################################### 3CS #########################################################################
# static_gtfs = temp.copy(deep=True)
# recharge_arcs = copy.deepcopy(full_recharge_arcs)
# start_time = time.time()
# print(f"{time.ctime()}")
# schedules_tab, schedules_3 = scheduler.constructive_scheduler_improved(static_gtfs, arcs, recharge_arcs, cs_ids)
# solution = vectorSchRepresentation(schedules_3)
# end_time = time.time()
# g_TCS3_time = end_time - start_time
# print(f"{time.ctime()}\nTime elapse to compute the solution = {(g_TCS3_time)} seconds")
# print(f"number of buses = {len(schedules_3)}")
# print(schedules_3)
# print('-'*100)
# print("starting simulated annealing....")
# start_time = time.time()
# # test_new_schedule, test_cost, test_cost_diffs, test_temp, test_it, test_costs, test_solutionspaces, test_best
# new_schedule_3, cost_3, cost_diffs_3, temp_3, it_3, costs_3, solution_spaces_3, best_costs_3 = annealing(solution, static_gtfs, arcs, recharge_arcs, D_MAX=D_MAX)
# end_time = time.time()
# T100CS3_time = (end_time-start_time) + g_TCS3_time
# print(f"{time.ctime()}\nTime elapse to compute the solution = {T100CS3_time} seconds")
# print(f"prev_schedule = {solution} with number of buses = {len(solution)}... \nnext_schedule = {new_schedule_3} with number of buses = {len(new_schedule_3)}")
# print(f"Number of buses previous = {len(solution)}... new = {len(new_schedule_3)} ")
# print(f"Prev: {solution}\nNew: {new_schedule_3}")
# print(f"Total Gap among buses=> prev = {get_total_gap(solution, static_gtfs, recharge_arcs)}... new = {get_total_gap(new_schedule_3, static_gtfs, recharge_arcs)}")

# fig1, ax1 = plt.subplots()
# ax1.plot(range(len(costs_3)), costs_3, label="Current_Solution Cost")
# ax1.plot(range(len(best_costs_3)), best_costs_3, label="Best_Solution Cost")
# ax1.set_xlabel("Iteration (it)")
# ax1.set_ylabel("Priority Normalized #ofBuses & Next Total Gaps(cost)")
# ax1.set_title("90Trips3ChargingStations_CHS-SA")
# ax1.legend(loc="upper right")
# fig1.savefig("HighFrequencyBuses26Aug24Trips1cs1Depot.png")

# HundredTrips_3cs_df = visualizeSolution(solution_spaces_3[1:], "HighFrequencyBuses26Aug243CS1D-CHS-SA Pareto Front", static_gtfs, recharge_arcs)
# newdf_3cs = visualizeResult(new_schedule_3, static_gtfs, "CHS_Trips-3CS")

# tripsdf_3cs = newdf_3cs.groupby('bus_id', group_keys=False).apply(apply_custom_shift)
# tripsdf_3cs['next_dep'] = tripsdf_3cs.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
# tripsdf_3cs['difference'] = tripsdf_3cs['next_dep'] - tripsdf_3cs['arr_time']
# tripsdf_3cs['difference'] = tripsdf_3cs['difference'].apply(lambda x: 0 if x < 0 else x)
# tripssoln_3cs = tripsdf_3cs.groupby(['bus_id'])['difference'].sum()
# from functools import partial
# # Create a partial function with fixed arguments
# energy_agg_func = partial(count_energy, temp=eb_gtfs)

# chs_300Trips3cs_IDLE_soln = tripsdf_1cs.groupby(['bus_id']).agg(
#     energy=('trip_id', energy_agg_func),
#     trips=('trip_id', concat_str),
#     numRecharge=('trip_id',countRecharge),
#     numTrips=('trip_id', countTrips),
#     gapTime=('difference', 'sum')
# )
# chs_300Trips3cs_IDLE_soln.sort_values(['gapTime'], ascending=False)
# chs_300Trips3cs_IDLE_soln.to_csv(f"Static_e_Schedule_40_50_{date_name}_3cs.csv")


# import pandas as pd
# import plotly.express as px
# df = newdf_3cs.copy(deep=True)
# # Assume df is your dataframe
# def minutes_to_time(minutes):
#     return pd.to_datetime(minutes, unit="m", origin=pd.Timestamp("2023-01-01"))

# df["dep_time_dt"] = df["dep_time"].apply(minutes_to_time)
# df["arr_time_dt"] = df["arr_time"].apply(minutes_to_time)
# fig = px.timeline(
#     df,
#     x_start="dep_time_dt",
#     x_end="arr_time_dt",
#     y="bus_id",
#     color="trip_id",
#     title="GTFS_Bus40&50_600Trips_3cs_RealTime",
#     labels={"dep_time_dt": "Departure Time", "arr_time_dt": "Arrival Time", "bus_id": "Bus ID"}
# )

# # Reverse the y-axis so Bus 1 is at the top
# fig.update_yaxes(autorange="reversed")

# # Improve x-axis formatting (show only HH:MM)
# fig.update_xaxes(
#     tickformat="%H:%M",
#     title="Time of Day"
# )

# # Save to HTML
# fig.write_html("GTFS_Bus40&50_600Trips_3cs_RealTime.html")

# fig.show()
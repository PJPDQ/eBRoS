# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:53:35 2024

@author: gozalid
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
pd.options.display.max_columns = 2000
filepath = r".\\MIP-"
saveFigDir = f"{filepath}Results\\Figures\\"
def countRecharge(x):
    trips = x.to_list()
    return len(list(filter(lambda x: x in cs_ids,trips)))

def countTrips(x):
    trips = x.to_list()
    return len(set(trips) - set(cs_ids))

def concat_str(x):
    return ','.join(x)

def visualize(results, all_sch, title):
    
    schedule = pd.DataFrame(results)
    trips = sorted(list(schedule['trip_id'].unique()))
    buses = sorted(list(schedule['bus_id'].unique()))
    makespan = schedule['arr_time'].max()
    def check_single_num(x, hour):
        if hour:
            return f'0{x}' if len(x) < 2 else f'{x}'
        else:
            return f'0{x}' if len(x) < 2 else f'{x}'
    def minute_to_str(x):
        if type(x) == int:
            hour = f"{x//60}"
            minute = f"{x%60}"
            return f'{check_single_num(hour, 1)}:{check_single_num(minute, 0)}' 
        else:
            return x.strftime("%H:%M:%S")
    schedule['dep_str'] = schedule['dep_time'].apply(minute_to_str)
    schedule['arr_str'] = schedule['arr_time'].apply(minute_to_str)
    
    bar_style = {'alpha':1.0, 'lw':25, 'solid_capstyle':'butt'}
    text_style = {'color':'black', 'weight':'bold', 'ha':'center', 'va':'center', 'fontsize': '8'}
    small_text_style = {'color':'black', 'weight':'bold', 'ha':'center', 'va':'center', 'fontsize':'5'}
    colors = mpl.cm.tab10.colors + mpl.cm.tab20.colors + mpl.cm.tab20b.colors + mpl.cm.tab20c.colors

    schedule.sort_values(by=['trip_id', 'dep_time'])
    schedule.set_index(['trip_id', 'bus_id'], inplace=True)

    fig, ax = plt.subplots(1,1, figsize=(15, 5+(len(trips)+len(buses))/16))
    for jdx, j in enumerate(trips, 1):
        for mdx, m in enumerate(buses, 1):
            if (j,m) in schedule.index:
                if type(schedule.loc[(j,m), 'dep_time']) == np.int64:
                    xs = schedule.loc[(j,m), 'dep_time']
                    xf = schedule.loc[(j,m), 'arr_time']
                    xs_str = schedule.loc[(j,m), 'dep_str']
                    xf_str = schedule.loc[(j,m), 'arr_str']
                else:
                    xs = schedule.loc[(j,m), 'dep_time'].iloc[0]
                    xf = schedule.loc[(j,m), 'arr_time'].iloc[0]
                    xs_str = schedule.loc[(j,m), 'dep_str'].iloc[0]
                    xf_str = schedule.loc[(j,m), 'arr_str'].iloc[0]
                dur = xf-xs
                # ax[0].plot([xs, xf], [jdx]*2, c=colors[mdx%len(all_sch)], **bar_style)
                ax.plot([xs, xf], [mdx]*2, c=colors[jdx%len(all_sch)], **bar_style)
                if dur < 100:
                    text = small_text_style
                    xs += 10
                    xf -= 10
                elif dur <= 130:
                    text = text_style
                    xs += 25
                    xf -= 25
                else:
                    text = text_style
                    xs += 20
                    xf -= 20
                # ax[0].text((xs + xf)/2, jdx, m, **text)
                # ax[0].text(xs, jdx, xs_str, **text)
                # ax[0].text(xf, jdx, xf_str, **text)
                ax.text((xs + xf)/2, mdx, j, **text)
                ax.text(xs, mdx, xs_str, **text)
                ax.text(xf, mdx, xf_str, **text)
    
    # for mdx, m in enumerate(buses, 1):    
    #     for jdx, j in enumerate(trips, 0):
    #         if (j,m) in schedule.index:
    #             if type(schedule.loc[(j,m), 'dep_time']) == np.int64:
    #                 xs = schedule.loc[(j,m), 'dep_time']
    #                 xf = schedule.loc[(j,m), 'arr_time']
    #                 xs1 = schedule.loc[(j+1,m), 'dep_time']
    #                 xf1 = schedule.loc[(j+1,m), 'arr_time']
    #                 xs_str = schedule.loc[(j,m), 'dep_str']
    #                 xf_str = schedule.loc[(j,m), 'arr_str']
    #             else:
    #                 xs = schedule.loc[(j,m), 'dep_time'].iloc[0]
    #                 xf = schedule.loc[(j,m), 'arr_time'].iloc[0]
    #                 xs1 = schedule.loc[(j+1,m), 'dep_time']
    #                 xf1 = schedule.loc[(j+1,m), 'arr_time']
    #                 xs_str = schedule.loc[(j,m), 'dep_str'].iloc[0]
    #                 xf_str = schedule.loc[(j,m), 'arr_str'].iloc[0]
    #             dur = xf-xs
    #             ax.plot([xf, xs1], [jdx, jdx+1], c=colors[jdx%len(all_sch)], **bar_style)
    #             # ax.plot(
    #             #     [current_trip['end_time'], next_trip['start_time']],
    #             #     [current_trip['trip_id'], next_trip['trip_id']],
    #             #     color="gray", linestyle="--", alpha=0.7
    #             # )
        
    # ax[0].set_title(f'{title} Trip Schedule')
    # ax[0].set_ylabel('Trip ID')
    ax.set_title(f'{title} Bus Schedule')
    ax.set_ylabel('Bus ID')
    
    for idx, s in enumerate([trips, buses]):
        if idx ==1:
            ax.set_ylim(0.5, len(s) + 0.5)
            ax.set_yticks(range(1, 1 + len(s)))
            ax.set_yticklabels(s)
            ax.text(makespan, ax.get_ylim()[0]-0.2, "{0:0.1f}".format(makespan), ha='center', va='top')
            ax.plot([makespan]*2, ax.get_ylim(), 'r--')
            ax.set_xlabel('Time')
            ax.grid(True)
        
    fig.tight_layout()
    fig.savefig(f"{saveFigDir}{title}")
    return fig

def load_data_hyperparameter(df2, title):
    print(f"title = {title}.... soln_filename = {soln_filename}")
    bus_schedules = []
    schedule = []
    if type(df2['dep_time'].iloc[0]) != np.int64:
        df2['dep_time'] = df2['dep_time'].apply(lambda x: x.hour * 60 + x.minute if x != '-' else 0)
        df2['arr_time'] = df2['arr_time'].apply(lambda x: x.hour * 60 + x.minute if x != '-' else 0)
    df3 = df2.reset_index() #df2.copy(deep=True)
    df3['ID'] = range(len(df3))  
    df3 = df3.rename(columns={'index': 'trip_id'})
    if soln_filename.endswith(".asc"):
        with open(soln_filename, "r") as f:
            content = f.readlines()
        busnumID = 'U'
        solnID = 'Y'
        numBuses = 0
        solns = []
        for line in content:
            if line.startswith(busnumID):
                numBuses += 1
            elif line.startswith(solnID):
                splitter = line.split(' ')
                if splitter[-1].startswith("1"):    
                    solns.append(line.split(" ")[0])
            elif line.startswith("Incumbent solution"):
                numBuses = 0
                solns = []
            else:
                pass
        data = [v.split("#")[1:] for v in solns]
        soln_df = pd.DataFrame(data, columns = ['origin', 'destination', 'BusID', 'RechargingCycle'])
        soln_df['origin'] = soln_df.origin.apply(int)
        soln_df['destination'] = soln_df.destination.apply(int)
        soln_df['RechargingCycle'] = soln_df.RechargingCycle.apply(int)
        merged_df = pd.merge(soln_df, df3[['ID', 'trip_id']], left_on='destination', right_on='ID', how='left')
        merged_df.rename(columns={"name": "trip_id"}, inplace=True)
        schedules = {}
        for bus_id, group in merged_df.groupby('BusID'):
            origin = 0
            cycle = 0
            temp = []
            dest = group.loc[(group['origin'] == origin) & (group['RechargingCycle'] == cycle), ['destination', 'trip_id', 'RechargingCycle']].iloc[0]
            while dest['destination'] != 0:
                temp.append(dest['trip_id'])
                origin = dest['destination']
                if origin in R:
                    cycle += 1
                dest = group.loc[(group['origin'] == origin) & (group['RechargingCycle'] == cycle), ['destination', 'trip_id', 'RechargingCycle']].iloc[0]
            schedules[bus_id] = temp
    else:
        with open(soln_filename, 'r') as f:
            lines = f.read()
            lines = lines.split("\n")
            for line in lines:
                if line.startswith("BusID"):
                    splitline = line.split(" -> ")
                    schedule.append(splitline)
        for sch in schedule:
            new_schedule = [int(i.split(" @ ")[0]) for i in sch[1:]]
            print(new_schedule)
            bus_schedules.append(new_schedule)
        print(bus_schedules)
    
        schedules = {}
        for bus_id, schedule in enumerate(bus_schedules):
            roster = []
            for trip in schedule:
                if trip != 0:
                    print(f"trip = {trip}...")
                    trip_id = df3.loc[df3.ID == trip, 'trip_id'].iloc[0]
                    roster.append(trip_id)
            schedules[bus_id] = roster
        schedules
        print(schedules)
    return visualizeResult(schedules, df3, title)

def visualizeResult(rosters, all_sch, title):
    all_sch['ID'] = range(len(all_sch))
    if "type" in all_sch.columns:
        cs_ids = all_sch.loc[all_sch.type == "cs"].trip_id.tolist()
    else:
        cs_ids = all_sch.loc[all_sch.ID.isin(R)].trip_id.tolist()
    depot = all_sch.loc[all_sch.ID.isin([0])].index
    all_sch.set_index('trip_id', inplace=True)
    res_dict = []
    for bus_id, roster in rosters.items():
        for idx in range(len(roster)):
            s = roster[idx]
            bus = {}
            bus['bus_id'] = int(bus_id)
            if s in cs_ids:
                s = all_sch.loc[all_sch.index == s, 'ID'].iloc[0]
                prev = all_sch.loc[all_sch.index == roster[idx-1], 'ID'].iloc[0]
                bus['trip_id'] = all_sch.loc[all_sch.ID == s].index[0]
                dur = all_sch.loc[all_sch.ID == s, 'duration'].iloc[0] - 50
                bus['dep_time'] = (all_sch.loc[all_sch.ID == prev, 'arr_time'].iloc[0]) + int(dur/2)
                bus['arr_time'] = bus['dep_time'] + 50 + int(dur/2)
                bus['duration'] = dur + 50
                bus['dep_terminal'] = all_sch.loc[all_sch.ID == prev, 'arr_terminal'].iloc[0]
                bus['arr_terminal'] = all_sch.loc[all_sch.ID == prev, 'arr_terminal'].iloc[0]
            elif s in depot:
                continue
            else:
                s = all_sch.loc[all_sch.index == s, 'ID'].iloc[0]
                bus['trip_id'] = all_sch.loc[all_sch.ID == s].index[0]
                bus['dep_time'] = all_sch.loc[all_sch.ID == s, 'dep_time'].iloc[0]
                bus['arr_time'] = all_sch.loc[all_sch.ID == s, 'arr_time'].iloc[0]
                bus['duration'] = all_sch.loc[all_sch.ID == s, 'duration'].iloc[0]
                bus['dep_terminal'] = all_sch.loc[all_sch.ID == s, 'dep_terminal'].iloc[0]
                bus['arr_terminal'] = all_sch.loc[all_sch.ID == s, 'arr_terminal'].iloc[0]
            res_dict.append(bus)
    schedule_df = pd.DataFrame(res_dict)
    fig = visualize(res_dict, all_sch, title)
    return schedule_df, fig

filepath = r".\\MIP-"
filename = f"{filepath}Data\\final_test.xlsx"
sheetname = "10Trips"

NTrips = 10 #trips
NTerms = 2 #terminals
NCS = 1 #recharging stations
CHARGING_TIME = 100 #minutes
NRechargeCycle = 3 #cycle
D_MAX = 350 #minutes of operation
DEPOT = 1 #depot
# print(filename)
df = pd.read_excel(filename, sheet_name="10Trips", usecols="A:G", nrows=NTrips+NCS+DEPOT)
term = pd.read_excel(filename, sheet_name="10Trips", usecols="J:L", skiprows=10, nrows=NTerms)
gamma = pd.read_excel(filename, sheet_name="10Trips", usecols="B:L", skiprows=16, nrows=NTrips+DEPOT)
delta = pd.read_excel(filename, sheet_name="10Trips", usecols="P:AA", skiprows=16, nrows=NTrips+DEPOT+NCS)
phi = pd.read_excel(filename, sheet_name="10Trips", usecols="B:L", skiprows=30, nrows=(NTrips+DEPOT)*NCS)
# zeta = pd.read_excel(filename, sheet_name="10Trips", usecols="", skiprows=, nrows=()))
df2 = df.set_index('trip_id')
gamma = gamma.to_numpy()
delta = delta.to_numpy()
phi = phi.to_numpy()

trips_df = df.copy()
s = len(trips_df)
k = (s // 2) - 2
D_MAX = 350

K = [i for i in range(1, k+1)]
# print(f"K = {K}")
Sprime = [i for i in range(s)]
# print(f"Sprime = {Sprime}")
S = Sprime[:-NCS]
# print(f"S = {S}")
S1 = S[1:]
# print(f"S1 = {S1}")
R = Sprime[-NCS:]
# print(f"R = {R}")
# print(f"Expected number of buses = {k} -> {K}\nNumber of available schedules {s} -> {Sprime}\n")
durations = trips_df.duration.to_list()

C = [i for i in range(1, NRechargeCycle+1)]
# print(C)
df2['ID'] = range(1, len(df2)+1)
cs_ids = df2.loc[df2.ID.isin(R)].index.to_list()
soln_filename = f"{filepath}cplex_results\\Het10TripsEVeh1CS_Idle.txt"
soln_filename = f"{filepath}Results\\TimeLimit\\Hetro10TripsEVeh1CS_Earliest.txt"
title = 'CPLEX-10Trips1CS'
cplex_1cs_schedule, cplex_1cs_fig = load_data_hyperparameter(df2, title)

soln_filename = f"{filepath}Results\\TimeLimit\\Het10TripsEVeh1CS_Idle.txt"
title = "CPLEX-10Trips1CS_IDLE"
cplex_1cs_IDLE_schedule, cplex_1cs_IDLE_fig = load_data_hyperparameter(df2, title)

soln_filename = f"{filepath}Results\\TimeLimit\\Het10TripsEVeh1CS_Idle_NEW.txt"
title = "CPLEX-10Trips1CS_IDLE_NEW"
cplex_1cs_IDLE_schedule_new, cplex_1cs_IDLE_fig_new = load_data_hyperparameter(df2, title)

soln_filename = f"{filepath}Results\\TimeLimit\\Het10TripsEVeh1CS_Idle_NEW_NEW.txt"
title = "CPLEX-10Trips1CS_IDLE_NEW"
cplex_1cs_IDLE_schedule_new_NEW, cplex_1cs_IDLE_fig_new_NEW = load_data_hyperparameter(df2, title)

############################################### 10 TRIPS MULTIPLE CSs ###########################################

filename = f"{filepath}Data\\final_test.xlsx"
sheetname = "10Trips"

NCS = 2 #recharging stations
df = pd.read_excel(filename, sheet_name="10Trips", usecols="A:G", nrows=NTrips+NCS+DEPOT)
term = pd.read_excel(filename, sheet_name="10Trips", usecols="J:L", skiprows=10, nrows=NTerms)
gamma = pd.read_excel(filename, sheet_name="10Trips", usecols="B:L", skiprows=16, nrows=NTrips+DEPOT)
delta = pd.read_excel(filename, sheet_name="10Trips", usecols="P:AA", skiprows=16, nrows=NTrips+DEPOT+NCS)
phi = pd.read_excel(filename, sheet_name="10Trips", usecols="B:L", skiprows=30, nrows=(NTrips+DEPOT)*NCS)
df2 = df.set_index('trip_id')
gamma = gamma.to_numpy()
delta = delta.to_numpy()
phi = phi.to_numpy()

trips_df = df.copy()
s = len(trips_df)
k = (s // 2) - 2
D_MAX = 350

K = [i for i in range(1, k+1)]
# print(f"K = {K}")
Sprime = [i for i in range(s)]
# print(f"Sprime = {Sprime}")
S = Sprime[:-NCS]
# print(f"S = {S}")
S1 = S[1:]
# print(f"S1 = {S1}")
R = Sprime[-NCS:]
# print(f"R = {R}")
# print(f"Expected number of buses = {k} -> {K}\nNumber of available schedules {s} -> {Sprime}\n")
durations = trips_df.duration.to_list()
C = [i for i in range(1, NRechargeCycle+1)]
# print(C)
df2['ID'] = range(1, len(df2)+1)
cs_ids = df2.loc[df2.ID.isin(R)].index.to_list()
soln_filename = f"{filepath}Results\\TimeLimit\\Hetro10TripsEVeh2CS_Earliest.txt"
title = 'CPLEX-10Trips2CS'
cplex_2cs_schedule, cplex_2cs_fig = load_data_hyperparameter(df2, title)

soln_filename = f"{filepath}Results\\TimeLimit\\Het10TripsEVeh2CS_Idle.txt"
title = 'CPLEX-10Trips2CS_IDLE'
cplex_2cs_IDLE_schedule, cplex_2cs_IDLE_fig = load_data_hyperparameter(df2, title)

soln_filename = f"{filepath}Results\\TimeLimit\\Het10TripsEVeh2CS_Idle_NEW.txt"
title = "CPLEX-10Trips2CS_IDLE_NEW"
cplex_2cs_IDLE_schedule_new, cplex_2cs_IDLE_fig_new = load_data_hyperparameter(df2, title)

soln_filename = f"{filepath}Results\\TimeLimit\\Het10TripsEVeh2CS_Idle_NEW_NEW.txt"
title = "CPLEX-10Trips2CS_IDLE_NEW"
cplex_2cs_IDLE_schedule_new_NEW, cplex_2cs_IDLE_fig_new_NEW = load_data_hyperparameter(df2, title)


############################################### 30 TRIPS Single CS ###########################################

sheetname = "30Trips"

NTrips = 30 #trips
NTerms = 4 #terminals
NCS = 1 #recharging stations
CHARGING_TIME = 100 #minutes
NRechargeCycle = 3 #cycle
D_MAX = 350 #minutes of operation
DEPOT = 1 #depot
df = pd.read_excel(filename, sheet_name="test", usecols="A:G", nrows=NTrips+NCS+DEPOT)
term = pd.read_excel(filename, sheet_name="test", usecols="J:L", skiprows=10, nrows=NTerms)
gamma = pd.read_excel(filename, sheet_name="feasible_mat", usecols="B:AF", nrows=NTrips+DEPOT)
delta = pd.read_excel(filename, sheet_name="duration_mat", usecols="B:AG", nrows=NTrips+DEPOT+NCS)
phi = pd.read_excel(filename, sheet_name="Recharged_Feasible_mat", usecols="C:AG", nrows=(NTrips+DEPOT)*NCS)
df2 = df.set_index('trip_id')
gamma = gamma.to_numpy()
delta = delta.to_numpy()
phi = phi.to_numpy()

trips_df = df.copy()
s = len(trips_df)
k = (s // 2) - 2

K = [i for i in range(1, k+1)]
# print(f"K = {K}")
Sprime = [i for i in range(s)]
# print(f"Sprime = {Sprime}")
S = Sprime[:-NCS]
# print(f"S = {S}")
S1 = S[1:]
# print(f"S1 = {S1}")
R = Sprime[-NCS:]
# print(f"R = {R}")
# print(f"Expected number of buses = {k} -> {K}\nNumber of available schedules {s} -> {Sprime}\n")
durations = trips_df.duration.to_list()
C = [i for i in range(1, NRechargeCycle+1)]
# print(C)
df2['ID'] = range(1, len(df2)+1)
cs_ids = df2.loc[df2.ID.isin(R)].index.to_list()
################# Time Limit ################
soln_filename = f"{filepath}Results\\TimeLimit\\Hetro30TripsEVeh1CS_Earliest.txt"
title = 'CPLEX-30Trips1CS'
cplex_30Trips1cs_schedule, cplex_30Trips1cs_fig = load_data_hyperparameter(df2, title)

############### HPC Z Folder ###############
soln_path = f"{filepath}Results\\HPC"
soln_filename = f"{soln_path}\\30TripsVehCS.asc"
title = 'CPLEX-30Trips1CS_Optimal'
cplex_30Trips1cs_schedule_OPT_VCS, cplex_30Trips1cs_fig_OPT_VCS = load_data_hyperparameter(df2, title)

############## IDLE #######################
soln_filename = f"{filepath}Results\\TimeLimit\\Het30TripsEVeh1CS_Idle.txt"
title = 'CPLEX-30Trips1CS_IDLE'
cplex_30Trips1cs_IDLE_schedule, cplex_30Trips1cs_IDLE_fig = load_data_hyperparameter(df2, title)

############################ NEW ####################################
soln_filename = f"{filepath}Results\\TimeLimit\\Het30TripsEVeh1CS_Idle_NEW_NEW.txt"
title = 'CPLEX-PC_30Trips1CS_Optimal_IDLE'
cplex_30Trips1cs_schedule_PC_IDLE, cplex_30Trips1cs_fig_PC_IDLE = load_data_hyperparameter(df2, title)
#####################################################################

############### HPC Z Folder ###############
soln_path = f"{filepath}Results\\HPC"
soln_filename = f"{soln_path}\\30Trips1CS_Idle.asc"
title = 'CPLEX-30Trips1CS_IDLE_Optimal'
cplex_30Trips1cs_schedule_OPT_IDLE, cplex_30Trips1cs_fig_OPT_IDLE = load_data_hyperparameter(df2, title)

############################################### 30 TRIPS 2CSs ###########################################

filename = f"{filepath}Data\\final_test.xlsx"
sheetname = "30Trips"

NCS = 2 #recharging stations
df = pd.read_excel(filename, sheet_name="test", usecols="A:G", nrows=NTrips+NCS+DEPOT)
term = pd.read_excel(filename, sheet_name="test", usecols="J:L", skiprows=10, nrows=NTerms)
gamma = pd.read_excel(filename, sheet_name="feasible_mat", usecols="B:AF", nrows=NTrips+DEPOT)
delta = pd.read_excel(filename, sheet_name="duration_mat", usecols="B:AH", nrows=NTrips+DEPOT+NCS)
phi = pd.read_excel(filename, sheet_name="Recharged_Feasible_mat", usecols="C:AG", nrows=(NTrips+DEPOT)*NCS)
df2 = df.set_index('trip_id')
gamma = gamma.to_numpy()
delta = delta.to_numpy()
phi = phi.to_numpy()

trips_df = df.copy()
s = len(trips_df)
k = (s // 2) - 2
D_MAX = 350

K = [i for i in range(1, k+1)]
# print(f"K = {K}")
Sprime = [i for i in range(s)]
# print(f"Sprime = {Sprime}")
S = Sprime[:-NCS]
# print(f"S = {S}")
S1 = S[1:]
# print(f"S1 = {S1}")
R = Sprime[-NCS:]
# print(f"R = {R}")
# print(f"Expected number of buses = {k} -> {K}\nNumber of available schedules {s} -> {Sprime}\n")
durations = trips_df.duration.to_list()
C = [i for i in range(1, NRechargeCycle+1)]
# print(C)
df2['ID'] = range(1, len(df2)+1)
cs_ids = df2.loc[df2.ID.isin(R)].index.to_list()
########## Time Limit Earliest ###################
soln_filename = f"{filepath}Results\\TimeLimit\\Hetro30TripsEVeh2CS_Earliest.txt"
# soln_filename = f"{filepath}Het30TripsEVeh2CS_Idle.txt"
title = 'CPLEX-30Trips2CS'
cplex_30Trips2cs_schedule, cplex_30Trips2cs_fig = load_data_hyperparameter(df2, title)

############### HPC Z Folder ###############
soln_path = f"{filepath}Results\\HPC"
soln_filename = f"{soln_path}\\30TripsVeh2CS.asc"
title = 'CPLEX-30Trips2CS_Optimal'
cplex_30Trips2cs_schedule_OPT_VCS, cplex_30Trips2cs_fig_OPT_VCS = load_data_hyperparameter(df2, title)

############## IDLE #######################
soln_filename = f"{filepath}Results\\TimeLimit\\Het30TripsEVeh2CS_Idle.txt"
title = 'CPLEX-30Trips2CS_IDLE'
cplex_30Trips2cs_IDLE_schedule, cplex_30Trips2cs_IDLE_fig = load_data_hyperparameter(df2, title)

############################ NEW ####################################
soln_filename = f"{filepath}Results\\TimeLimit\\Het30TripsEVeh2CS_Idle_NEW_NEW.txt"
title = 'CPLEX-PC_30Trips2CS_Optimal_IDLE'
cplex_30Trips2cs_schedule_PC_IDLE, cplex_30Trips2cs_fig_PC_IDLE = load_data_hyperparameter(df2, title)
#####################################################################

############### HPC Z Folder ########################
soln_filename = f"{soln_path}\\30Trips2CS_Idle.asc"
title = 'CPLEX-30Trips2CS_IDLE_Optimal'
cplex_30Trips2cs_schedule_OPT_IDLE, cplex_30Trips2cs_fig_OPT_IDLE = load_data_hyperparameter(df2, title)

############################################### 30 TRIPS 3CSs ###########################################
filename = f"{filepath}Data\\final_test.xlsx"
sheetname = "30Trips"

NCS = 3 #recharging stations
df = pd.read_excel(filename, sheet_name="test", usecols="A:G", nrows=NTrips+NCS+DEPOT)
term = pd.read_excel(filename, sheet_name="test", usecols="J:L", skiprows=10, nrows=NTerms)
gamma = pd.read_excel(filename, sheet_name="feasible_mat", usecols="B:AF", nrows=NTrips+DEPOT)
delta = pd.read_excel(filename, sheet_name="duration_mat", usecols="B:AI", nrows=NTrips+DEPOT+NCS)
phi = pd.read_excel(filename, sheet_name="Recharged_Feasible_mat", usecols="C:AG", nrows=(NTrips+DEPOT)*NCS)
df2 = df.set_index('trip_id')
gamma = gamma.to_numpy()
delta = delta.to_numpy()
phi = phi.to_numpy()

trips_df = df.copy()
s = len(trips_df)
k = (s // 2) - 2
D_MAX = 350

K = [i for i in range(1, k+1)]
# print(f"K = {K}")
Sprime = [i for i in range(s)]
# print(f"Sprime = {Sprime}")
S = Sprime[:-NCS]
# print(f"S = {S}")
S1 = S[1:]
# print(f"S1 = {S1}")
R = Sprime[-NCS:]
# print(f"R = {R}")
# print(f"Expected number of buses = {k} -> {K}\nNumber of available schedules {s} -> {Sprime}\n")
durations = trips_df.duration.to_list()
C = [i for i in range(1, NRechargeCycle+1)]
# print(C)
df2['ID'] = range(1, len(df2)+1)
cs_ids = df2.loc[df2.ID.isin(R)].index.to_list()

soln_filename = f"{filepath}Results\\TimeLimit\\Het30TripsEVeh3CS_Idle.txt"
title = 'CPLEX-30Trips3CS_IDLE'
cplex_30Trips3cs_schedule, cplex_30Trips3cs_fig = load_data_hyperparameter(df2, title)
############################ NEW ####################################
soln_filename = f"{filepath}Results\\TimeLimit\\Het30TripsEVeh3CS_Idle_NEW_NEW.txt"
title = 'CPLEX-PC_30Trips3CS_Optimal_IDLE'
cplex_30Trips3cs_schedule_PC_IDLE, cplex_30Trips3cs_fig_PC_IDLE = load_data_hyperparameter(df2, title)
#####################################################################
# ############### HPC Z Folder ########################
# soln_path = r"Z:\\cplex\\"
# soln_filename = f"{soln_path}\\30Trips3CS_Idle.asc"
# title = 'CPLEX-30Trips3CS_IDLE_Optimal'
# cplex_30Trips3cs_schedule_OPT_IDLE, cplex_30Trips3cs_fig_OPT_IDLE = load_data_hyperparameter(df2, title)

# ############################################### 54 TRIPS Single CS ###########################################
# filename = f"{filepath}Data\\54Trips_3CS.xlsx"
# sheetname = "54Trips"

# NTrips = 54 #trips
# NTerms = 4 #terminals
# NCS = 1 #recharging stations
# CHARGING_TIME = 100 #minutes
# NRechargeCycle = 3 #cycle
# D_MAX = 350 #minutes of operation
# DEPOT = 1 #depot
# df = pd.read_excel(filename, sheet_name="main", usecols="A:H", nrows=NTrips+NCS+DEPOT)
# term = pd.read_excel(filename, sheet_name="main", usecols="L:M", skiprows=10, nrows=NTerms)
# gamma = pd.read_excel(filename, sheet_name="gamma", usecols="B:BD", nrows=NTrips+DEPOT)
# delta = pd.read_excel(filename, sheet_name="delta", usecols="B:BE", nrows=NTrips+DEPOT+NCS)
# phi = pd.read_excel(filename, sheet_name="phi", usecols="B:BD", nrows=(NTrips+DEPOT)*NCS)
# df.rename(columns={'name':'trip_id', 'dep_term': 'dep_terminal', 'arr_term': 'arr_terminal'}, inplace=True)
# df2 = df.set_index('trip_id')
# gamma = gamma.to_numpy()
# delta = delta.to_numpy()
# phi = phi.to_numpy()

# trips_df = df.copy()
# s = len(trips_df)
# k = (s // 2) - 2
# D_MAX = 350

# K = [i for i in range(1, k+1)]
# # print(f"K = {K}")
# Sprime = [i for i in range(s)]
# # print(f"Sprime = {Sprime}")
# S = Sprime[:-NCS]
# # print(f"S = {S}")
# S1 = S[1:]
# # print(f"S1 = {S1}")
# R = Sprime[-NCS:]
# # print(f"R = {R}")
# # print(f"Expected number of buses = {k} -> {K}\nNumber of available schedules {s} -> {Sprime}\n")
# durations = trips_df.duration.to_list()
# C = [i for i in range(1, NRechargeCycle+1)]
# # print(C)
# df2['ID'] = range(1, len(df2)+1)
# cs_ids = df2.loc[df2.ID.isin(R)].index.to_list()
# # print(cs_ids)

# ########## Time Limit Earliest ###################
# soln_filename = f"{filepath}Results\\TimeLimit\\Hetro54TripsEVeh1CS_Earliest.txt"
# title = 'CPLEX-54Trips1CS'
# cplex_54Trips1cs_schedule, cplex_54Trips1cs_fig = load_data_hyperparameter(df2, title)

# ############### HPC Z Folder ###############
# soln_path = f"{filepath}Results\\HPC"
# soln_filename = f"{soln_path}\\54Trips1EVehCSs.asc"
# title = 'CPLEX-54Trips1CS_Optimal'
# cplex_54Trips1cs_schedule_OPT, cplex_54Trips1cs_fig_OPT = load_data_hyperparameter(df2, title)

# ############### HPC Z Folder ###############
# soln_path = f"{filepath}Results\\HPC"
# soln_filename = f"{soln_path}\\PC_IDLE_54Trips1CS.asc"
# title = 'CPLEX-PC_54Trips1CS_Optimal_IDLE'
# cplex_54Trips1cs_schedule_OPT_PC_IDLE, cplex_54Trips1cs_fig_OPT_PC_IDLE = load_data_hyperparameter(df2, title)

# ############################################### 54 TRIPS MULTIPLE CSs ###########################################
# filename = f"{filepath}Data\\54Trips_3CS.xlsx"
# sheetname = "54Trips"

# NTrips = 54 #trips
# NTerms = 4 #terminals
# NCS = 3 #recharging stations
# CHARGING_TIME = 100 #minutes
# NRechargeCycle = 3 #cycle
# D_MAX = 350 #minutes of operation
# DEPOT = 1 #depot
# df = pd.read_excel(filename, sheet_name="main", usecols="A:H", nrows=NTrips+NCS+DEPOT)
# term = pd.read_excel(filename, sheet_name="main", usecols="L:M", skiprows=10, nrows=NTerms)
# gamma = pd.read_excel(filename, sheet_name="gamma", usecols="B:BD", nrows=NTrips+DEPOT)
# delta = pd.read_excel(filename, sheet_name="delta", usecols="B:BG", nrows=NTrips+DEPOT+NCS)
# phi = pd.read_excel(filename, sheet_name="phi", usecols="B:BD", nrows=(NTrips+DEPOT)*NCS)
# df.rename(columns={'name':'trip_id', 'dep_term': 'dep_terminal', 'arr_term': 'arr_terminal'}, inplace=True)
# df2 = df.set_index('trip_id')
# gamma = gamma.to_numpy()
# delta = delta.to_numpy()
# phi = phi.to_numpy()

# trips_df = df.copy()
# s = len(trips_df)
# k = (s // 2) - 2
# D_MAX = 350

# K = [i for i in range(1, k+1)]
# # print(f"K = {K}")
# Sprime = [i for i in range(s)]
# # print(f"Sprime = {Sprime}")
# S = Sprime[:-NCS]
# # print(f"S = {S}")
# S1 = S[1:]
# # print(f"S1 = {S1}")
# R = Sprime[-NCS:]
# # print(f"R = {R}")
# # print(f"Expected number of buses = {k} -> {K}\nNumber of available schedules {s} -> {Sprime}\n")
# durations = trips_df.duration.to_list()
# C = [i for i in range(1, NRechargeCycle+1)]
# # print(C)
# df2['ID'] = range(1, len(df2)+1)
# cs_ids = df2.loc[df2.ID.isin(R)].index.to_list()
# # print(cs_ids)
# soln_filename = f"{filepath}Results\\TimeLimit\\Hetro54TripsEVeh3CS_Earliest.txt"
# title = 'CPLEX-54Trips3CS'
# cplex_54Trips3cs_schedule, cplex_54Trips3cs_fig = load_data_hyperparameter(df2, title)

# # ############### HPC Z Folder ###############
# # soln_path = f"{filepath}Results\\HPC"
# # soln_filename = f"{soln_path}\\54Trips3CS_Exact.asc"
# # title = 'CPLEX-54Trips3CS_Optimal'
# # cplex_54Trips3cs_schedule_OPT, cplex_54Trips3cs_fig_OPT = load_data_hyperparameter(df2, title)

# ############### HPC Z Folder ###############
# soln_path = f"{filepath}Results\\HPC"
# soln_filename = f"{soln_path}\\54Trips3CS_Idle.asc"
# title = 'CPLEX-54Trips3CS_Optimal'
# cplex_54Trips3cs_schedule_OPT, cplex_54Trips3cs_fig_OPT = load_data_hyperparameter(df2, title)

# soln_path = f"{filepath}Results\\HPC"
# soln_filename = f"{soln_path}\\PC_IDLE_54Trips3CS.asc"
# title = 'CPLEX-PC_54Trips3CS_Optimal_IDLE'
# cplex_54Trips3cs_schedule_OPT_PC_IDLE, cplex_54Trips3cs_fig_OPT_PC_IDLE = load_data_hyperparameter(df2, title)

# # soln_filename = f"{filepath}Hetro54TripsEVeh3CS_Earliest.txt"
# # title = 'CPLEX-54Trips3CS'
# # cplex_54Trips3cs_schedule, cplex_54Trips3cs_fig = load_data_hyperparameter(df2, title)

########################################################### Analysis ################################################################

###### 10Trips ############
cplex_1cs_schedule_df = cplex_1cs_schedule.copy(deep=True)
cplex_1cs_schedule_df['next_dep'] = cplex_1cs_schedule_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
cplex_1cs_schedule_df['difference'] = cplex_1cs_schedule_df['next_dep'] - cplex_1cs_schedule_df['arr_time']
cplex_1cs_schedule_df['difference'] = cplex_1cs_schedule_df['difference'].apply(lambda x: 0 if x < 0 else x)
cplexCS1_soln = cplex_1cs_schedule_df.groupby(['bus_id']).agg(
    trips=('trip_id', concat_str),
    numRecharge=('trip_id',countRecharge),
    numTrips=('trip_id', countTrips),
    gapTime=('difference', 'sum')
)

print(cplexCS1_soln)
# cplexCS1_soln.describe()


cplex_2cs_schedule_df = cplex_2cs_schedule.copy(deep=True)
cplex_2cs_schedule_df['next_dep'] = cplex_2cs_schedule_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
cplex_2cs_schedule_df['difference'] = cplex_2cs_schedule_df['next_dep'] - cplex_2cs_schedule_df['arr_time']
cplex_2cs_schedule_df['difference'] = cplex_2cs_schedule_df['difference'].apply(lambda x: 0 if x < 0 else x)
cplexCS2_soln = cplex_2cs_schedule_df.groupby(['bus_id']).agg(
    trips=('trip_id', concat_str),
    numRecharge=('trip_id',countRecharge),
    numTrips=('trip_id', countTrips),
    gapTime=('difference', 'sum')
)
print(cplexCS2_soln)
cplexCS2_soln.describe()
print("Exact Solution for 10Trips between 1CS vs. 2CS....")
print(pd.concat([cplexCS1_soln.describe(), cplexCS2_soln.describe()], axis=1))
print("-"*150)

test = cplexCS1_soln.describe().loc['mean']
test['numBuses'] = cplexCS1_soln.describe().loc['count','gapTime']
test['time_to_best_soln'] = 2.92
test = test.to_frame().rename(columns={"mean":"CPLEX_10Trips1CS"})

test2 = cplexCS2_soln.describe().loc['mean']
test2['numBuses'] = cplexCS2_soln.describe().loc['count','gapTime']
test2['time_to_best_soln'] = 3.8
test2 = test2.to_frame().rename(columns={"mean":"CPLEX_10Trips2CS"})

result_CPLEX_10Trips = pd.concat([test, test2], axis=1)
print(result_CPLEX_10Trips)
# trips10_1cs = cplex_1cs_schedule_df.groupby(['bus_id']).agg({'trip_id':concat_str, 'dep_time': 'first', 'arr_time': 'last', 'dep_terminal': 'first', 'arr_terminal': 'last', 'difference': 'sum'})
# trips10_1cs.to_csv("cplex_30Trips1CS_IDLE.csv")
# trips10_2cs = cplex_2cs_schedule_df.groupby(['bus_id']).agg({'trip_id':concat_str, 'dep_time': 'first', 'arr_time': 'last', 'dep_terminal': 'first', 'arr_terminal': 'last', 'difference': 'sum'})
# trips10_2cs.to_csv("cplex_30Trips2CS_IDLE.csv")

###### 10Trips ############
cplex_1cs_IDLE_schedule_df = cplex_1cs_IDLE_schedule.copy(deep=True)
cplex_1cs_IDLE_schedule_df['next_dep'] = cplex_1cs_IDLE_schedule_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
cplex_1cs_IDLE_schedule_df['difference'] = cplex_1cs_IDLE_schedule_df['next_dep'] - cplex_1cs_IDLE_schedule_df['arr_time']
cplex_1cs_IDLE_schedule_df['difference'] = cplex_1cs_IDLE_schedule_df['difference'].apply(lambda x: 0 if x < 0 else x)
cplexCS1_IDLE_soln = cplex_1cs_IDLE_schedule_df.groupby(['bus_id']).agg(
    trips=('trip_id', concat_str),
    numRecharge=('trip_id',countRecharge),
    numTrips=('trip_id', countTrips),
    gapTime=('difference', 'sum')
)

print(cplexCS1_IDLE_soln)
# cplexCS1_IDLE_soln.describe()


cplex_2cs_IDLE_schedule_df = cplex_2cs_IDLE_schedule.copy(deep=True)
cplex_2cs_IDLE_schedule_df['next_dep'] = cplex_2cs_IDLE_schedule_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
cplex_2cs_IDLE_schedule_df['difference'] = cplex_2cs_IDLE_schedule_df['next_dep'] - cplex_2cs_IDLE_schedule_df['arr_time']
cplex_2cs_IDLE_schedule_df['difference'] = cplex_2cs_IDLE_schedule_df['difference'].apply(lambda x: 0 if x < 0 else x)
cplexCS2_IDLE_soln = cplex_2cs_IDLE_schedule_df.groupby(['bus_id']).agg(
    trips=('trip_id', concat_str),
    numRecharge=('trip_id',countRecharge),
    numTrips=('trip_id', countTrips),
    gapTime=('difference', 'sum')
)
print(cplexCS2_IDLE_soln)
# cplexCS2_IDLE_soln.describe()
print("HPC with IDLE Exact Solution for 10Trips between 1CS vs. 2CS....")
print(pd.concat([cplexCS1_IDLE_soln.describe(), cplexCS2_IDLE_soln.describe()], axis=1))
print("-"*150)

test = cplexCS1_IDLE_soln.describe().loc['mean']
test['numBuses'] = cplexCS1_IDLE_soln.describe().loc['count','gapTime']
test['time_to_best_soln'] = 2.92
test = test.to_frame().rename(columns={"mean":"CPLEX_10Trips1CS"})

test2 = cplexCS2_IDLE_soln.describe().loc['mean']
test2['numBuses'] = cplexCS2_IDLE_soln.describe().loc['count','gapTime']
test2['time_to_best_soln'] = 3.8
test2 = test2.to_frame().rename(columns={"mean":"CPLEX_10Trips2CS"})

result_HPC_CPLEX_10Trips = pd.concat([test, test2], axis=1)
print(result_HPC_CPLEX_10Trips)

###### 10Trips_NEW ############
new_cplex_1cs_IDLE_schedule_df = cplex_1cs_IDLE_schedule_new.copy(deep=True)
new_cplex_1cs_IDLE_schedule_df['next_dep'] = new_cplex_1cs_IDLE_schedule_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
new_cplex_1cs_IDLE_schedule_df['difference'] = new_cplex_1cs_IDLE_schedule_df['next_dep'] - new_cplex_1cs_IDLE_schedule_df['arr_time']
new_cplex_1cs_IDLE_schedule_df['difference'] = new_cplex_1cs_IDLE_schedule_df['difference'].apply(lambda x: 0 if x < 0 else x)
new_cplexCS1_IDLE_soln = new_cplex_1cs_IDLE_schedule_df.groupby(['bus_id']).agg(
    trips=('trip_id', concat_str),
    numRecharge=('trip_id',countRecharge),
    numTrips=('trip_id', countTrips),
    gapTime=('difference', 'sum')
)

print(new_cplexCS1_IDLE_soln)
# cplexCS1_IDLE_soln.describe()

new_new_cplex_1cs_IDLE_schedule_df = cplex_1cs_IDLE_schedule_new_NEW.copy(deep=True)
new_new_cplex_1cs_IDLE_schedule_df['next_dep'] = new_new_cplex_1cs_IDLE_schedule_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
new_new_cplex_1cs_IDLE_schedule_df['difference'] = new_new_cplex_1cs_IDLE_schedule_df['next_dep'] - new_new_cplex_1cs_IDLE_schedule_df['arr_time']
new_new_cplex_1cs_IDLE_schedule_df['difference'] = new_new_cplex_1cs_IDLE_schedule_df['difference'].apply(lambda x: 0 if x < 0 else x)
new_new_cplexCS1_IDLE_soln = new_new_cplex_1cs_IDLE_schedule_df.groupby(['bus_id']).agg(
    trips=('trip_id', concat_str),
    numRecharge=('trip_id',countRecharge),
    numTrips=('trip_id', countTrips),
    gapTime=('difference', 'sum')
)
print(new_new_cplexCS1_IDLE_soln)
# cplexCS2_IDLE_soln.describe()
print("PC IDLE HPC Results between Previous Defined vs. New Defined")
print(pd.concat([new_cplexCS1_IDLE_soln.describe(), new_new_cplexCS1_IDLE_soln.describe()], axis=1))
print("-"*150)


new_cplex_2cs_IDLE_schedule_df = cplex_2cs_IDLE_schedule_new.copy(deep=True)
new_cplex_2cs_IDLE_schedule_df['next_dep'] = new_cplex_2cs_IDLE_schedule_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
new_cplex_2cs_IDLE_schedule_df['difference'] = new_cplex_2cs_IDLE_schedule_df['next_dep'] - new_cplex_2cs_IDLE_schedule_df['arr_time']
new_cplex_2cs_IDLE_schedule_df['difference'] = new_cplex_2cs_IDLE_schedule_df['difference'].apply(lambda x: 0 if x < 0 else x)
new_cplexCS2_IDLE_soln = new_cplex_2cs_IDLE_schedule_df.groupby(['bus_id']).agg(
    trips=('trip_id', concat_str),
    numRecharge=('trip_id',countRecharge),
    numTrips=('trip_id', countTrips),
    gapTime=('difference', 'sum')
)
print(new_cplexCS2_IDLE_soln)
# cplexCS2_IDLE_soln.describe()
print("HPC with IDLE Exact Solution for 10Trips between 1CS vs. 2CS....")
print(pd.concat([new_cplexCS1_IDLE_soln.describe(), new_cplexCS2_IDLE_soln.describe()], axis=1))
print("-"*150)

new_new_cplex_2cs_IDLE_schedule_df = cplex_2cs_IDLE_schedule_new_NEW.copy(deep=True)
new_new_cplex_2cs_IDLE_schedule_df['next_dep'] = new_new_cplex_2cs_IDLE_schedule_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
new_new_cplex_2cs_IDLE_schedule_df['difference'] = new_new_cplex_2cs_IDLE_schedule_df['next_dep'] - new_new_cplex_2cs_IDLE_schedule_df['arr_time']
new_new_cplex_2cs_IDLE_schedule_df['difference'] = new_new_cplex_2cs_IDLE_schedule_df['difference'].apply(lambda x: 0 if x < 0 else x)
new_new_cplexCS2_IDLE_soln = new_new_cplex_2cs_IDLE_schedule_df.groupby(['bus_id']).agg(
    trips=('trip_id', concat_str),
    numRecharge=('trip_id',countRecharge),
    numTrips=('trip_id', countTrips),
    gapTime=('difference', 'sum')
)
print(new_new_cplexCS2_IDLE_soln)
# cplexCS2_IDLE_soln.describe()
print("PC IDLE HPC Results between Previous Defined vs. New Defined")
print(pd.concat([new_cplexCS2_IDLE_soln.describe(), new_new_cplexCS2_IDLE_soln.describe()], axis=1))
print("-"*150)

# trips10_1cs = cplex_1cs_schedule_df.groupby(['bus_id']).agg({'trip_id':concat_str, 'dep_time': 'first', 'arr_time': 'last', 'dep_terminal': 'first', 'arr_terminal': 'last', 'difference': 'sum'})
# trips10_1cs.to_csv("cplex_30Trips1CS_IDLE.csv")
# trips10_2cs = cplex_2cs_schedule_df.groupby(['bus_id']).agg({'trip_id':concat_str, 'dep_time': 'first', 'arr_time': 'last', 'dep_terminal': 'first', 'arr_terminal': 'last', 'difference': 'sum'})
# trips10_2cs.to_csv("cplex_30Trips2CS_IDLE.csv")

###### 30Trips ############
cplex_30Trips1cs_schedule_df = cplex_30Trips1cs_schedule.copy(deep=True)
cplex_30Trips1cs_schedule_df['next_dep'] = cplex_30Trips1cs_schedule_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
cplex_30Trips1cs_schedule_df['difference'] = cplex_30Trips1cs_schedule_df['next_dep'] - cplex_30Trips1cs_schedule_df['arr_time']
cplex_30Trips1cs_schedule_df['difference'] = cplex_30Trips1cs_schedule_df['difference'].apply(lambda x: 0 if x < 0 else x)
cplex_30Trips1cs_soln = cplex_30Trips1cs_schedule_df.groupby(['bus_id']).agg(
    trips=('trip_id', concat_str),
    numRecharge=('trip_id',countRecharge),
    numTrips=('trip_id', countTrips),
    gapTime=('difference', 'sum')
)
print(cplex_30Trips1cs_soln)
# cplex_30Trips1cs_soln.describe()

cplex_30Trips2cs_schedule_df = cplex_30Trips2cs_schedule.copy(deep=True)
cplex_30Trips2cs_schedule_df['next_dep'] = cplex_30Trips2cs_schedule_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
cplex_30Trips2cs_schedule_df['difference'] = cplex_30Trips2cs_schedule_df['next_dep'] - cplex_30Trips2cs_schedule_df['arr_time']
cplex_30Trips2cs_schedule_df['difference'] = cplex_30Trips2cs_schedule_df['difference'].apply(lambda x: 0 if x < 0 else x)
cplex_30Trips2cs_soln = cplex_30Trips2cs_schedule_df.groupby(['bus_id']).agg(
    trips=('trip_id', concat_str),
    numRecharge=('trip_id',countRecharge),
    numTrips=('trip_id', countTrips),
    gapTime=('difference', 'sum')
)
print(cplex_30Trips2cs_soln)
# cplex_30Trips2cs_soln.describe()

cplex_30Trips3cs_schedule_df = cplex_30Trips3cs_schedule.copy(deep=True)
cplex_30Trips3cs_schedule_df['next_dep'] = cplex_30Trips3cs_schedule_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
cplex_30Trips3cs_schedule_df['difference'] = cplex_30Trips3cs_schedule_df['next_dep'] - cplex_30Trips3cs_schedule_df['arr_time']
cplex_30Trips3cs_schedule_df['difference'] = cplex_30Trips3cs_schedule_df['difference'].apply(lambda x: 0 if x < 0 else x)
cplex_30Trips3cs_soln = cplex_30Trips3cs_schedule_df.groupby(['bus_id']).agg(
    trips=('trip_id', concat_str),
    numRecharge=('trip_id',countRecharge),
    numTrips=('trip_id', countTrips),
    gapTime=('difference', 'sum')
)
print(cplex_30Trips3cs_soln)
# cplex_30Trips3cs_soln.describe()
print("Time Limit Exact Solution for 30Trips between 1CS vs. 2CS vs. 3CS....")
print(pd.concat([cplex_30Trips1cs_soln.describe(), cplex_30Trips2cs_soln.describe(), cplex_30Trips3cs_soln.describe()], axis=1))
print("-"*150)

######################### EXACT ########################################################
cplex_30Trips1cs_schedule_OPT_VCS_df = cplex_30Trips1cs_schedule_OPT_VCS.copy(deep=True)
cplex_30Trips1cs_schedule_OPT_VCS_df['next_dep'] = cplex_30Trips1cs_schedule_OPT_VCS_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
cplex_30Trips1cs_schedule_OPT_VCS_df['difference'] = cplex_30Trips1cs_schedule_OPT_VCS_df['next_dep'] - cplex_30Trips1cs_schedule_OPT_VCS_df['arr_time']
cplex_30Trips1cs_schedule_OPT_VCS_df['difference'] = cplex_30Trips1cs_schedule_OPT_VCS_df['difference'].apply(lambda x: 0 if x < 0 else x)
cplex_30Trips1cs_OPT_VCS_soln = cplex_30Trips1cs_schedule_OPT_VCS_df.groupby(['bus_id']).agg(
    trips=('trip_id', concat_str),
    numRecharge=('trip_id',countRecharge),
    numTrips=('trip_id', countTrips),
    gapTime=('difference', 'sum')
)
print(cplex_30Trips1cs_OPT_VCS_soln)
# cplex_30Trips1cs_OPT_VCS_soln.describe()

cplex_30Trips2cs_schedule_OPT_VCS_df = cplex_30Trips2cs_schedule_OPT_VCS.copy(deep=True)
cplex_30Trips2cs_schedule_OPT_VCS_df['next_dep'] = cplex_30Trips2cs_schedule_OPT_VCS_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
cplex_30Trips2cs_schedule_OPT_VCS_df['difference'] = cplex_30Trips2cs_schedule_OPT_VCS_df['next_dep'] - cplex_30Trips2cs_schedule_OPT_VCS_df['arr_time']
cplex_30Trips2cs_schedule_OPT_VCS_df['difference'] = cplex_30Trips2cs_schedule_OPT_VCS_df['difference'].apply(lambda x: 0 if x < 0 else x)
cplex_30Trips2cs_OPT_VCS_soln = cplex_30Trips2cs_schedule_OPT_VCS_df.groupby(['bus_id']).agg(
    trips=('trip_id', concat_str),
    numRecharge=('trip_id',countRecharge),
    numTrips=('trip_id', countTrips),
    gapTime=('difference', 'sum')
)
print(cplex_30Trips2cs_OPT_VCS_soln)
# cplex_30Trips2cs_OPT_VCS_soln.describe()
print("HPC Exact Solution for 30Trips between 1CS vs. 2CS....")
print(pd.concat([cplex_30Trips1cs_OPT_VCS_soln.describe(), cplex_30Trips2cs_OPT_VCS_soln.describe()], axis=1))
print("-"*150)

test = cplex_30Trips1cs_OPT_VCS_soln.describe().loc['mean']
test['numBuses'] = cplex_30Trips1cs_OPT_VCS_soln.describe().loc['count','gapTime']
test['time_to_best_soln'] = 19729.29
test = test.to_frame().rename(columns={"mean":"CPLEX_HPC_EXACT_30Trips1CS_VCS"})

test2 = cplex_30Trips2cs_OPT_VCS_soln.describe().loc['mean']
test2['numBuses'] = cplex_30Trips2cs_OPT_VCS_soln.describe().loc['count','gapTime']
test2['time_to_best_soln'] = 248592.42
test2 = test2.to_frame().rename(columns={"mean":"CPLEX_HPC_EXACT_30Trips2CS_VCS"})

result_CPLEX_HPC_30Trips_VCS = pd.concat([test, test2], axis=1)
print(result_CPLEX_HPC_30Trips_VCS)

###### 30Trips IDLE ############
cplex_30Trips1cs_IDLE_schedule_df = cplex_30Trips1cs_IDLE_schedule.copy(deep=True)
cplex_30Trips1cs_IDLE_schedule_df['next_dep'] = cplex_30Trips1cs_IDLE_schedule_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
cplex_30Trips1cs_IDLE_schedule_df['difference'] = cplex_30Trips1cs_IDLE_schedule_df['next_dep'] - cplex_30Trips1cs_IDLE_schedule_df['arr_time']
cplex_30Trips1cs_IDLE_schedule_df['difference'] = cplex_30Trips1cs_IDLE_schedule_df['difference'].apply(lambda x: 0 if x < 0 else x)
cplex_30Trips1cs_IDLE_soln = cplex_30Trips1cs_IDLE_schedule_df.groupby(['bus_id']).agg(
    trips=('trip_id', concat_str),
    numRecharge=('trip_id',countRecharge),
    numTrips=('trip_id', countTrips),
    gapTime=('difference', 'sum')
)
print(cplex_30Trips1cs_IDLE_soln)
# cplex_30Trips1cs_IDLE_soln.describe()

cplex_30Trips2cs_IDLE_schedule_df = cplex_30Trips2cs_IDLE_schedule.copy(deep=True)
cplex_30Trips2cs_IDLE_schedule_df['next_dep'] = cplex_30Trips2cs_IDLE_schedule_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
cplex_30Trips2cs_IDLE_schedule_df['difference'] = cplex_30Trips2cs_IDLE_schedule_df['next_dep'] - cplex_30Trips2cs_IDLE_schedule_df['arr_time']
cplex_30Trips2cs_IDLE_schedule_df['difference'] = cplex_30Trips2cs_IDLE_schedule_df['difference'].apply(lambda x: 0 if x < 0 else x)
cplex_30Trips2cs_IDLE_soln = cplex_30Trips2cs_IDLE_schedule_df.groupby(['bus_id']).agg(
    trips=('trip_id', concat_str),
    numRecharge=('trip_id',countRecharge),
    numTrips=('trip_id', countTrips),
    gapTime=('difference', 'sum')
)
print(cplex_30Trips2cs_IDLE_soln)
# cplex_30Trips2cs_IDLE_soln.describe()
print("Time Limit With IDLE Exact Solution for 30Trips between 1CS vs. 2CS....")
print(pd.concat([cplex_30Trips1cs_IDLE_soln.describe(), cplex_30Trips2cs_IDLE_soln.describe()], axis=1))
print("-"*150)

test = cplex_30Trips1cs_IDLE_soln.describe().loc['mean']
test['numBuses'] = cplex_30Trips1cs_IDLE_soln.describe().loc['count','gapTime']
test['time_to_best_soln'] = 1360.34
test = test.to_frame().rename(columns={"mean":"CPLEX_30Trips1CS"})

test2 = cplex_30Trips2cs_IDLE_soln.describe().loc['mean']
test2['numBuses'] = cplex_30Trips2cs_IDLE_soln.describe().loc['count','gapTime']
test2['time_to_best_soln'] = 1616.89
test2 = test2.to_frame().rename(columns={"mean":"CPLEX_30Trips2CS"})

result_CPLEX_30Trips = pd.concat([test, test2], axis=1)
print(result_CPLEX_30Trips)


###### 30Trips IDLE OPTIMAL ############
cplex_30Trips1cs_schedule_OPT_IDLE_df = cplex_30Trips1cs_schedule_OPT_IDLE.copy(deep=True)
cplex_30Trips1cs_schedule_OPT_IDLE_df['next_dep'] = cplex_30Trips1cs_schedule_OPT_IDLE_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
cplex_30Trips1cs_schedule_OPT_IDLE_df['difference'] = cplex_30Trips1cs_schedule_OPT_IDLE_df['next_dep'] - cplex_30Trips1cs_schedule_OPT_IDLE_df['arr_time']
cplex_30Trips1cs_schedule_OPT_IDLE_df['difference'] = cplex_30Trips1cs_schedule_OPT_IDLE_df['difference'].apply(lambda x: 0 if x < 0 else x)
cplex_30Trips1cs_OPT_IDLE_soln = cplex_30Trips1cs_schedule_OPT_IDLE_df.groupby(['bus_id']).agg(
    trips=('trip_id', concat_str),
    numRecharge=('trip_id',countRecharge),
    numTrips=('trip_id', countTrips),
    gapTime=('difference', 'sum')
)
print(cplex_30Trips1cs_OPT_IDLE_soln)
# cplex_30Trips1cs_OPT_IDLE_soln.describe()

cplex_30Trips2cs_schedule_OPT_IDLE_df = cplex_30Trips2cs_schedule_OPT_IDLE.copy(deep=True)
cplex_30Trips2cs_schedule_OPT_IDLE_df['next_dep'] = cplex_30Trips2cs_schedule_OPT_IDLE_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
cplex_30Trips2cs_schedule_OPT_IDLE_df['difference'] = cplex_30Trips2cs_schedule_OPT_IDLE_df['next_dep'] - cplex_30Trips2cs_schedule_OPT_IDLE_df['arr_time']
cplex_30Trips2cs_schedule_OPT_IDLE_df['difference'] = cplex_30Trips2cs_schedule_OPT_IDLE_df['difference'].apply(lambda x: 0 if x < 0 else x)
cplex_30Trips2cs_OPT_IDLE_soln = cplex_30Trips2cs_schedule_OPT_IDLE_df.groupby(['bus_id']).agg(
    trips=('trip_id', concat_str),
    numRecharge=('trip_id',countRecharge),
    numTrips=('trip_id', countTrips),
    gapTime=('difference', 'sum')
)
print(cplex_30Trips2cs_OPT_IDLE_soln)
cplex_30Trips2cs_OPT_IDLE_soln.describe()
print("HPC with IDLE Exact Solution for 30Trips between 1CS vs. 2CS....")
print(pd.concat([cplex_30Trips1cs_OPT_IDLE_soln.describe(), cplex_30Trips2cs_OPT_IDLE_soln.describe()], axis=1))
print("-"*150)

test = cplex_30Trips1cs_OPT_IDLE_soln.describe().loc['mean']
test['numBuses'] = cplex_30Trips1cs_OPT_IDLE_soln.describe().loc['count','gapTime']
test['time_to_best_soln'] = 6372.83
test = test.to_frame().rename(columns={"mean":"CPLEX_HPC_EXACT_30Trips1CS"})

test2 = cplex_30Trips2cs_OPT_IDLE_soln.describe().loc['mean']
test2['numBuses'] = cplex_30Trips2cs_OPT_IDLE_soln.describe().loc['count','gapTime']
test2['time_to_best_soln'] = 248718.55
test2 = test2.to_frame().rename(columns={"mean":"CPLEX_HPC_EXACT_30Trips2CS"})

result_CPLEX_HPC_30Trips = pd.concat([test, test2], axis=1)
print(result_CPLEX_HPC_30Trips)

###### PC 30Trips IDLE OPTIMAL ############
cplex_30Trips1cs_schedule_PC_IDLE_df = cplex_30Trips1cs_schedule_PC_IDLE.copy(deep=True)
cplex_30Trips1cs_schedule_PC_IDLE_df['next_dep'] = cplex_30Trips1cs_schedule_PC_IDLE_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
cplex_30Trips1cs_schedule_PC_IDLE_df['difference'] = cplex_30Trips1cs_schedule_PC_IDLE_df['next_dep'] - cplex_30Trips1cs_schedule_PC_IDLE_df['arr_time']
cplex_30Trips1cs_schedule_PC_IDLE_df['difference'] = cplex_30Trips1cs_schedule_PC_IDLE_df['difference'].apply(lambda x: 0 if x < 0 else x)
cplex_30Trips1cs_PC_IDLE_soln = cplex_30Trips1cs_schedule_PC_IDLE_df.groupby(['bus_id']).agg(
    trips=('trip_id', concat_str),
    numRecharge=('trip_id',countRecharge),
    numTrips=('trip_id', countTrips),
    gapTime=('difference', 'sum')
)
print(cplex_30Trips1cs_PC_IDLE_soln)
print("PC IDLE HPC Results between Previous Defined vs. New Defined")
print(pd.concat([cplex_30Trips1cs_OPT_IDLE_soln.describe(), cplex_30Trips1cs_PC_IDLE_soln.describe()], axis=1))
print("#"*50)

# cplex_30Trips1cs_OPT_IDLE_soln.describe()

cplex_30Trips2cs_schedule_PC_IDLE_df = cplex_30Trips2cs_schedule_PC_IDLE.copy(deep=True)
cplex_30Trips2cs_schedule_PC_IDLE_df['next_dep'] = cplex_30Trips2cs_schedule_PC_IDLE_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
cplex_30Trips2cs_schedule_PC_IDLE_df['difference'] = cplex_30Trips2cs_schedule_PC_IDLE_df['next_dep'] - cplex_30Trips2cs_schedule_PC_IDLE_df['arr_time']
cplex_30Trips2cs_schedule_PC_IDLE_df['difference'] = cplex_30Trips2cs_schedule_PC_IDLE_df['difference'].apply(lambda x: 0 if x < 0 else x)
cplex_30Trips2cs_PC_IDLE_soln = cplex_30Trips2cs_schedule_PC_IDLE_df.groupby(['bus_id']).agg(
    trips=('trip_id', concat_str),
    numRecharge=('trip_id',countRecharge),
    numTrips=('trip_id', countTrips),
    gapTime=('difference', 'sum')
)
print(cplex_30Trips2cs_PC_IDLE_soln)

cplex_30Trips3cs_schedule_PC_IDLE_df = cplex_30Trips3cs_schedule_PC_IDLE.copy(deep=True)
cplex_30Trips3cs_schedule_PC_IDLE_df['next_dep'] = cplex_30Trips3cs_schedule_PC_IDLE_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
cplex_30Trips3cs_schedule_PC_IDLE_df['difference'] = cplex_30Trips3cs_schedule_PC_IDLE_df['next_dep'] - cplex_30Trips3cs_schedule_PC_IDLE_df['arr_time']
cplex_30Trips3cs_schedule_PC_IDLE_df['difference'] = cplex_30Trips3cs_schedule_PC_IDLE_df['difference'].apply(lambda x: 0 if x < 0 else x)
cplex_30Trips3cs_PC_IDLE_soln = cplex_30Trips3cs_schedule_PC_IDLE_df.groupby(['bus_id']).agg(
    trips=('trip_id', concat_str),
    numRecharge=('trip_id',countRecharge),
    numTrips=('trip_id', countTrips),
    gapTime=('difference', 'sum')
)
print(cplex_30Trips3cs_PC_IDLE_soln)

print("PC IDLE HPC Results between Previous Defined vs. New Defined")
print(pd.concat([cplex_30Trips2cs_OPT_IDLE_soln.describe(), cplex_30Trips2cs_PC_IDLE_soln.describe()], axis=1))
print('#'*50)
print("HPC with IDLE Exact Solution for 30Trips between 1CS vs. 2CS....")
print(pd.concat([cplex_30Trips1cs_OPT_IDLE_soln.describe(), cplex_30Trips2cs_OPT_IDLE_soln.describe()], axis=1))
print("-"*150)

print("PC IDLE HPC Results 30Trips between 1CS vs. 2CS....")
print(pd.concat([cplex_30Trips1cs_PC_IDLE_soln.describe(), cplex_30Trips2cs_PC_IDLE_soln.describe()], axis=1))
print("-"*150)

test = cplex_30Trips1cs_OPT_IDLE_soln.describe().loc['mean']
test['numBuses'] = cplex_30Trips1cs_OPT_IDLE_soln.describe().loc['count','gapTime']
test['time_to_best_soln'] = 6372.83
test = test.to_frame().rename(columns={"mean":"CPLEX_HPC_EXACT_30Trips1CS"})

test2 = cplex_30Trips2cs_OPT_IDLE_soln.describe().loc['mean']
test2['numBuses'] = cplex_30Trips2cs_OPT_IDLE_soln.describe().loc['count','gapTime']
test2['time_to_best_soln'] = 248718.55
test2 = test2.to_frame().rename(columns={"mean":"CPLEX_HPC_EXACT_30Trips2CS"})

result_CPLEX_HPC_30Trips = pd.concat([test, test2], axis=1)
print(result_CPLEX_HPC_30Trips)

# trips30_1cs = cplex_30Trips1cs_schedule_df.groupby(['bus_id']).agg({'trip_id':concat_str, 'dep_time': 'first', 'arr_time': 'last', 'dep_terminal': 'first', 'arr_terminal': 'last', 'difference': 'sum'})
# trips30_1cs
# trips30_1cs = cplex_30Trips1cs_schedule_df.groupby(['bus_id']).agg({'trip_id':concat_str, 'dep_time': 'first', 'arr_time': 'last', 'dep_terminal': 'first', 'arr_terminal': 'last', 'difference': 'sum'})
# trips30_1cs.to_csv("cplex_30Trips1CS_IDLE.csv")
# trips30_2cs = cplex_30Trips2cs_schedule_df.groupby(['bus_id']).agg({'trip_id':concat_str, 'dep_time': 'first', 'arr_time': 'last', 'dep_terminal': 'first', 'arr_terminal': 'last', 'difference': 'sum'})
# trips30_2cs.to_csv("cplex_30Trips2CS_IDLE.csv")
# trips30_3cs = cplex_30Trips3cs_schedule_df.groupby(['bus_id']).agg({'trip_id':concat_str, 'dep_time': 'first', 'arr_time': 'last', 'dep_terminal': 'first', 'arr_terminal': 'last', 'difference': 'sum'})
# trips30_3cs.to_csv("cplex_30Trips3CS_IDLE.csv")

# ###### 54Trips ############
# cplex_54Trips1cs_schedule_df = cplex_54Trips1cs_schedule.copy(deep=True)
# cplex_54Trips1cs_schedule_df['next_dep'] = cplex_54Trips1cs_schedule_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
# cplex_54Trips1cs_schedule_df['difference'] = cplex_54Trips1cs_schedule_df['next_dep'] - cplex_54Trips1cs_schedule_df['arr_time']
# cplex_54Trips1cs_schedule_df['difference'] = cplex_54Trips1cs_schedule_df['difference'].apply(lambda x: 0 if x < 0 else x)
# cplex_54Trips1cs_soln = cplex_54Trips1cs_schedule_df.groupby(['bus_id']).agg(
#     trips=('trip_id', concat_str),
#     numRecharge=('trip_id',countRecharge),
#     numTrips=('trip_id', countTrips),
#     gapTime=('difference', 'sum')
# )
# print(cplex_54Trips1cs_soln)
# cplex_54Trips1cs_soln.describe()

# cplex_54Trips3cs_schedule_df = cplex_54Trips3cs_schedule.copy(deep=True)
# cplex_54Trips3cs_schedule_df['next_dep'] = cplex_54Trips3cs_schedule_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
# cplex_54Trips3cs_schedule_df['difference'] = cplex_54Trips3cs_schedule_df['next_dep'] - cplex_54Trips3cs_schedule_df['arr_time']
# cplex_54Trips3cs_schedule_df['difference'] = cplex_54Trips3cs_schedule_df['difference'].apply(lambda x: 0 if x < 0 else x)
# cplex_54Trips3cs_soln = cplex_54Trips3cs_schedule_df.groupby(['bus_id']).agg(
#     trips=('trip_id', concat_str),
#     numRecharge=('trip_id',countRecharge),
#     numTrips=('trip_id', countTrips),
#     gapTime=('difference', 'sum')
# )
# print(cplex_54Trips3cs_soln)
# # cplex_54Trips3cs_soln.describe()
# print("Time limit Exact Solution for 54Trips between 1CS vs. 3CS....")
# print(pd.concat([cplex_54Trips1cs_soln.describe(), cplex_54Trips3cs_soln.describe()], axis=1))
# print("-"*150)

# test = cplex_54Trips1cs_soln.describe().loc['mean']
# test['numBuses'] = cplex_54Trips1cs_soln.describe().loc['count','gapTime']
# test['time_to_best_soln'] = 3843.17
# test = test.to_frame().rename(columns={"mean":"CPLEX_54Trips1CS"})

# test2 = cplex_54Trips3cs_soln.describe().loc['mean']
# test2['numBuses'] = cplex_54Trips3cs_soln.describe().loc['count','gapTime']
# test2['time_to_best_soln'] = 4910.53
# test2 = test2.to_frame().rename(columns={"mean":"CPLEX_54Trips3CS"})

# result_CPLEX_54Trips = pd.concat([test, test2], axis=1)
# print(result_CPLEX_54Trips)

# ###### 54Trips IDLE OPTIMAL ############
# cplex_54Trips1cs_schedule_OPT_df = cplex_54Trips1cs_schedule_OPT.copy(deep=True)
# cplex_54Trips1cs_schedule_OPT_df['next_dep'] = cplex_54Trips1cs_schedule_OPT_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
# cplex_54Trips1cs_schedule_OPT_df['difference'] = cplex_54Trips1cs_schedule_OPT_df['next_dep'] - cplex_54Trips1cs_schedule_OPT_df['arr_time']
# cplex_54Trips1cs_schedule_OPT_df['difference'] = cplex_54Trips1cs_schedule_OPT_df['difference'].apply(lambda x: 0 if x < 0 else x)
# cplex_54Trips1cs_OPT_soln = cplex_54Trips1cs_schedule_OPT_df.groupby(['bus_id']).agg(
#     trips=('trip_id', concat_str),
#     numRecharge=('trip_id',countRecharge),
#     numTrips=('trip_id', countTrips),
#     gapTime=('difference', 'sum')
# )
# print(cplex_54Trips1cs_OPT_soln)
# # cplex_54Trips1cs_OPT_soln.describe()

# cplex_54Trips3cs_schedule_OPT_df = cplex_54Trips3cs_schedule_OPT.copy(deep=True)
# cplex_54Trips3cs_schedule_OPT_df['next_dep'] = cplex_54Trips3cs_schedule_OPT_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
# cplex_54Trips3cs_schedule_OPT_df['difference'] = cplex_54Trips3cs_schedule_OPT_df['next_dep'] - cplex_54Trips3cs_schedule_OPT_df['arr_time']
# cplex_54Trips3cs_schedule_OPT_df['difference'] = cplex_54Trips3cs_schedule_OPT_df['difference'].apply(lambda x: 0 if x < 0 else x)
# cplex_54Trips3cs_OPT_soln = cplex_54Trips3cs_schedule_OPT_df.groupby(['bus_id']).agg(
#     trips=('trip_id', concat_str),
#     numRecharge=('trip_id',countRecharge),
#     numTrips=('trip_id', countTrips),
#     gapTime=('difference', 'sum')
# )
# print(cplex_54Trips3cs_OPT_soln)
# cplex_54Trips3cs_OPT_soln.describe()
# print("HPC Exact Solution for 54Trips between 1CS vs. 3CS....")
# print(pd.concat([cplex_54Trips1cs_OPT_soln.describe(), cplex_54Trips3cs_OPT_soln.describe()], axis=1))
# print("-"*150)

# test = cplex_54Trips1cs_OPT_soln.describe().loc['mean']
# test['numBuses'] = cplex_54Trips1cs_OPT_soln.describe().loc['count','gapTime']
# test['time_to_best_soln'] = 248605.91
# test = test.to_frame().rename(columns={"mean":"CPLEX_HPC_EXACT_54Trips1CS"})

# test2 = cplex_54Trips3cs_OPT_soln.describe().loc['mean']
# test2['numBuses'] = cplex_54Trips3cs_OPT_soln.describe().loc['count','gapTime']
# test2['time_to_best_soln'] = 248400.18
# test2 = test2.to_frame().rename(columns={"mean":"CPLEX_HPC_EXACT_54Trips3CS"})

# result_CPLEX_HPC_54Trips = pd.concat([test, test2], axis=1)
# print(result_CPLEX_HPC_54Trips)

# ###### 54Trips IDLE OPTIMAL ############
# cplex_54Trips1cs_schedule_OPT_df = cplex_54Trips1cs_schedule_OPT.copy(deep=True)
# cplex_54Trips1cs_schedule_OPT_df['next_dep'] = cplex_54Trips1cs_schedule_OPT_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
# cplex_54Trips1cs_schedule_OPT_df['difference'] = cplex_54Trips1cs_schedule_OPT_df['next_dep'] - cplex_54Trips1cs_schedule_OPT_df['arr_time']
# cplex_54Trips1cs_schedule_OPT_df['difference'] = cplex_54Trips1cs_schedule_OPT_df['difference'].apply(lambda x: 0 if x < 0 else x)
# cplex_54Trips1cs_OPT_soln = cplex_54Trips1cs_schedule_OPT_df.groupby(['bus_id']).agg(
#     trips=('trip_id', concat_str),
#     numRecharge=('trip_id',countRecharge),
#     numTrips=('trip_id', countTrips),
#     gapTime=('difference', 'sum')
# )
# print(cplex_54Trips1cs_OPT_soln)
# # cplex_54Trips1cs_OPT_soln.describe()

# cplex_54Trips3cs_schedule_OPT_df = cplex_54Trips3cs_schedule_OPT.copy(deep=True)
# cplex_54Trips3cs_schedule_OPT_df['next_dep'] = cplex_54Trips3cs_schedule_OPT_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
# cplex_54Trips3cs_schedule_OPT_df['difference'] = cplex_54Trips3cs_schedule_OPT_df['next_dep'] - cplex_54Trips3cs_schedule_OPT_df['arr_time']
# cplex_54Trips3cs_schedule_OPT_df['difference'] = cplex_54Trips3cs_schedule_OPT_df['difference'].apply(lambda x: 0 if x < 0 else x)
# cplex_54Trips3cs_OPT_soln = cplex_54Trips3cs_schedule_OPT_df.groupby(['bus_id']).agg(
#     trips=('trip_id', concat_str),
#     numRecharge=('trip_id',countRecharge),
#     numTrips=('trip_id', countTrips),
#     gapTime=('difference', 'sum')
# )
# print(cplex_54Trips3cs_OPT_soln)
# cplex_54Trips3cs_OPT_soln.describe()
# print("HPC Exact Solution for 54Trips between 1CS vs. 3CS....")
# print(pd.concat([cplex_54Trips1cs_OPT_soln.describe(), cplex_54Trips3cs_OPT_soln.describe()], axis=1))
# print("-"*150)

# test = cplex_54Trips1cs_OPT_soln.describe().loc['mean']
# test['numBuses'] = cplex_54Trips1cs_OPT_soln.describe().loc['count','gapTime']
# test['time_to_best_soln'] = 248605.91
# test = test.to_frame().rename(columns={"mean":"CPLEX_HPC_EXACT_54Trips1CS"})

# test2 = cplex_54Trips3cs_OPT_soln.describe().loc['mean']
# test2['numBuses'] = cplex_54Trips3cs_OPT_soln.describe().loc['count','gapTime']
# test2['time_to_best_soln'] = 248400.18
# test2 = test2.to_frame().rename(columns={"mean":"CPLEX_HPC_EXACT_54Trips3CS"})

# result_CPLEX_HPC_54Trips = pd.concat([test, test2], axis=1)
# print(result_CPLEX_HPC_54Trips)
# ################################################################################################################
# cplex_54Trips1cs_schedule_PC_IDLE_df = cplex_54Trips1cs_schedule_OPT_PC_IDLE.copy(deep=True)
# cplex_54Trips1cs_schedule_PC_IDLE_df['next_dep'] = cplex_54Trips1cs_schedule_PC_IDLE_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
# cplex_54Trips1cs_schedule_PC_IDLE_df['difference'] = cplex_54Trips1cs_schedule_PC_IDLE_df['next_dep'] - cplex_54Trips1cs_schedule_PC_IDLE_df['arr_time']
# cplex_54Trips1cs_schedule_PC_IDLE_df['difference'] = cplex_54Trips1cs_schedule_PC_IDLE_df['difference'].apply(lambda x: 0 if x < 0 else x)
# cplex_54Trips1cs_PC_IDLE_soln = cplex_54Trips1cs_schedule_PC_IDLE_df.groupby(['bus_id']).agg(
#     trips=('trip_id', concat_str),
#     numRecharge=('trip_id',countRecharge),
#     numTrips=('trip_id', countTrips),
#     gapTime=('difference', 'sum')
# )
# print(cplex_54Trips1cs_PC_IDLE_soln)
# print("PC IDLE HPC Results between Previous Defined vs. New Defined")
# print(pd.concat([cplex_54Trips1cs_OPT_soln.describe(), cplex_54Trips1cs_PC_IDLE_soln.describe()], axis=1))
# print('#'*50)

# cplex_54Trips3cs_schedule_PC_IDLE_df = cplex_54Trips3cs_schedule_OPT_PC_IDLE.copy(deep=True)
# cplex_54Trips3cs_schedule_PC_IDLE_df['next_dep'] = cplex_54Trips3cs_schedule_PC_IDLE_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
# cplex_54Trips3cs_schedule_PC_IDLE_df['difference'] = cplex_54Trips3cs_schedule_PC_IDLE_df['next_dep'] - cplex_54Trips3cs_schedule_PC_IDLE_df['arr_time']
# cplex_54Trips3cs_schedule_PC_IDLE_df['difference'] = cplex_54Trips3cs_schedule_PC_IDLE_df['difference'].apply(lambda x: 0 if x < 0 else x)
# cplex_54Trips3cs_PC_IDLE_soln = cplex_54Trips3cs_schedule_PC_IDLE_df.groupby(['bus_id']).agg(
#     trips=('trip_id', concat_str),
#     numRecharge=('trip_id',countRecharge),
#     numTrips=('trip_id', countTrips),
#     gapTime=('difference', 'sum')
# )
# print(cplex_54Trips3cs_PC_IDLE_soln)
# print("PC IDLE HPC Results between Previous Defined vs. New Defined")
# print(pd.concat([cplex_54Trips3cs_OPT_soln.describe(), cplex_54Trips3cs_PC_IDLE_soln.describe()], axis=1))
# print('#'*50)

# cplex_54Trips3cs_PC_IDLE_soln.describe()
# print("HPC PC_IDLES Exact Solution for 54Trips between 1CS vs. 3CS....")
# print(pd.concat([cplex_54Trips1cs_PC_IDLE_soln.describe(), cplex_54Trips3cs_PC_IDLE_soln.describe()], axis=1))
# print("-"*150)

#################### Difference #############################


# ###############################################################################################################
# # test = cplexCS1_IDLE_soln.describe().loc['mean']
# # test['numBuses'] = cplexCS1_IDLE_soln.describe().loc['count','gapTime']
# # test['time_to_best_soln'] = 2.92
# # test = test.to_frame().rename(columns={"mean":"CPLEX_10Trips1CS"})

# # test2 = cplexCS2_IDLE_soln.describe().loc['mean']
# # test2['numBuses'] = cplexCS2_IDLE_soln.describe().loc['count','gapTime']
# # test2['time_to_best_soln'] = 3.8
# # test2 = test2.to_frame().rename(columns={"mean":"CPLEX_10Trips2CS"})

# # result_HPC_CPLEX_10Trips = pd.concat([test, test2], axis=1)
# # print(result_HPC_CPLEX_10Trips)

# # test = cplex_30Trips1cs_soln.describe().loc['mean']
# # test['numBuses'] = cplex_30Trips1cs_soln.describe().loc['count','gapTime']
# # test['time_to_best_soln'] = 1360.34
# # test = test.to_frame().rename(columns={"mean":"CPLEX_30Trips1CS"})

# # test2 = cplexCS2_IDLE_soln.describe().loc['mean']
# # test2['numBuses'] = cplexCS2_IDLE_soln.describe().loc['count','gapTime']
# # test2['time_to_best_soln'] = 1616.89
# # test2 = test2.to_frame().rename(columns={"mean":"CPLEX_30Trips2CS"})

# # result_CPLEX_30Trips = pd.concat([test, test2], axis=1)
# # print(result_CPLEX_30Trips)

# # test = cplex_54Trips1cs_soln.describe().loc['mean']
# # test['numBuses'] = cplex_54Trips1cs_soln.describe().loc['count','gapTime']
# # test['time_to_best_soln'] = 3843.17
# # test = test.to_frame().rename(columns={"mean":"CPLEX_54Trips1CS"})

# # test2 = cplex_54Trips3cs_soln.describe().loc['mean']
# # test2['numBuses'] = cplex_54Trips3cs_soln.describe().loc['count','gapTime']
# # test2['time_to_best_soln'] = 4910.53
# # test2 = test2.to_frame().rename(columns={"mean":"CPLEX_54Trips3CS"})

# # result_CPLEX_54Trips = pd.concat([test, test2], axis=1)
# # print(result_CPLEX_54Trips)
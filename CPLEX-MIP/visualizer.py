# -*- coding: utf-8 -*-
"""
Created on Tue May 13 22:04:52 2025

@author: gozalid
"""

import matplotlib.ticker as ticker
import pandas as pd
def visualizeOptimization(soln_filename, title):
    print(f"starting {title}...")
    with open(soln_filename, "r") as f:
        content = f.readlines()
    gaps = []
    time = []
    i = 0
    diffs = []
    opt_gap_identifier = "Elapsed time = "
    timegap = -1
    initialize = "Log started (V22.1.1.0)"
    for idx, line in enumerate(content):
        lines = line.split()
        if line.startswith(initialize) or "22.1.1.0" in line:
            tempgaps = []
            temptime = []
            tempdiffs = []
            i = 0
        elif line.startswith("Total (root+branch&cut) ="):
            gaps.append(tempgaps)
            time.append(temptime)
            diffs.append(tempdiffs)
        elif line.startswith("The value of the objective function is (Total Cost): "):
            cost = line.split(" ")[-1]
            print(cost)
            timegap = int(cost)
        if len(lines) > 2:
            if lines[-1].endswith("%"):
                # print(lines)
                i+=1
                gap = float(lines[-1].split("%")[0])
                tempgaps.append(gap)
                # print(f"gaps = {tempgaps}")
            if (line.startswith(opt_gap_identifier) and "solutions = " in line) or (line.startswith("  Real time") and content[idx+1].startswith("  Sync time")):
                elapse_time = float(lines[3])
                temptime.append(float(elapse_time))
                if len(tempdiffs) > 0 and i > 0:
                    step = (elapse_time-temptime[-1])/i
                    tempdiffs += [x*step+temptime[-1] for x in range(i)]
                else:
                    step = elapse_time/(i+1) 
                    tempdiffs += [x*step for x in range(i)]
                i = 0
            # else:
            #     print(lines)
    
    print(f"diff = {diffs}.... gaps = {gaps}..timegap = {timegap}.")
    test = pd.DataFrame({"time": diffs[0], "opt_gap": gaps[0]})
    ax = test.groupby('opt_gap')[['opt_gap', 'time']].last().sort_values(by=['time']).plot(x='time', y='opt_gap', xlabel="Time", ylabel="Opt Gap (%)", legend=False, title=title)
    fig = ax.get_figure()
    # fig.savefig(f"{saveFigDir}{title}-Optimality_Gap")
    return ax, timegap

def apply_custom_shift(group):
    dep_shift = group['dep_time'].shift(-1)
    arr_shift = group['arr_time'].shift(-1)
    trip_shift = group['trip_id'].shift(-1)
    group['next_dep'] = group.apply(
        lambda row: arr_shift[row.name] if trip_shift[row.name] != None and trip_shift[row.name].startswith('CS') else dep_shift[row.name],
        axis=1
    )
    return group

filepath = r".\\MIP-"
filename = f"{filepath}Data\\final_test.xlsx"
sheetname = "10Trips"

NTrips = 10 #trips
NTerms = 2 #terminals
NCS = 3 #recharging stations
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
minBuses = 3
maxBuses = 11
cplex_pareto_10Trips3CS = []
for i in range(minBuses, maxBuses):
    soln_filename = f"{filepath}Results\\TimeLimit\\FINAL\\10TripsEVeh3CS_{i}Buses.txt"
    title = f'CPLEX-10Trips3CS_{i}Buses'
    cplex_3cs_schedule, cplex_3cs_fig = load_data_hyperparameter(df2, title, soln_filename, R)
    cplex_pareto_10Trips3CS.append(cplex_3cs_schedule)

    #ax = visualizeOptimization(soln_filename, title)
cplex10T3CS_nbus = [x for x in range(minBuses, maxBuses)]
cplex10T3CS_gap = []
cplex10T3CS_solns = []
for cplex_10T3CS in cplex_pareto_10Trips3CS:
    ###### 20Trips ##########
    cplex_10T3CS_df = cplex_10T3CS.groupby('bus_id', group_keys=False).apply(apply_custom_shift)
    cplex_10T3CS_df['next_dep'] = cplex_10T3CS_df['next_dep'].fillna(0)
    cplex_10T3CS_df['difference'] = cplex_10T3CS_df['next_dep'] - cplex_10T3CS_df['arr_time']
    cplex_10T3CS_df['difference'] = cplex_10T3CS_df['difference'].apply(lambda x: 0 if x < 0 else x)
    cplex_10T3CS_soln = cplex_10T3CS_df.groupby(['bus_id']).agg(
        trips=('trip_id', concat_str),
        numRecharge=('trip_id',countRecharge),
        numTrips=('trip_id', countTrips),
        gapTime=('difference', 'sum')
    )
    cplex10T3CS_solns.append(cplex_10T3CS_soln)
    cplex10T3CS_gap.append(cplex_10T3CS_soln['gapTime'].sum())
test = pd.DataFrame({"NumBuses[#]": cplex10T3CS_nbus, "IdleTime(Gap)[sec]": cplex10T3CS_gap})
ax = test.sort_values(by=['NumBuses[#]']).plot(x='NumBuses[#]', y='IdleTime(Gap)[sec]', xlabel="numBuses", ylabel="IdleTime(Gap)[sec]", legend=False, title="10Trips3CS_ParetoFrontier")
# Set integer x-axis ticks only
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
fig = ax.get_figure()

# ############################################### 20 TRIPS Single CS ###########################################
# filepath = r".\\MIP-"
# filename = f"{filepath}Data\\data_20_40trips.xlsx"
# sheetname = "20Trips"

# NTrips = 20 #trips
# NTerms = 2 #terminals
# NCS = 1 #recharging stations
# CHARGING_TIME = 100 #minutes
# NRechargeCycle = 3 #cycle
# D_MAX = 350 #minutes of operation
# DEPOT = 1 #depot
# # print(filename)
# df = pd.read_excel(filename, sheet_name="20Trips", usecols="A:G", nrows=NTrips+NCS+DEPOT)
# term = pd.read_excel(filename, sheet_name="20Trips", usecols="AB:AD", skiprows=31, nrows=NTerms)
# gamma = pd.read_excel(filename, sheet_name="feasible_mat", usecols="B:V", nrows=NTrips+DEPOT)
# delta = pd.read_excel(filename, sheet_name="duration_mat", usecols="B:W", nrows=NTrips+DEPOT+NCS)
# phi = pd.read_excel(filename, sheet_name="Recharged_Feasible_matv3", usecols="C:W", nrows=(NTrips+DEPOT)*NCS)
# # zeta = pd.read_excel(filename, sheet_name="10Trips", usecols="", skiprows=, nrows=()))
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
# # # soln_filename = f"{filepath}cplex_results\\10TripsEVeh1CS_Eq1.txt"
# # soln_filename = f"{filepath}Results\\TimeLimit\\1May25\\20TripsEVeh1CS_Eq1.txt"
# # title = 'CPLEX-20Trips1CS'
# # cplex_20Trips1cs_schedule, cplex_20Trips1cs_fig = load_data_hyperparameter(df2, title, soln_filename, R)

# # soln_filename = f"{filepath}Results\\HPC\\RB_20T1CS.asc"
# # title = 'CPLEX-20Trips1CS_HPC'
# # cplex_20Trips1cs_schedule_OPT, cplex_20Trips1cs_fig_OPT = load_data_hyperparameter(df2, title, soln_filename, R)

# minBuses = 8
# maxBuses = 14
# cplex_pareto_20Trips1CS = []
# for i in range(minBuses, maxBuses):
#     soln_filename = f"{filepath}Results\\TimeLimit\\FINAL\\20TripsEVeh1CS_{i}Buses.txt"
#     title = f'CPLEX-20Trips1CS_{i}Buses'
#     cplex_1cs_schedule, cplex_1cs_fig = load_data_hyperparameter(df2, title, soln_filename, R)
#     cplex_pareto_20Trips1CS.append(cplex_1cs_schedule)

#     ax = visualizeOptimization(soln_filename, title)

# cplex20T1CS_nbus = [x for x in range(minBuses, maxBuses)]
# cplex20T1CS_gap = []
# cplex20T1CS_solns = []
# for cplex_20T1CS in cplex_pareto_20Trips1CS:
#     ###### 20Trips ##########
#     # cplex_10T1CS_df = cplex_10T1CS.copy(deep=True)
#     cplex_20T1CS_df = cplex_20T1CS.groupby('bus_id', group_keys=False).apply(apply_custom_shift)
#     cplex_20T1CS_df['next_dep'] = cplex_20T1CS_df['next_dep'].fillna(0)
#     cplex_20T1CS_df['difference'] = cplex_20T1CS_df['next_dep'] - cplex_20T1CS_df['arr_time']
#     cplex_20T1CS_df['difference'] = cplex_20T1CS_df['difference'].apply(lambda x: 0 if x < 0 else x)
#     cplex_20T1CS_soln = cplex_20T1CS_df.groupby(['bus_id']).agg(
#         trips=('trip_id', concat_str),
#         numRecharge=('trip_id',countRecharge),
#         numTrips=('trip_id', countTrips),
#         gapTime=('difference', 'sum')
#     )
#     cplex20T1CS_solns.append(cplex_20T1CS_soln)
#     cplex20T1CS_gap.append(cplex_20T1CS_soln['gapTime'].sum())

# test = pd.DataFrame({"NumBuses[#]": cplex20T1CS_nbus, "IdleTime(Gap)[sec]": cplex20T1CS_gap})
# ax = test.sort_values(by=['NumBuses[#]']).plot(
#     x='NumBuses[#]', 
#     y='IdleTime(Gap)[sec]', 
#     xlabel="numBuses", 
#     ylabel="IdleTime(Gap)[sec]", 
#     legend=False, 
#     title=f"20Trips1CS_ParetoFrontier", 
#     style=".-"
# )
# # Set integer x-axis ticks only
# ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
# fig = ax.get_figure()


# ############################################### 20 TRIPS MULTIPLE CSs ###########################################
# NTrips = 20 #trips
# NTerms = 2 #terminals
# CHARGING_TIME = 100 #minutes
# NRechargeCycle = 3 #cycle
# D_MAX = 350 #minutes of operation
# DEPOT = 1 #depot
# NCS = 3 #recharging stations
# df = pd.read_excel(filename, sheet_name="20Trips", usecols="A:G", nrows=NTrips+NCS+DEPOT)
# term = pd.read_excel(filename, sheet_name="20Trips", usecols="AB:AD", skiprows=31, nrows=NTerms)
# gamma = pd.read_excel(filename, sheet_name="feasible_mat", usecols="B:V", nrows=NTrips+DEPOT)
# delta = pd.read_excel(filename, sheet_name="duration_mat", usecols="B:Y", nrows=NTrips+DEPOT+NCS)
# phi = pd.read_excel(filename, sheet_name="Recharged_Feasible_matv3", usecols="C:W", nrows=(NTrips+DEPOT)*NCS)
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

# # soln_filename = f"{filepath}Results\\TimeLimit\\1May25\\20TripsEVeh3CS_Eq1.txt"
# # title = 'CPLEX-20Trips3CS'
# # cplex_20Trips3cs_schedule, cplex_20Trips3cs_fig = load_data_hyperparameter(df2, title, soln_filename, R)

# # soln_filename = f"{filepath}Results\\HPC\\RB_20T3CS.asc"
# # title = 'CPLEX-20Trips3CS_HPC'
# # cplex_20Trips3cs_schedule_OPT, cplex_20Trips3cs_fig_OPT = load_data_hyperparameter(df2, title, soln_filename, R)

# minBuses = 7
# maxBuses = 14
# cplex_pareto_20Trips3CS = []
# for i in range(minBuses, maxBuses):
#     soln_filename = f"{filepath}Results\\TimeLimit\\FINAL\\20TripsEVeh3CS_{i}Buses.txt"
#     title = f'CPLEX-20Trips3CS_{i}Buses'
#     cplex_3CS_schedule, cplex_3CS_fig = load_data_hyperparameter(df2, title, soln_filename, R)
#     cplex_pareto_20Trips3CS.append(cplex_3CS_schedule)

#     #ax = visualizeOptimization(soln_filename, title)

# cplex20T3CS_nbus = [x for x in range(minBuses, maxBuses)]
# cplex20T3CS_gap = []
# cplex20T3CS_solns = []
# for cplex_20T3CS in cplex_pareto_20Trips3CS:
#     ###### 20Trips ##########
#     # cplex_10T1CS_df = cplex_10T1CS.copy(deep=True)
#     cplex_20T3CS_df = cplex_20T3CS.groupby('bus_id', group_keys=False).apply(apply_custom_shift)
#     cplex_20T3CS_df['next_dep'] = cplex_20T3CS_df['next_dep'].fillna(0)
#     cplex_20T3CS_df['difference'] = cplex_20T3CS_df['next_dep'] - cplex_20T3CS_df['arr_time']
#     cplex_20T3CS_df['difference'] = cplex_20T3CS_df['difference'].apply(lambda x: 0 if x < 0 else x)
#     cplex_20T3CS_soln = cplex_20T3CS_df.groupby(['bus_id']).agg(
#         trips=('trip_id', concat_str),
#         numRecharge=('trip_id',countRecharge),
#         numTrips=('trip_id', countTrips),
#         gapTime=('difference', 'sum')
#     )
#     cplex20T3CS_solns.append(cplex_20T3CS_soln)
#     cplex20T3CS_gap.append(cplex_20T3CS_soln['gapTime'].sum())

# test = pd.DataFrame({"NumBuses[#]": cplex20T3CS_nbus, "IdleTime(Gap)[sec]": cplex20T3CS_gap})
# ax = test.sort_values(by=['NumBuses[#]']).plot(
#     x='NumBuses[#]', 
#     y='IdleTime(Gap)[sec]', 
#     xlabel="numBuses", 
#     ylabel="IdleTime(Gap)[sec]", 
#     legend=False, 
#     title=f"20Trips3CS_ParetoFrontier", 
#     style=".-"
# )
# # Set integer x-axis ticks only
# ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
# fig = ax.get_figure()



# ############################################### 30 TRIPS Single CS ###########################################
# filepath = r".\\MIP-"
# filename = f"{filepath}Data\\final_test.xlsx"
# sheetname = "30Trips"

# NTrips = 30 #trips
# NTerms = 4 #terminals
# NCS = 1 #recharging stations
# CHARGING_TIME = 100 #minutes
# NRechargeCycle = 3 #cycle
# D_MAX = 350 #minutes of operation
# DEPOT = 1 #depot
# df = pd.read_excel(filename, sheet_name="test", usecols="A:G", nrows=NTrips+NCS+DEPOT)
# term = pd.read_excel(filename, sheet_name="test", usecols="J:L", skiprows=10, nrows=NTerms)
# gamma = pd.read_excel(filename, sheet_name="feasible_mat", usecols="B:AF", nrows=NTrips+DEPOT)
# delta = pd.read_excel(filename, sheet_name="duration_mat", usecols="B:AG", nrows=NTrips+DEPOT+NCS)
# phi = pd.read_excel(filename, sheet_name="Recharged_Feasible_mat", usecols="C:AG", nrows=(NTrips+DEPOT)*NCS)
# df2 = df.set_index('trip_id')
# gamma = gamma.to_numpy()
# delta = delta.to_numpy()
# phi = phi.to_numpy()

# trips_df = df.copy()
# s = len(trips_df)
# k = (s // 2) - 2

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

# # ################# Time Limit ################
# # soln_filename = f"{filepath}Results\\TimeLimit\\RB\\30TripsEVeh1CS_Eq1_RB.txt"
# # title = 'CPLEX-30Trips1CS'
# # cplex_30Trips1cs_schedule, cplex_30Trips1cs_fig = load_data_hyperparameter(df2, title, soln_filename, R)

# # ############### HPC Z Folder ###############
# # soln_path = f"{filepath}Results\\HPC"
# # soln_filename = f"{soln_path}\\RB_30T1CS.asc"
# # title = 'CPLEX-30Trips1CS_HPC'
# # cplex_30Trips1cs_schedule_OPT, cplex_30Trips1cs_fig_OPT = load_data_hyperparameter(df2, title, soln_filename, R)
# minBuses = 10
# maxBuses = 16

# cplex_pareto_30Trips1CS = []
# for i in range(minBuses, maxBuses):
#     soln_filename = f"{filepath}Results\\TimeLimit\\FINAL\\30TripsEVeh1CS_{i}Buses.asc"
#     title = f'CPLEX-30Trips1CS_{i}Buses'
#     cplex_1CS_schedule, cplex_1CS_fig = load_data_hyperparameter(df2, title, soln_filename, R)
#     cplex_pareto_30Trips1CS.append(cplex_1CS_schedule)

#     ax = visualizeOptimization(soln_filename, title)

# cplex30T1CS_nbus = [x for x in range(minBuses, maxBuses)]
# cplex30T1CS_gap = []
# cplex30T1CS_solns = []
# for cplex_30T1CS in cplex_pareto_30Trips1CS:
#     ###### 30Trips ##########
#     cplex_30T1CS_df = cplex_30T1CS.copy(deep=True)
#     cplex_30T1CS_df['next_dep'] = cplex_30T1CS_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
#     cplex_30T1CS_df['difference'] = cplex_30T1CS_df['next_dep'] - cplex_30T1CS_df['arr_time']
#     cplex_30T1CS_df['difference'] = cplex_30T1CS_df['difference'].apply(lambda x: 0 if x < 0 else x)
#     cplex_30T1CS_soln = cplex_30T1CS_df.groupby(['bus_id']).agg(
#         trips=('trip_id', concat_str),
#         numRecharge=('trip_id',countRecharge),
#         numTrips=('trip_id', countTrips),
#         gapTime=('difference', 'sum')
#     )
#     cplex30T1CS_solns.append(cplex_30T1CS_soln)
#     cplex30T1CS_gap.append(cplex_30T1CS_soln['gapTime'].sum())

# test = pd.DataFrame({"NumBuses[#]": cplex30T1CS_nbus, "IdleTime(Gap)[sec]": cplex30T1CS_gap})
# ax = test.sort_values(by=['NumBuses[#]']).plot(
#     x='NumBuses[#]', 
#     y='IdleTime(Gap)[sec]', 
#     xlabel="numBuses", 
#     ylabel="IdleTime(Gap)[sec]", 
#     legend=False, 
#     title=f"30Trips1CS_ParetoFrontier", 
#     style=".-"
# )
# # Set integer x-axis ticks only
# ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
# fig = ax.get_figure()


# ############################################### 30 TRIPS 3CSs ###########################################
# filename = f"{filepath}Data\\final_test.xlsx"
# sheetname = "30Trips"
# filepath = r".\\MIP-"
# filename = f"{filepath}Data\\final_test.xlsx"
# sheetname = "30Trips"


# NTrips = 30 #trips
# NTerms = 4 #terminals
# CHARGING_TIME = 100 #minutes
# NRechargeCycle = 3 #cycle
# D_MAX = 350 #minutes of operation
# DEPOT = 1 #depot
# NCS = 3 #recharging stations
# df = pd.read_excel(filename, sheet_name="test", usecols="A:G", nrows=NTrips+NCS+DEPOT)
# term = pd.read_excel(filename, sheet_name="test", usecols="J:L", skiprows=10, nrows=NTerms)
# gamma = pd.read_excel(filename, sheet_name="feasible_mat", usecols="B:AF", nrows=NTrips+DEPOT)
# delta = pd.read_excel(filename, sheet_name="duration_mat", usecols="B:AI", nrows=NTrips+DEPOT+NCS)
# phi = pd.read_excel(filename, sheet_name="Recharged_Feasible_mat", usecols="C:AG", nrows=(NTrips+DEPOT)*NCS)
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

# # soln_filename = f"{filepath}Results\\TimeLimit\\RB\\30TripsEVeh3CS_Eq1_RB.txt"
# # title = 'CPLEX-30Trips3CS_IDLE'
# # cplex_30Trips3cs_schedule, cplex_30Trips3cs_fig = load_data_hyperparameter(df2, title, soln_filename, R)

# # soln_path = f"{filepath}Results\\HPC"
# # soln_filename = f"{soln_path}\\RB_30T3CS.asc"
# # title = 'CPLEX-30Trips3CS_HPC'
# # cplex_30Trips3cs_schedule_OPT, cplex_30Trips3cs_fig_OPT = load_data_hyperparameter(df2, title, soln_filename, R)

# minBuses = 9
# maxBuses = 14
# cplex_pareto_30Trips3CS = []
# for i in range(minBuses, maxBuses):
#     soln_filename = f"{filepath}Results\\TimeLimit\\FINAL\\30TripsEVeh3CS_{i}Buses.asc"
#     title = f'CPLEX-30Trips3CS_{i}Buses'
#     cplex_3CS_schedule, cplex_3CS_fig = load_data_hyperparameter(df2, title, soln_filename, R)
#     cplex_pareto_30Trips3CS.append(cplex_3CS_schedule)

#     ax = visualizeOptimization(soln_filename, title)

# cplex30T3CS_nbus = [x for x in range(minBuses, maxBuses)]
# cplex30T3CS_gap = []
# cplex30T3CS_solns = []
# for cplex_30T3CS in cplex_pareto_30Trips3CS:
#     ###### 30Trips ##########
#     cplex_30T3CS_df = cplex_30T3CS.groupby('bus_id', group_keys=False).apply(apply_custom_shift)
#     cplex_30T3CS_df['next_dep'] = cplex_30T3CS_df['next_dep'].fillna(0)
#     cplex_30T3CS_df['difference'] = cplex_30T3CS_df['next_dep'] - cplex_30T3CS_df['arr_time']
#     cplex_30T3CS_df['difference'] = cplex_30T3CS_df['difference'].apply(lambda x: 0 if x < 0 else x)
#     cplex_30T3CS_soln = cplex_30T3CS_df.groupby(['bus_id']).agg(
#         trips=('trip_id', concat_str),
#         numRecharge=('trip_id',countRecharge),
#         numTrips=('trip_id', countTrips),
#         gapTime=('difference', 'sum')
#     )
#     cplex30T3CS_solns.append(cplex_30T3CS_soln)
#     cplex30T3CS_gap.append(cplex_30T3CS_soln['gapTime'].sum())

# test = pd.DataFrame({"NumBuses[#]": cplex30T3CS_nbus, "IdleTime(Gap)[sec]": cplex30T3CS_gap})
# ax = test.sort_values(by=['NumBuses[#]']).plot(
#     x='NumBuses[#]', 
#     y='IdleTime(Gap)[sec]', 
#     xlabel="numBuses", 
#     ylabel="IdleTime(Gap)[sec]", 
#     legend=False, 
#     title=f"30Trips3CS_ParetoFrontier", 
#     style=".-"
# )
# # Set integer x-axis ticks only
# ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
# fig = ax.get_figure()

# ###### 30Trips ##########
# cplex_30Trips1cs_schedule_df = cplex_30Trips1cs_schedule.copy(deep=True)
# cplex_30Trips1cs_schedule_df['next_dep'] = cplex_30Trips1cs_schedule_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
# cplex_30Trips1cs_schedule_df['difference'] = cplex_30Trips1cs_schedule_df['next_dep'] - cplex_30Trips1cs_schedule_df['arr_time']
# cplex_30Trips1cs_schedule_df['difference'] = cplex_30Trips1cs_schedule_df['difference'].apply(lambda x: 0 if x < 0 else x)
# cplex_30Trips1cs_soln = cplex_30Trips1cs_schedule_df.groupby(['bus_id']).agg(
#     trips=('trip_id', concat_str),
#     numRecharge=('trip_id',countRecharge),
#     numTrips=('trip_id', countTrips),
#     gapTime=('difference', 'sum')
# )
# ###### 30Trips ############
# cplex_30Trips1cs_schedule_df = cplex_30Trips1cs_schedule_OPT.copy(deep=True)
# cplex_30Trips1cs_schedule_df['next_dep'] = cplex_30Trips1cs_schedule_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
# cplex_30Trips1cs_schedule_df['difference'] = cplex_30Trips1cs_schedule_df['next_dep'] - cplex_30Trips1cs_schedule_df['arr_time']
# cplex_30Trips1cs_schedule_df['difference'] = cplex_30Trips1cs_schedule_df['difference'].apply(lambda x: 0 if x < 0 else x)
# cplex_30Trips1cs_soln_OPT = cplex_30Trips1cs_schedule_df.groupby(['bus_id']).agg(
#     trips=('trip_id', concat_str),
#     numRecharge=('trip_id',countRecharge),
#     numTrips=('trip_id', countTrips),
#     gapTime=('difference', 'sum')
# )

# cplex_30Trips3cs_schedule_df = cplex_30Trips3cs_schedule.copy(deep=True)
# cplex_30Trips3cs_schedule_df['next_dep'] = cplex_30Trips3cs_schedule_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
# cplex_30Trips3cs_schedule_df['difference'] = cplex_30Trips3cs_schedule_df['next_dep'] - cplex_30Trips3cs_schedule_df['arr_time']
# cplex_30Trips3cs_schedule_df['difference'] = cplex_30Trips3cs_schedule_df['difference'].apply(lambda x: 0 if x < 0 else x)
# cplex_30Trips3cs_soln = cplex_30Trips3cs_schedule_df.groupby(['bus_id']).agg(
#     trips=('trip_id', concat_str),
#     numRecharge=('trip_id',countRecharge),
#     numTrips=('trip_id', countTrips),
#     gapTime=('difference', 'sum')
# )

# cplex_30Trips3cs_schedule_df = cplex_30Trips3cs_schedule_OPT.copy(deep=True)
# cplex_30Trips3cs_schedule_df['next_dep'] = cplex_30Trips3cs_schedule_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
# cplex_30Trips3cs_schedule_df['difference'] = cplex_30Trips3cs_schedule_df['next_dep'] - cplex_30Trips3cs_schedule_df['arr_time']
# cplex_30Trips3cs_schedule_df['difference'] = cplex_30Trips3cs_schedule_df['difference'].apply(lambda x: 0 if x < 0 else x)
# cplex_30Trips3cs_soln_OPT = cplex_30Trips3cs_schedule_df.groupby(['bus_id']).agg(
#     trips=('trip_id', concat_str),
#     numRecharge=('trip_id',countRecharge),
#     numTrips=('trip_id', countTrips),
#     gapTime=('difference', 'sum')
# )
# # cplex_30Trips3cs_soln.describe()
# print("Time Limit Exact Solution for 30Trips between 1CS vs. 2CS vs. 3CS....")
# print(pd.concat([cplex_30Trips1cs_soln.describe(), cplex_30Trips3cs_soln.describe()], axis=1))
# print(pd.concat([cplex_30Trips1cs_soln_OPT.describe(), cplex_30Trips3cs_soln_OPT.describe()], axis=1))
# print("-"*150)

# ############################################### 40 TRIPS Single CS ###########################################
# filepath = r".\\MIP-"
# filename = f"{filepath}Data\\data_20_40trips.xlsx"
# sheetname = "40Trips"

# NTrips = 40 #trips
# NTerms = 2 #terminals
# NCS = 1 #recharging stations
# CHARGING_TIME = 100 #minutes
# NRechargeCycle = 3 #cycle
# D_MAX = 350 #minutes of operation
# DEPOT = 1 #depot
# # print(filename)
# df = pd.read_excel(filename, sheet_name="40Trips", usecols="A:G", nrows=NTrips+NCS+DEPOT)
# term = pd.read_excel(filename, sheet_name="40Trips", usecols="J:L", skiprows=10, nrows=NTerms)
# gamma = pd.read_excel(filename, sheet_name="gamma", usecols="B:AP", nrows=NTrips+DEPOT)
# delta = pd.read_excel(filename, sheet_name="delta", usecols="B:AR", nrows=NTrips+DEPOT+NCS)
# phi = pd.read_excel(filename, sheet_name="phi", usecols="C:AQ", nrows=(NTrips+DEPOT)*NCS)
# # zeta = pd.read_excel(filename, sheet_name="10Trips", usecols="", skiprows=, nrows=()))
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
# # soln_filename = f"{filepath}cplex_results\\10TripsEVeh1CS_Eq1.txt"
# # soln_filename = f"{filepath}Results\\TimeLimit\\RB\\40TripsEVeh1CS_Eq1_RB.txt"
# # title = 'CPLEX-40Trips1CS'
# # cplex_40Trips1cs_schedule, cplex_40Trips1cs_fig = load_data_hyperparameter(df2, title, soln_filename, R)

# # soln_filename = f"{filepath}Results\\HPC\\RB_40T1CS.asc"
# # title = 'CPLEX-40Trips1CS'
# # cplex_40Trips1cs_schedule_OPT, cplex_40Trips1cs_fig_OPT = load_data_hyperparameter(df2, title, soln_filename, R)

# minBuses = 15
# maxBusesasc = 16
# minBusesASC = maxBusesasc
# maxBusestxt = 20
# minBusesasc = maxBusestxt
# maxBuses = 28
# cplex_pareto_40Trips1CS = []
# for i in range(minBuses, maxBuses):
#     soln_filename = f"{filepath}Results\\TimeLimit\\FINAL\\40TripsEVeh1CS_{i}Buses.asc"
#     if not os.path.isfile(soln_filename):
#         pass
#     else:
#         print(f"soln = {soln_filename}")
#     title = f'CPLEX-40Trips1CS_{i}Buses'
#     cplex_1CS_schedule, cplex_1CS_fig = load_data_hyperparameter(df2, title, soln_filename, R)
#     cplex_pareto_40Trips1CS.append(cplex_1CS_schedule)

#     ax = visualizeOptimization(soln_filename, title)

# cplex40T1CS_nbus = [x for x in range(minBuses, maxBuses)]
# cplex40T1CS_gap = []
# for cplex_40T1CS in cplex_pareto_40Trips1CS:
#     ###### 40Trips ##########
#     cplex_40T1CS_df = cplex_40T1CS.copy(deep=True)
#     cplex_40T1CS_df['next_dep'] = cplex_40T1CS_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
#     cplex_40T1CS_df['difference'] = cplex_40T1CS_df['next_dep'] - cplex_40T1CS_df['arr_time']
#     cplex_40T1CS_df['difference'] = cplex_40T1CS_df['difference'].apply(lambda x: 0 if x < 0 else x)
#     cplex_40T1CS_soln = cplex_40T1CS_df.groupby(['bus_id']).agg(
#         trips=('trip_id', concat_str),
#         numRecharge=('trip_id',countRecharge),
#         numTrips=('trip_id', countTrips),
#         gapTime=('difference', 'sum')
#     )
#     cplex40T1CS_gap.append(cplex_40T1CS_soln['gapTime'].sum())

# test = pd.DataFrame({"NumBuses[#]": cplex40T1CS_nbus, "IdleTime(Gap)[sec]": cplex40T1CS_gap})
# ax = test.sort_values(by=['NumBuses[#]']).plot(
#     x='NumBuses[#]', 
#     y='IdleTime(Gap)[sec]', 
#     xlabel="numBuses", 
#     ylabel="IdleTime(Gap)[sec]", 
#     legend=False, 
#     title=f"40Trips1CS_ParetoFrontier", 
#     style=".-"
# )
# # Set integer x-axis ticks only
# ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
# fig = ax.get_figure()
# ############################################### 40 TRIPS MULTIPLE CSs ###########################################
# filepath = r".\\MIP-"
# filename = f"{filepath}Data\\data_20_40trips.xlsx"
# sheetname = "40Trips"

# NTrips = 40 #trips
# NTerms = 4 #terminals
# CHARGING_TIME = 100 #minutes
# NRechargeCycle = 3 #cycle
# D_MAX = 350 #minutes of operation
# DEPOT = 1 #depot
# NCS = 3 #recharging stations
# df = pd.read_excel(filename, sheet_name="40Trips", usecols="A:G", nrows=NTrips+NCS+DEPOT)
# term = pd.read_excel(filename, sheet_name="40Trips", usecols="J:L", skiprows=10, nrows=NTerms)
# gamma = pd.read_excel(filename, sheet_name="gamma", usecols="B:AP", nrows=NTrips+DEPOT)
# delta = pd.read_excel(filename, sheet_name="delta", usecols="B:AR", nrows=NTrips+DEPOT+NCS)
# phi = pd.read_excel(filename, sheet_name="phi", usecols="C:AQ", nrows=(NTrips+DEPOT)*NCS)
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

# minBuses = 15
# maxBuses = 20
# cplex_pareto_40Trips3CS = []
# numBuses = []
# for i in range(minBuses, maxBuses):
#     soln_filename = f"{filepath}Results\\TimeLimit\\FINAL\\40TripsEVeh3CS_{i}Buses.asc"
#     if not os.path.isfile(soln_filename):
#         pass
#     else:
#         print(f"soln = {soln_filename}")
#         title = f'CPLEX-40Trips3CS_{i}Buses'
#         cplex_3CS_schedule, cplex_3CS_fig = load_data_hyperparameter(df2, title, soln_filename, R)
#         cplex_pareto_40Trips3CS.append(cplex_3CS_schedule)
#         numBuses.append(i)
#         ax = visualizeOptimization(soln_filename, title)

# cplex40T3CS_nbus = numBuses
# cplex40T3CS_gap = []
# for cplex_40T3CS in cplex_pareto_40Trips3CS:
#     ###### 40Trips ##########
#     cplex_40T3CS_df = cplex_40T3CS.copy(deep=True)
#     cplex_40T3CS_df['next_dep'] = cplex_40T3CS_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
#     cplex_40T3CS_df['difference'] = cplex_40T3CS_df['next_dep'] - cplex_40T3CS_df['arr_time']
#     cplex_40T3CS_df['difference'] = cplex_40T3CS_df['difference'].apply(lambda x: 0 if x < 0 else x)
#     cplex_40T3CS_soln = cplex_40T3CS_df.groupby(['bus_id']).agg(
#         trips=('trip_id', concat_str),
#         numRecharge=('trip_id',countRecharge),
#         numTrips=('trip_id', countTrips),
#         gapTime=('difference', 'sum')
#     )
#     cplex40T3CS_gap.append(cplex_40T3CS_soln['gapTime'].sum())

# test = pd.DataFrame({"NumBuses[#]": cplex40T3CS_nbus, "IdleTime(Gap)[sec]": cplex40T3CS_gap})
# ax = test.sort_values(by=['NumBuses[#]']).plot(
#     x='NumBuses[#]', 
#     y='IdleTime(Gap)[sec]', 
#     xlabel="numBuses", 
#     ylabel="IdleTime(Gap)[sec]", 
#     legend=False, 
#     title=f"40Trips3CS_ParetoFrontier", 
#     style=".-"
# )
# # Set integer x-axis ticks only
# ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
# fig = ax.get_figure()

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

# # ########## Time Limit Earliest ###################
# # # soln_filename = f"{filepath}Results\\TimeLimit\\Hetro54TripsEVeh1CS_Earliest.txt"
# # soln_filename = f"{filepath}Results\\TimeLimit\\1May25\\54TripsEVeh1CS_Eq1_58mins.txt"
# # title = 'CPLEX-54Trips1CS'
# # cplex_54Trips1cs_schedule, cplex_54Trips1cs_fig = load_data_hyperparameter(df2, title, soln_filename, R)

# # ############### HPC Z Folder ###############
# # soln_path = f"{filepath}Results\\HPC"
# # soln_filename = f"{soln_path}\\RB_54T1CS.asc"
# # title = 'CPLEX-54Trips1CS_Optimal'
# # cplex_54Trips1cs_schedule_OPT, cplex_54Trips1cs_fig_OPT = load_data_hyperparameter(df2, title, soln_filename, R)

# minBuses = 17
# maxBuses = 27
# cplex_pareto_54Trips1CS = []
# num_buses = []
# for i in range(minBuses, maxBuses):
#     soln_filename =f"{filepath}Results\\TimeLimit\\FINAL\\54T1CS{i}B_PF.asc"
#     if not os.path.isfile(soln_filename):
#         pass
#     else:
#         print(f"soln = {soln_filename}")
#         title = f'CPLEX-54Trips1CS_{i}Buses'
#         cplex_1CS_schedule, cplex_1CS_fig = load_data_hyperparameter(df2, title, soln_filename, R)
#         cplex_pareto_54Trips1CS.append(cplex_1CS_schedule)

#         ax = visualizeOptimization(soln_filename, title)
#         num_buses.append(i)

# cplex54T1CS_nbus = num_buses #[x for x in range(minBuses, maxBuses)]
# cplex54T1CS_gap = []
# for cplex_54T1CS in cplex_pareto_54Trips1CS:
#     ###### 54Trips ##########
#     cplex_54T1CS_df = cplex_54T1CS.copy(deep=True)
#     cplex_54T1CS_df['next_dep'] = cplex_54T1CS_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
#     cplex_54T1CS_df['difference'] = cplex_54T1CS_df['next_dep'] - cplex_54T1CS_df['arr_time']
#     cplex_54T1CS_df['difference'] = cplex_54T1CS_df['difference'].apply(lambda x: 0 if x < 0 else x)
#     cplex_54T1CS_soln = cplex_54T1CS_df.groupby(['bus_id']).agg(
#         trips=('trip_id', concat_str),
#         numRecharge=('trip_id',countRecharge),
#         numTrips=('trip_id', countTrips),
#         gapTime=('difference', 'sum')
#     )
#     cplex54T1CS_gap.append(cplex_54T1CS_soln['gapTime'].sum())

# test = pd.DataFrame({"NumBuses[#]": cplex54T1CS_nbus, "IdleTime(Gap)[sec]": cplex54T1CS_gap})
# ax = test.sort_values(by=['NumBuses[#]']).plot(
#     x='NumBuses[#]', 
#     y='IdleTime(Gap)[sec]', 
#     xlabel="numBuses", 
#     ylabel="IdleTime(Gap)[sec]", 
#     legend=False, 
#     title=f"54Trips1CS_ParetoFrontier", 
#     style=".-"
# )
# # Set integer x-axis ticks only
# ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
# fig = ax.get_figure()

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
# # # soln_filename = f"{filepath}Results\\TimeLimit\\Hetro54TripsEVeh3CS_Earliest.txt"
# # soln_filename = f"{filepath}Results\\TimeLimit\\1May25\\54TripsEVeh3CS_Eq1_53mins.txt"
# # title = 'CPLEX-54Trips3CS'
# # cplex_54Trips3cs_schedule, cplex_54Trips3cs_fig = load_data_hyperparameter(df2, title, soln_filename, R)

# # ############### HPC Z Folder ###############
# # soln_path = f"{filepath}Results\\HPC"
# # soln_filename = f"{soln_path}\\RB_54T3CS.asc"
# # title = 'CPLEX-54Trips3CS_Optimal'
# # cplex_54Trips3cs_schedule_OPT, cplex_54Trips3cs_fig_OPT = load_data_hyperparameter(df2, title, soln_filename, R)

# minBuses = 17
# maxBuses = 23
# cplex_pareto_54Trips3CS = []
# num_buses = []
# for i in range(minBuses, maxBuses):
#     soln_filename = f"{filepath}Results\\TimeLimit\\FINAL\\54TripsEVeh3CS_{i}Buses.asc"
#     if not os.path.isfile(soln_filename):
#         pass
#     else:
#         print(f"soln = {soln_filename}")
#         title = f'CPLEX-54Trips3CS_{i}Buses'
#         cplex_3CS_schedule, cplex_3CS_fig = load_data_hyperparameter(df2, title, soln_filename, R)
#         cplex_pareto_54Trips3CS.append(cplex_3CS_schedule)

#         ax = visualizeOptimization(soln_filename, title)
#         num_buses.append(i)

# cplex54T3CS_nbus = num_buses #[x for x in range(minBuses, maxBuses)]
# cplex54T3CS_gap = []
# for cplex_54T3CS in cplex_pareto_54Trips3CS:
#     ###### 54Trips ##########
#     cplex_54T3CS_df = cplex_54T3CS.copy(deep=True)
#     cplex_54T3CS_df['next_dep'] = cplex_54T3CS_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
#     cplex_54T3CS_df['difference'] = cplex_54T3CS_df['next_dep'] - cplex_54T3CS_df['arr_time']
#     cplex_54T3CS_df['difference'] = cplex_54T3CS_df['difference'].apply(lambda x: 0 if x < 0 else x)
#     cplex_54T3CS_soln = cplex_54T3CS_df.groupby(['bus_id']).agg(
#         trips=('trip_id', concat_str),
#         numRecharge=('trip_id',countRecharge),
#         numTrips=('trip_id', countTrips),
#         gapTime=('difference', 'sum')
#     )
#     cplex54T3CS_gap.append(cplex_54T3CS_soln['gapTime'].sum())

# test = pd.DataFrame({"NumBuses[#]": cplex54T3CS_nbus, "IdleTime(Gap)[sec]": cplex54T3CS_gap})
# ax = test.sort_values(by=['NumBuses[#]']).plot(
#     x='NumBuses[#]', 
#     y='IdleTime(Gap)[sec]', 
#     xlabel="numBuses", 
#     ylabel="IdleTime(Gap)[sec]", 
#     legend=False, 
#     title=f"54Trips3CS_ParetoFrontier", 
#     style=".-"
# )
# # Set integer x-axis ticks only
# ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
# fig = ax.get_figure()

# ###### 54Trips ##########
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

# ###### 54Trips ############
# cplex_54Trips1cs_schedule_df = cplex_54Trips1cs_schedule_OPT.copy(deep=True)
# cplex_54Trips1cs_schedule_df['next_dep'] = cplex_54Trips1cs_schedule_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
# cplex_54Trips1cs_schedule_df['difference'] = cplex_54Trips1cs_schedule_df['next_dep'] - cplex_54Trips1cs_schedule_df['arr_time']
# cplex_54Trips1cs_schedule_df['difference'] = cplex_54Trips1cs_schedule_df['difference'].apply(lambda x: 0 if x < 0 else x)
# cplex_54Trips1cs_soln_OPT = cplex_54Trips1cs_schedule_df.groupby(['bus_id']).agg(
#     trips=('trip_id', concat_str),
#     numRecharge=('trip_id',countRecharge),
#     numTrips=('trip_id', countTrips),
#     gapTime=('difference', 'sum')
# )

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

# cplex_54Trips3cs_schedule_df = cplex_54Trips3cs_schedule_OPT.copy(deep=True)
# cplex_54Trips3cs_schedule_df['next_dep'] = cplex_54Trips3cs_schedule_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
# cplex_54Trips3cs_schedule_df['difference'] = cplex_54Trips3cs_schedule_df['next_dep'] - cplex_54Trips3cs_schedule_df['arr_time']
# cplex_54Trips3cs_schedule_df['difference'] = cplex_54Trips3cs_schedule_df['difference'].apply(lambda x: 0 if x < 0 else x)
# cplex_54Trips3cs_soln_OPT = cplex_54Trips3cs_schedule_df.groupby(['bus_id']).agg(
#     trips=('trip_id', concat_str),
#     numRecharge=('trip_id',countRecharge),
#     numTrips=('trip_id', countTrips),
#     gapTime=('difference', 'sum')
# )
# # cplex_30Trips3cs_soln.describe()
# print("Time Limit Exact Solution for 54Trips between 1CS vs. 2CS vs. 3CS....")
# print(pd.concat([cplex_54Trips1cs_soln.describe(), cplex_54Trips3cs_soln.describe()], axis=1))
# print(pd.concat([cplex_54Trips1cs_soln_OPT.describe(), cplex_54Trips3cs_soln_OPT.describe()], axis=1))
# print("-"*150)

# ############################################### 100 TRIPS Single CS ###########################################
# filename = f"{filepath}Data\\trip100cs1.xlsx"
# sheetname = "main"

# NTrips = 100 #trips
# NTerms = 4 #terminals
# NCS = 1 #recharging stations
# CHARGING_TIME = 100 #minutes
# NRechargeCycle = 3 #cycle
# D_MAX = 350 #minutes of operation
# DEPOT = 1 #depot
# df = pd.read_excel(filename, sheet_name="main", usecols="A:H", nrows=NTrips+NCS+DEPOT)
# term = pd.read_excel(filename, sheet_name="main", usecols="L:M", skiprows=19, nrows=NTerms)
# gamma = pd.read_excel(filename, sheet_name="gamma", usecols="B:CX", nrows=NTrips+DEPOT)
# delta = pd.read_excel(filename, sheet_name="delta", usecols="B:CY", nrows=NTrips+DEPOT+NCS)
# phi = pd.read_excel(filename, sheet_name="phi", usecols="B:CX", nrows=(NTrips+DEPOT)*NCS)
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

# # ########## Time Limit Earliest ###################
# # # soln_filename = f"{filepath}Results\\TimeLimit\\Hetro54TripsEVeh1CS_Earliest.txt"
# # soln_filename = f"{filepath}Results\\TimeLimit\\1May25\\54TripsEVeh1CS_Eq1_58mins.txt"
# # title = 'CPLEX-54Trips1CS'
# # cplex_54Trips1cs_schedule, cplex_54Trips1cs_fig = load_data_hyperparameter(df2, title, soln_filename, R)

# # ############### HPC Z Folder ###############
# # soln_path = f"{filepath}Results\\HPC"
# # soln_filename = f"{soln_path}\\RB_54T1CS.asc"
# # title = 'CPLEX-54Trips1CS_Optimal'
# # cplex_54Trips1cs_schedule_OPT, cplex_54Trips1cs_fig_OPT = load_data_hyperparameter(df2, title, soln_filename, R)

# minBuses = 31
# maxBuses = 32
# cplex_pareto_100Trips1CS = []
# num_buses = []
# for i in range(minBuses, maxBuses):
#     soln_filename = f"{filepath}Results\\TimeLimit\\ParetoFront\\100TripsEVeh1CS_{i}Buses.asc"
#     if not os.path.isfile(soln_filename):
#         pass
#     else:
#         print(f"soln = {soln_filename}")
#         title = f'CPLEX-100Trips1CS_{i}Buses'
#         print(df2)
#         cplex_1CS_schedule, cplex_1CS_fig = load_data_hyperparameter(df2, title, soln_filename, R)
#         cplex_pareto_100Trips1CS.append(cplex_1CS_schedule)

#         ax = visualizeOptimization(soln_filename, title)
#         num_buses.append(i)

# cplex100T1CS_nbus = num_buses #[x for x in range(minBuses, maxBuses)]
# cplex100T1CS_gap = []
# for cplex_100T1CS in cplex_pareto_100Trips1CS:
#     ###### 54Trips ##########
#     cplex_100T1CS_df = cplex_100T1CS.copy(deep=True)
#     cplex_100T1CS_df['next_dep'] = cplex_100T1CS_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
#     cplex_100T1CS_df['difference'] = cplex_100T1CS_df['next_dep'] - cplex_100T1CS_df['arr_time']
#     cplex_100T1CS_df['difference'] = cplex_100T1CS_df['difference'].apply(lambda x: 0 if x < 0 else x)
#     cplex_100T1CS_soln = cplex_100T1CS_df.groupby(['bus_id']).agg(
#         trips=('trip_id', concat_str),
#         numRecharge=('trip_id',countRecharge),
#         numTrips=('trip_id', countTrips),
#         gapTime=('difference', 'sum')
#     )
#     cplex100T1CS_gap.append(cplex_100T1CS_soln['gapTime'].sum())

# test = pd.DataFrame({"NumBuses[#]": cplex100T1CS_nbus, "IdleTime(Gap)[sec]": cplex100T1CS_gap})
# ax = test.sort_values(by=['NumBuses[#]']).plot(
#     x='NumBuses[#]', 
#     y='IdleTime(Gap)[sec]', 
#     xlabel="numBuses", 
#     ylabel="IdleTime(Gap)[sec]", 
#     legend=False, 
#     title=f"100Trips1CS_ParetoFrontier", 
#     style=".-"
# )
# # Set integer x-axis ticks only
# ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
# fig = ax.get_figure()

# ############################################### 100 TRIPS MULTIPLE CSs ###########################################
# filename = f"{filepath}Data\\trip100cs3.xlsx"
# sheetname = "main"

# NTrips = 100 #trips
# NTerms = 4 #terminals
# NCS = 3 #recharging stations
# CHARGING_TIME = 100 #minutes
# NRechargeCycle = 3 #cycle
# D_MAX = 350 #minutes of operation
# DEPOT = 1 #depot
# df = pd.read_excel(filename, sheet_name="main", usecols="A:H", nrows=NTrips+NCS+DEPOT)
# term = pd.read_excel(filename, sheet_name="main", usecols="L:M", skiprows=19, nrows=NTerms)
# gamma = pd.read_excel(filename, sheet_name="gamma", usecols="B:CX", nrows=NTrips+DEPOT)
# delta = pd.read_excel(filename, sheet_name="delta", usecols="B:DA", nrows=NTrips+DEPOT+NCS)
# phi = pd.read_excel(filename, sheet_name="phi", usecols="B:CX", nrows=(NTrips+DEPOT)*NCS)
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

# minBuses = 31
# maxBuses = 35
# cplex_pareto_100Trips3CS = []
# num_buses = []
# for i in range(minBuses, maxBuses):
#     soln_filename = f"{filepath}Results\\TimeLimit\\FINAL\\100T3CS{i}B_PF.asc"
#     if not os.path.isfile(soln_filename):
#         pass
#     else:
#         print(f"soln = {soln_filename}")
#         title = f'CPLEX-100Trips3CS_{i}Buses'
#         cplex_3CS_schedule, cplex_3CS_fig = load_data_hyperparameter(df2, title, soln_filename, R)
#         cplex_pareto_100Trips3CS.append(cplex_3CS_schedule)
#         num_buses.append(i)

#         ax = visualizeOptimization(soln_filename, title)

# cplex100T3CS_nbus = num_buses #[x for x in range(minBuses, maxBuses)]
# cplex100T3CS_gap = []
# for cplex_100T3CS in cplex_pareto_100Trips3CS:
#     ###### 54Trips ##########
#     cplex_100T3CS_df = cplex_100T3CS.copy(deep=True)
#     cplex_100T3CS_df['next_dep'] = cplex_100T3CS_df.groupby('bus_id')['dep_time'].shift(-1).fillna(0)
#     cplex_100T3CS_df['difference'] = cplex_100T3CS_df['next_dep'] - cplex_100T3CS_df['arr_time']
#     cplex_100T3CS_df['difference'] = cplex_100T3CS_df['difference'].apply(lambda x: 0 if x < 0 else x)
#     cplex_100T3CS_soln = cplex_100T3CS_df.groupby(['bus_id']).agg(
#         trips=('trip_id', concat_str),
#         numRecharge=('trip_id',countRecharge),
#         numTrips=('trip_id', countTrips),
#         gapTime=('difference', 'sum')
#     )
#     cplex100T3CS_gap.append(cplex_100T3CS_soln['gapTime'].sum())

# test = pd.DataFrame({"NumBuses[#]": cplex100T3CS_nbus, "IdleTime(Gap)[sec]": cplex100T3CS_gap})
# ax = test.sort_values(by=['NumBuses[#]']).plot(
#     x='NumBuses[#]', 
#     y='IdleTime(Gap)[sec]', 
#     xlabel="numBuses", 
#     ylabel="IdleTime(Gap)[sec]", 
#     legend=False, 
#     title=f"100Trips3CS_ParetoFrontier", 
#     style=".-"
# )
# # Set integer x-axis ticks only
# ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
# fig = ax.get_figure()

# # ############################################### 100 TRIPS Single CS ###########################################
# # filename = f"{filepath}Data\\trip100cs1.xlsx"
# # sheetname = "main"

# # NTrips = 100 #trips
# # NTerms = 4 #terminals
# # NCS = 1 #recharging stations
# # CHARGING_TIME = 50 #minutes
# # NRechargeCycle = 3 #cycle
# # D_MAX = 350 #minutes of operation
# # DEPOT = 1 #depot
# # df = pd.read_excel(filename, sheet_name="main", usecols="A:H", nrows=NTrips+NCS+DEPOT)
# # term = pd.read_excel(filename, sheet_name="main", usecols="L:M", skiprows=25, nrows=NTerms)
# # gamma = pd.read_excel(filename, sheet_name="gamma", usecols="B:CX", nrows=NTrips+DEPOT)
# # delta = pd.read_excel(filename, sheet_name="delta", usecols="B:CY", nrows=NTrips+DEPOT+NCS)
# # phi = pd.read_excel(filename, sheet_name="phi", usecols="B:CX", nrows=(NTrips+DEPOT)*NCS)
# # df.rename(columns={'name':'trip_id', 'dep_term': 'dep_terminal', 'arr_term': 'arr_terminal'}, inplace=True)
# # df2 = df.set_index('trip_id')
# # gamma = gamma.to_numpy()
# # delta = delta.to_numpy()
# # phi = phi.to_numpy()

# # trips_df = df.copy()
# # s = len(trips_df)
# # k = (s // 2) - 2

# # K = [i for i in range(1, k+1)]
# # # print(f"K = {K}")
# # Sprime = [i for i in range(s)]
# # # print(f"Sprime = {Sprime}")
# # S = Sprime[:-NCS]
# # # print(f"S = {S}")
# # S1 = S[1:]
# # # print(f"S1 = {S1}")
# # R = Sprime[-NCS:]
# # # print(f"R = {R}")
# # # print(f"Expected number of buses = {k} -> {K}\nNumber of available schedules {s} -> {Sprime}\n")
# # durations = trips_df.duration.to_list()
# # C = [i for i in range(1, NRechargeCycle+1)]
# # # print(C)
# # df2['ID'] = range(1, len(df2)+1)
# # cs_ids = df2.loc[df2.ID.isin(R)].index.to_list()
# # # print(cs_ids)

# # ############### HPC Z Folder ###############
# # soln_path = f"{filepath}Results\\HPC"
# # soln_filename = f"{soln_path}\\RB_100T1CS.asc"
# # title = 'CPLEX-100Trips1CS_Optimal'
# # cplex_100Trips1cs_schedule_OPT, cplex_100Trips1cs_fig_OPT = load_data_hyperparameter(df2, title, soln_filename, R)

# # ############################################### 100 TRIPS MULTIPLE CSs ###########################################
# # filename = f"{filepath}Data\\trip100cs3.xlsx"
# # sheetname = "main"

# # NTrips = 100 #trips
# # NTerms = 4 #terminals
# # NCS = 3 #recharging stations
# # CHARGING_TIME = 50 #minutes
# # NRechargeCycle = 3 #cycle
# # D_MAX = 350 #minutes of operation
# # DEPOT = 1 #depot
# # df = pd.read_excel(filename, sheet_name="main", usecols="A:H", nrows=NTrips+NCS+DEPOT)
# # term = pd.read_excel(filename, sheet_name="main", usecols="L:M", skiprows=25, nrows=NTerms)
# # gamma = pd.read_excel(filename, sheet_name="gamma", usecols="B:CX", nrows=NTrips+DEPOT)
# # delta = pd.read_excel(filename, sheet_name="delta", usecols="B:CY", nrows=NTrips+DEPOT+NCS)
# # phi = pd.read_excel(filename, sheet_name="phi", usecols="B:CX", nrows=(NTrips+DEPOT)*NCS)
# # df.rename(columns={'name':'trip_id', 'dep_term': 'dep_terminal', 'arr_term': 'arr_terminal'}, inplace=True)
# # df2 = df.set_index('trip_id')
# # gamma = gamma.to_numpy()
# # delta = delta.to_numpy()
# # phi = phi.to_numpy()

# # trips_df = df.copy()
# # s = len(trips_df)
# # k = (s // 2) - 2

# # K = [i for i in range(1, k+1)]
# # # print(f"K = {K}")
# # Sprime = [i for i in range(s)]
# # # print(f"Sprime = {Sprime}")
# # S = Sprime[:-NCS]
# # # print(f"S = {S}")
# # S1 = S[1:]
# # # print(f"S1 = {S1}")
# # R = Sprime[-NCS:]
# # # print(f"R = {R}")
# # # print(f"Expected number of buses = {k} -> {K}\nNumber of available schedules {s} -> {Sprime}\n")
# # durations = trips_df.duration.to_list()
# # C = [i for i in range(1, NRechargeCycle+1)]
# # # print(C)
# # df2['ID'] = range(1, len(df2)+1)
# # cs_ids = df2.loc[df2.ID.isin(R)].index.to_list()
# # # print(cs_ids)
# # ############### HPC Z Folder ###############
# # soln_path = f"{filepath}Results\\HPC"
# # soln_filename = f"{soln_path}\\RB_100T3CS.asc"
# # title = 'CPLEX-100Trips3CS_Optimal'
# # cplex_100Trips3cs_schedule_OPT, cplex_100Trips3cs_fig_OPT = load_data_hyperparameter(df2, title, soln_filename, R)




# soln_filename = f"{filepath}Results\\TimeLimit\\1May25\\10TripsEVeh3CS_Eq1.txt"
# title = 'CPLEX-10Trips1CS'
# ax = visualizeOptimization(soln_filename, title)

# soln_filename = f"{filepath}Results\\TimeLimit\\1May25\\10TripsEVeh3CS_Eq1.txt"
# title = 'CPLEX-10Trips3CS'
# ax = visualizeOptimization(soln_filename, title)

# soln_filename = f"{filepath}Results\\TimeLimit\\1May25\\20TripsEVeh1CS_Eq1.txt"
# title = 'CPLEX-20Trips1CS'
# ax = visualizeOptimization(soln_filename, title)

# soln_filename = f"{filepath}Results\\TimeLimit\\1May25\\20TripsEVeh3CS_Eq1.txt"
# title = 'CPLEX-20Trips3CS'
# ax = visualizeOptimization(soln_filename, title)

# soln_filename = f"{filepath}Results\\TimeLimit\\1May25\\20TripsEVeh3CS_Eq1.txt"
# title = 'CPLEX-20Trips3CS'
# ax = visualizeOptimization(soln_filename, title)

# soln_filename = f"{filepath}Results\\TimeLimit\\30TripsEVeh1CS_Eq1(TimeLimit)1062.txt"
# title = 'CPLEX-30Trips1CS'
# ax = visualizeOptimization(soln_filename, title)


# soln_filename = f"{filepath}Results\\TimeLimit\\30TripsEVeh3CS_Eq1(TimeLimit)2871.txt"
# title = 'CPLEX-30Trips3CS_IDLE'
# ax = visualizeOptimization(soln_filename, title)

# soln_filename = f"{filepath}Results\\TimeLimit\\1May25\\archives\\40TripsEVeh1CS_Eq1_SINGLE.txt"
# title = 'CPLEX-40Trips1CS'
# ax = visualizeOptimization(soln_filename, title)


# soln_filename = f"{filepath}Results\\TimeLimit\\1May25\\archives\\40TripsEVeh3CS_Eq1_SINGLE.txt"
# title = 'CPLEX-40Trips3CS'
# ax = visualizeOptimization(soln_filename, title)


# soln_filename = f"{filepath}Results\\TimeLimit\\1May25\\54TripsEVeh1CS_Eq1_58mins.txt"
# title = 'CPLEX-54Trips1CS'
# ax = visualizeOptimization(soln_filename, title)
# soln_filename = f"{filepath}Results\\TimeLimit\\RB\\54TripsEVeh3CS_Eq1_RB.txt"
# title = 'CPLEX-54Trips3CS'
# ax = visualizeOptimization(soln_filename, title)
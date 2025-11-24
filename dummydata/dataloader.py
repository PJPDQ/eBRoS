import pandas as pd
import numpy as np

metadata = {
    "# of trips": 0, 
    "# of depots": 0, 
    "# of EB types": 0, 
    "# of terminals": 0, 
    "minimum layover time (mins)": 0,
    "EB battery size (kWh)": 0,
    "charging rate (kWh/min)": 0,
    "electricity price ($/kWh)": 0,
    "SoC_max": 0,
    "SoC_min": 0,
    "# of intervals": 1,
    "monetary_cost per kWh_1": 1,
    "monetary_cost per kWh_2": 1,
    "monetary_cost per kWh_3": 1,
    "monetary_cost per kWh_4": 1,
    "depot name": 2,
    "EB types": 3,
    "terminal name": 4,
    "SINGLE Fixed cost": 5,
    "DOUBLE Fixed cost": 5,
    "SINGLE energy consumption rate (kWh/km)": 6,
    "DOUBLE energy consumption rate (kWh/km)": 6,
    "BB-BB terminal deadhead time matrix": 7,
    "BB-MC terminal deadhead time matrix": 7,
    "MC-BB terminal deadhead time matrix": 8,
    "MC-MC terminal deadhead time matrix": 8,
    "BB-BB terminal deadhead distance matrix": 9,
    "BB-MC terminal deadhead distance matrix": 9,
    "MC-BB terminal deadhead distance matrix": 10,
    "MC-MC terminal deadhead distance matrix": 10,
    "depot-terminal BB deadhead time matrix": 11,
    "depot-terminal MC deadhead time matrix": 11,
    "depot-terminal BB deadhead distance matrix": 12,
    "depot-terminal MC deadhead distance matrix": 12,
    "available # of SINGLE EB types per depot": 13,
    "available # of DOUBLE EB types per depot": 13
}
env = {}

filename = r"./D1_R2_T30.txt"
with open(filename, "r") as f:
    content = f.readlines()

dataset = []
for idx, line in enumerate(content):
    columns = list(map(lambda x: x[0], filter(lambda item: item[1] == idx, metadata.items())))
    data = line.split("	")
    # print(data)
    # print(columns)
    if len(data) == len(columns):
        for col, d in zip(columns, data):
            env[col] = d.split("\n")[0] if "\n" in d else d
    else:
        if len(data) > 9:
            data = [d.split("\n")[0] if "\n" in d else d for d in data]
            dataset.append(data)
        else:
            env[columns[0]] = data

    
columns = [
    "tripID",
    "routeName",
    "length",
    "dep_term",
    "arr_term",
    "dep_time",
    "arr_time",
    "dep_time_mins",
    "arr_time_mins",
    "PEAK?"
]

print(f"env = {env}")
df = pd.DataFrame(data=dataset, columns=columns)
df['tripID'] = df['tripID'].astype(int)
df['routeName'] = df['routeName'].astype(int)
df['length'] = df['length'].astype(float)
df['dep_time_mins'] = df['dep_time_mins'].astype(int)
df['arr_time_mins'] = df['arr_time_mins'].astype(int)
df['duration(mins)'] = df.apply(lambda x: x['arr_time_mins'] - x['dep_time_mins'], axis=1)
print(df)
terminals = {}
for i in range(int(env['# of terminals'])):
    if "\n" in env['terminal name'][i]:
        env['terminal name'][i] = env['terminal name'][i].split("\n")[0]
    terminals[env['terminal name'][i]] = {'max_interval' : env['minimum layover time (mins)']}
### Assuming single type of eBus energy consumption rate = 2 kWh/km
### EB Battery Size = 200 kWh
####### Therefore, the total minimum SoC per km => (200 - 0.1(200)) / 2 => 90 km
### Charging Rate = 5 kWh/min
### SoC min = 0.1 (20kWh)
## D_MAX = 150
cs_deadheads = {
    "Bukit_Batok": {"Marina_Center": 28},
    "Marina_Center": {"Bukit_Batok": 22}
}
CHARGING_TIME = 36 # (200 - 0.1(200)) / 5 => 36mins

def feasible_pairs(data, terminals=terminals):
    bus_data = data.copy(deep=True)
    bus_data['index'] = bus_data['tripID']
    bus_data.set_index("index", inplace=True)
    # bus_data = bus_data.reset_index()
    # bus_data.rename(columns={'index': 'trip_id'}, inplace=True)
    # bus_data.index = np.arange(0, len(data))
    print(bus_data)
    ### duration (delta)
    duration_trips = (
        #lambda pair: bus_data.loc[pair[1], "duration"] + bus_data.loc[pair[0], 'duration'] if not (bus_data.loc[pair[1], 'type'] == 'depot' or bus_data.loc[pair[0], 'type'] == 'depot') else bus_data.loc[pair[1], "duration"]  if bus_data.loc[pair[0], 'type'] == 'depot' else 0 #+ bus_data.loc[pair[0], "duration"] if not (bus_data.loc[pair[0], 'type'] == 'depot' or bus_data.loc[pair[1], "type"] == 'depot') else 0
        lambda pair: 0 if bus_data.loc[pair[1], 'type'] == 'depot' else bus_data.loc[pair[1], "duration"]
        #if not (bus_data.loc[pair[1], 'type'] == 'depot') else bus_data.loc[pair[0], "duration"] 
    )
    connected_trips = (
        lambda pair: bus_data.loc[pair[1], "dep_time_mins"] - bus_data.loc[pair[0], "arr_time_mins"]
        # lambda pair: bus_data.loc[pair[1], 'trip_id'] != bus_data.loc[pair[0], 'trip_id']
    )
    ## feasible (gamma)
    pairs = filter(
        # lambda pair: (bus_data.loc[pair[0], 'type'] == 'depot' or bus_data.loc[pair[1], "type"] == 'depot') or (connected_trips(pair) >= terminals[bus_data.loc[pair[1], "arr_term"]]["max_interval"] and bus_data.loc[pair[0], "arr_term"] == bus_data.loc[pair[1], "dep_term"]),
        # [(i,j) for i in bus_data.index for j in bus_data.index if i != j],
        lambda pair: (
            ((pair[0] == 0 or pair[1] == 0) and pair[0] != pair[1]) and
            (connected_trips(pair) >= terminals[bus_data.loc[pair[1], "arr_time"]['max_interval']] and bus_data.loc[pair[0], "arr_term"] == bus_data.loc[pair[1], "dep_term"]) or 
            (connected_trips(pair) >= (terminals[bus_data.loc[pair[1], "arr_time"]['max_interval']] + cs_deadheads[bus_data.loc[pair[0], "arr_term"]][bus_data.loc[pair[1], "dep_term"]]) and bus_data.loc[pair[0], "arr_term"] != bus_data.loc[pair[1], "dep_term"])
        ), 
        [(i,j) for i in range(len(bus_data)) for j in range(len(bus_data))]
    )
    return {pair: {'duration': duration_trips(pair)} for pair in pairs}
# ## Creating Gamma and Delta
# print(feasible_pairs(trips_df, terminals=terminals))
# arcs = feasible_pairs(trips_df, terminals=terminals)

# def feasible_recharge(data, gamma, recharge=charging_stations):
#     bus_data = data.copy(deep=True)
#     bus_data = bus_data.reset_index()
#     bus_data.rename(columns={'index': 'trip_id'}, inplace=True)
#     bus_data.index = np.arange(0, len(data))
#     print(bus_data)
#     connected_trips = (
#         lambda pair: bus_data.loc[pair[1], "dep_time"] - bus_data.loc[pair[0], "arr_time"]
#         # lambda pair: bus_data.loc[pair[1], 'trip_id'] != bus_data.loc[pair[0], 'trip_id']
#     )
#     ### duration (delta)
#     duration_trips = (
#         lambda pair: connected_trips(pair) - (recharge[pair[2]]['duration'])
#     )
#     ## feasible (phi)
#     pairs = filter(
#         lambda pair: (bus_data.loc[pair[0], 'type'] != 'depot' and bus_data.loc[pair[1], "type"] != 'depot') and ((connected_trips(pair) >= (recharge[pair[2]]['duration']))),
#         [(i,j,k) for k in recharge.keys() for i in bus_data.index for j in bus_data.index if i != j],
#     )
#     return {pair: {'duration': duration_trips(pair)} for pair in pairs}
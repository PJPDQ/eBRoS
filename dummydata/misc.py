from ast import literal_eval
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
CHARGING_TIME = 50
def flatten_link_names(x):
    """
    flatten the list of names lists to generate a single list of items
    """
    flat_list = []
    q_list = []
    a = x.copy()
    for sublist in a:
        if isinstance(sublist, str):
            sublist = literal_eval(sublist)
        if not sublist in q_list:
            q_list.append(sublist)
            for item in sublist[:-1]:
                if item not in flat_list:
                    flat_list.append(item)
    return flat_list

def vectorSchRepresentation(schedules):
    """
    Transform (i,j)/ (i,j,k) -> [i,j]/ [i,j,k]
    """
    all_schs = []
#     print(schedules)
    for bus in schedules:
        schs = []
        for pair in bus:
            if len(pair) < 3:
                schs.append(pair[1])
            else:
                schs += [pair[2]] + [pair[1]]
                
        all_schs.append(schs)
    return all_schs

def convert_unique_path(new_path):
    return flatten_link_names(vectorSchRepresentation([new_path]))

def filter_arcs(arcs, nodes, cs, considered_node=None):
    nodes = nodes + [considered_node]
    arcs = dict(filter(lambda x: x[0][0] in nodes and x[0][1] in nodes, arcs.items()))
    res = dict(filter(lambda x: x[0][1] in nodes, arcs.items()))
    for idx in range(len(nodes)-1):
        i = nodes[idx]
        j = nodes[idx+1]
        if j == 0:
            tmp = dict(filter(lambda x: x[0][0] == i, arcs.items()))
            res = {**res, **tmp}
        elif i in cs or j in cs:
            tmp = dict(filter(lambda x: x[0][0] == i or x[0][1] == j, arcs.items()))
            res = {**res, **tmp}        
        else:
            tmp = dict(filter(lambda x: x[0][0] == i and x[0][1] == j, arcs.items()))
            res = {**res, **tmp}
    return res

def feasible_pairs(data, terminals):
    bus_data = data.copy(deep=True)
    bus_data = bus_data.reset_index()
    bus_data.rename(columns={'index': 'trip_id'}, inplace=True)
    bus_data.index = np.arange(0, len(data))
    print(bus_data)
    ### duration (delta)
    duration_trips = (
        lambda pair: 0 if bus_data.loc[pair[1], 'type'] == 'depot' else bus_data.loc[pair[1], "duration"]
    )
    connected_trips = (
        lambda pair: bus_data.loc[pair[1], "dep_time"] - bus_data.loc[pair[0], "arr_time"]
    )
    ## feasible (gamma)
    pairs = filter(
        lambda pair: (bus_data.loc[pair[0], 'type'] == 'depot' or bus_data.loc[pair[1], "type"] == 'depot') or (connected_trips(pair) > terminals[bus_data.loc[pair[1], "arr_term"]]["max_interval"]),
        [(i,j) for i in bus_data.index for j in bus_data.index if i != j],
    )
    return {pair: {'duration': duration_trips(pair)} for pair in pairs}

def feasible_recharge(data, deadheads, recharge, terminals):
    bus_data = data.copy(deep=True)
    bus_data = bus_data.reset_index()
    bus_data.rename(columns={'index': 'trip_id'}, inplace=True)
    bus_data.index = np.arange(0, len(data))
    connected_trips = (
        lambda pair: bus_data.loc[pair[1], "dep_time"] - bus_data.loc[pair[0], "arr_time"]
    )
    # Recharging Feasibility becomes a spatial dependent variable with the sum of the deadheads and the duration to recharge.
    recharging_costs = (
        lambda pair: (recharge[pair[2]]['dep_term'] == "-" and connected_trips(pair)>=(CHARGING_TIME +deadheads[bus_data.loc[pair[0], "arr_term"]]["CS1"])) or (recharge[pair[2]]['dep_term'] == bus_data.loc[pair[1], "dep_term"])
    )
    ### duration (delta)
    duration_trips = (
        lambda pair: deadheads[bus_data.loc[pair[0], "arr_term"]][recharge[pair[2]]['name']]
    )
    ## feasible (phi)
    pairs = filter(
        lambda pair: (bus_data.loc[pair[0], 'type'] != 'depot' and bus_data.loc[pair[1], "type"] != 'depot') and (connected_trips(pair) >= terminals[bus_data.loc[pair[1], "arr_term"]]["max_interval"]) and (recharging_costs(pair)),
        [(i,j,k) for k in recharge.keys() for i in bus_data.index for j in bus_data.index if i != j],
    )

    return {pair: {'duration': duration_trips(pair)} for pair in pairs}
# def feasible_pairs(data, terminals):
#     bus_data = data.copy(deep=True)
#     bus_data = bus_data.reset_index()
#     bus_data.rename(columns={'index': 'trip_id'}, inplace=True)
#     bus_data.index = np.arange(0, len(data))
#     print(bus_data)
#     ### duration (delta)
#     duration_trips = (
#         #lambda pair: bus_data.loc[pair[1], "duration"] + bus_data.loc[pair[0], 'duration'] if not (bus_data.loc[pair[1], 'type'] == 'depot' or bus_data.loc[pair[0], 'type'] == 'depot') else bus_data.loc[pair[1], "duration"]  if bus_data.loc[pair[0], 'type'] == 'depot' else 0 #+ bus_data.loc[pair[0], "duration"] if not (bus_data.loc[pair[0], 'type'] == 'depot' or bus_data.loc[pair[1], "type"] == 'depot') else 0
#         lambda pair: 0 if bus_data.loc[pair[1], 'type'] == 'depot' else bus_data.loc[pair[1], "duration"]
#         #if not (bus_data.loc[pair[1], 'type'] == 'depot') else bus_data.loc[pair[0], "duration"] 
#     )
#     connected_trips = (
#         lambda pair: bus_data.loc[pair[1], "dep_time"] - bus_data.loc[pair[0], "arr_time"]
#         # lambda pair: bus_data.loc[pair[1], 'trip_id'] != bus_data.loc[pair[0], 'trip_id']
#     )
#     ## feasible (gamma)
#     pairs = filter(
#         lambda pair: (bus_data.loc[pair[0], 'type'] == 'depot' or bus_data.loc[pair[1], "type"] == 'depot') or (connected_trips(pair) >= terminals[bus_data.loc[pair[1], "arr_term"]]["max_interval"] and bus_data.loc[pair[0], "arr_term"] == bus_data.loc[pair[1], "dep_term"]),
#         [(i,j) for i in bus_data.index for j in bus_data.index if i != j],
#     )
#     return {pair: {'duration': duration_trips(pair)} for pair in pairs}

# def feasible_recharge(data, deadheads, recharge):
#     bus_data = data.copy(deep=True)
#     bus_data = bus_data.reset_index()
#     bus_data.rename(columns={'index': 'trip_id'}, inplace=True)
#     bus_data.index = np.arange(0, len(data))
#     print(bus_data)
#     connected_trips = (
#         lambda pair: bus_data.loc[pair[1], "dep_time"] - bus_data.loc[pair[0], "arr_time"]
#         # lambda pair: bus_data.loc[pair[1], 'trip_id'] != bus_data.loc[pair[0], 'trip_id']
#     )
#     # Recharging Feasibility becomes a spatial dependent variable with the sum of the deadheads and the duration to recharge.
#     recharging_costs = (
#         lambda pair: recharge[pair[2]]['duration'] + deadheads[bus_data.loc[pair[0], "arr_term"]][recharge[pair[2]]['name']]
#     )
#     ### duration (delta)
#     duration_trips = (
#         lambda pair: deadheads[bus_data.loc[pair[0], "arr_term"]][recharge[pair[2]]['name']]
#     )
#     ## feasible (phi)
#     pairs = filter(
#         lambda pair: (bus_data.loc[pair[0], 'type'] != 'depot' and bus_data.loc[pair[1], "type"] != 'depot') and ((connected_trips(pair) >= recharging_costs(pair))),
#         [(i,j,k) for k in recharge.keys() for i in bus_data.index for j in bus_data.index if i != j],
#     )
#     return {pair: {'duration': duration_trips(pair)} for pair in pairs}

def get_total_gap(schedules, all_nodes, recharge_arc):
    total_waiting = 0
    for schedule in schedules:
        for idx in range(len(schedule)-1):
            i = schedule[idx]
            j = schedule[idx+1]
#             print(all_nodes.iloc[i, 1])
            if all_nodes.iloc[i, 1] == 'cs': ## time difference between the end of charging to the next departure
                h = schedule[idx-1]
                total_waiting += recharge_arc[(h,j,i)]['duration']
            elif all_nodes.iloc[j, 1] == 'cs': ## going to charging only cost deadhead time to the charging station
                pass
            elif all_nodes.iloc[j, 1] == 'depot':
                pass
            else:
                gap_time = all_nodes.iloc[j, 3] - all_nodes.iloc[i, 4] # next_departure - prev_arrival
                total_waiting += gap_time
    return total_waiting

def visualizeSolution(solution_spaces, title, all_sch, recharge_arc):
    numBuses = []
    gaps = []
    for space in solution_spaces:
        numBuses.append(len(space))
        gap = get_total_gap(space, all_sch, recharge_arc)
        gaps.append(gap)

    soln_df = pd.DataFrame({'solution': solution_spaces, 'numBuses': numBuses, 'Gaps': gaps})
    ax = soln_df[['numBuses', 'Gaps']].plot(x='numBuses', y='Gaps', style='*', title=title)
    ax.set_xlabel("numBuses")
    ax.set_ylabel("Total Gap")
    return soln_df

def visualize(results, title):
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

    fig, ax = plt.subplots(1,1, figsize=(12, int(len(buses)/2) + 3))
    for jdx, j in enumerate(trips, 1):
        for mdx, m in enumerate(buses, 1):
            if (j,m) in schedule.index:
                if type(schedule.loc[(j,m), 'dep_time']) == np.int64:
                    xs = schedule.loc[(j,m), 'dep_time']
                    xf = schedule.loc[(j,m), 'arr_time']
                    xs_str = schedule.loc[(j,m), 'dep_str']
                    xf_str = schedule.loc[(j,m), 'arr_str']
                else:
#                 print(type(schedule.loc[(j,m), 'dep_time']))
                    xs = schedule.loc[(j,m), 'dep_time'].iloc[0]
                    xf = schedule.loc[(j,m), 'arr_time'].iloc[0]
                    xs_str = schedule.loc[(j,m), 'dep_str'].iloc[0]
                    xf_str = schedule.loc[(j,m), 'arr_str'].iloc[0]
                dur = xf-xs
                print(f"colors = {colors[len(trips)%jdx]}")
                ax.plot([xs, xf], [mdx]*2, c=colors[len(trips)%jdx], **bar_style)
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
                ax.text((xs + xf)/2, mdx, j, **text)
                ax.text(xs, mdx, xs_str, **text)
                ax.text(xf, mdx, xf_str, **text)
                
    ax.set_title(f'{title} Bus Schedule')
    ax.set_ylabel('Bus ID')
    
    s = buses
    ax.set_ylim(0.5, len(s) + 0.5)
    ax.set_yticks(range(1, 1 + len(s)))
    ax.set_yticklabels(s)
    ax.text(makespan, ax.get_ylim()[0]-0.2, "{0:0.1f}".format(makespan), ha='center', va='top')
    ax.plot([makespan]*2, ax.get_ylim(), 'r--')
    ax.set_xlabel('Time')
    ax.grid(True)    
    fig.tight_layout()
    return fig


def visualizeTripsBuses(results, title):
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
    fig, ax = plt.subplots(2,1, figsize=(12, 5+(len(trips)+len(buses))/4))
    for jdx, j in enumerate(trips, 1):
        for mdx, m in enumerate(buses, 1):
            if (j,m) in schedule.index:
                xs = schedule.loc[(j,m), 'dep_time']
                xf = schedule.loc[(j,m), 'arr_time']
                xs_str = schedule.loc[(j,m), 'dep_str']
                xf_str = schedule.loc[(j,m), 'arr_str']
#                 print(type(schedule.loc[(j,m), 'dep_time']))
#                 xs = schedule.loc[(j,m), 'dep_time'].iloc[0]
#                 xf = schedule.loc[(j,m), 'arr_time'].iloc[0]
#                 xs_str = schedule.loc[(j,m), 'dep_str'].iloc[0]
#                 xf_str = schedule.loc[(j,m), 'arr_str'].iloc[0]
                dur = xf-xs
                ax[0].plot([xs, xf], [jdx]*2, c=colors[mdx%len(trips)], **bar_style)
                ax[1].plot([xs, xf], [mdx]*2, c=colors[jdx%len(trips)], **bar_style)
                if dur < 120:
                    text = small_text_style
                    xs += 10
                    xf -= 10
                else:
                    text = text_style
                    xs += 20
                    xf -= 20
                ax[0].text((xs + xf)/2, jdx, m, **text)
                ax[0].text(xs, jdx, xs_str, **text)
                ax[0].text(xf, jdx, xf_str, **text)
                ax[1].text((xs + xf)/2, mdx, j, **text)
                ax[1].text(xs, mdx, xs_str, **text)
                ax[1].text(xf, mdx, xf_str, **text)
                
    ax[0].set_title(f'{title} Trip Schedule')
    ax[0].set_ylabel('Trip ID')
    ax[1].set_title(f'{title} Bus Schedule')
    ax[1].set_ylabel('Bus ID')
    
    for idx, s in enumerate([trips, buses]):
        ax[idx].set_ylim(0.5, len(s) + 0.5)
        ax[idx].set_yticks(range(1, 1 + len(s)))
        ax[idx].set_yticklabels(s)
        ax[idx].text(makespan, ax[idx].get_ylim()[0]-0.2, "{0:0.1f}".format(makespan), ha='center', va='top')
        ax[idx].plot([makespan]*2, ax[idx].get_ylim(), 'r--')
        ax[idx].set_xlabel('Time')
        ax[idx].grid(True)
        
    fig.tight_layout()
    return fig

def visualizeResult(schedules, all_sch, title):
    cs_ids = all_sch.loc[all_sch.type == "cs", 'ID'].to_list()
    trips_ids = all_sch.loc[all_sch.type == "trip", 'ID'].to_list()
    depot = all_sch.loc[all_sch.type == "depot", 'ID'].to_list()
    res_dict = []
    for bus_id, schedule in enumerate(schedules, start=1):
        for s in schedule:
            bus = {}
            bus['bus_id'] = int(bus_id)
            if s in cs_ids:
                idx = schedule.index(s)
                bus['trip_id'] = all_sch.loc[all_sch.ID == s]['name'].iloc[0]
                bus['dep_time'] = (all_sch.loc[all_sch.ID == schedule[idx-1], 'arr_time'].iloc[0]) + 20
                bus['arr_time'] = bus['dep_time'] + all_sch.loc[all_sch.ID == s, 'duration'].iloc[0]
                print(all_sch.loc[all_sch.ID == idx, 'duration'].iloc[0])
                bus['duration'] = all_sch.loc[all_sch.ID == s, 'duration'].iloc[0] + 20
                bus['dep_terminal'] = all_sch.loc[all_sch.ID == schedule[idx-1], 'arr_term'].iloc[0]
                bus['arr_terminal'] = all_sch.loc[all_sch.ID == schedule[idx-1], 'arr_term'].iloc[0]
                print(bus)
            elif s in depot:
                continue
            else:
                print(all_sch)
                bus['trip_id'] = all_sch.loc[all_sch.ID == s]['name'].iloc[0]
                bus['dep_time'] = all_sch.loc[all_sch.ID == s, 'dep_time'].iloc[0]
                bus['arr_time'] = all_sch.loc[all_sch.ID == s, 'arr_time'].iloc[0]
                bus['duration'] = all_sch.loc[all_sch.ID == s, 'duration'].iloc[0]
                bus['dep_terminal'] = all_sch.loc[all_sch.ID == s, 'dep_term'].iloc[0]
                bus['arr_terminal'] = all_sch.loc[all_sch.ID == s, 'arr_term'].iloc[0]
#             print(bus)
            res_dict.append(bus)
    schedule_df = pd.DataFrame(res_dict)
    title=title
    fig = visualize(res_dict, title)
    fig.savefig(title)
    return schedule_df


def saveToExcel(all_schs, arcs, recharge_arcs, cs):
    durations = {idx: {'duration': all_schs.loc[idx, 'duration']} for idx in all_schs.index if idx != 0}
    n = len(all_schs)
    cond_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                cond_mat[i,j] = (i,j) in arcs.keys()
    gamma = pd.DataFrame(cond_mat, columns=all_schs.index, index=all_schs.index)
    n = len(all_schs)
    cond_mat  = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if (i,j) in arcs:
                cond_mat[i,j] = arcs[(i,j)]['duration']
            else:
                if i != j and ((i in cs and j not in cs) or (i not in cs and j in cs)):
                    cond_mat[i,j] = 30
    delta = pd.DataFrame(cond_mat, columns=all_schs.index, index=all_schs.index)
    n = len(all_schs)
    cond_mat = np.zeros((n * len(cs), n))
    for idx, k in enumerate(cs, start=1):
        for i in range(n):
            for j in range(n):
                if (i,j,k) in recharge_arcs:
    #                 print(f"i = {i}... j = {j}... k = {k}... mat = {(k-30-1) * (30+1)+i}... j")
                    cond_mat[(k-30-1) * (30+1)+i,j] = (i, j, k) in recharge_arcs

    phi = pd.DataFrame(cond_mat, columns=range(n), index=range(n * len(cs)))

    with pd.ExcelWriter(f"test.xlsx") as writer:
        all_schs.to_excel(writer, sheet_name="main", index=False)
        gamma.to_excel(writer, sheet_name="gamma", index=False)
        delta.to_excel(writer, sheet_name="delta", index=False)
        phi.to_excel(writer, sheet_name="phi", index=False)

import matplotlib.pyplot as plt

# visualize BusTrips data
def drawBusTrips(data):
    bar_style = {"alpha": 1.0, "lw": 10, "solid_capstyle": "butt"}
    text_style = {
        "fontsize": 5,
        "color": "white",
        "weight": "bold",
        "ha": "center",
        "va": "center",
    }
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, xlabel="Departure/Arrival Time", title="Bus Schedule")
    ax.get_yaxis().set_visible(False)
    idx = 1
    print(data.head())
    for id, row in data.iterrows():
        name, dep_time, arr_time, dep_city, arr_city = row
        ax.plot([dep_time, arr_time], [idx] * 2, "gray", **bar_style)
        ax.text((dep_time + arr_time) / 2, idx, f"Trip {name}", **text_style)
        ax.text(dep_time+20, idx, f"{dep_time}", {**text_style, 'fontsize': 5})
        ax.text(arr_time-20, idx, f"{arr_time}", {**text_style, 'fontsize': 5})
        idx += 1
    for hr in range(0, 1440, 100):
        ax.axvline(hr, alpha=0.1)
    ax.set_xlim(280, 1440)
    ax.set_xticks([200 + 100*i for i in range(1, 13)])
    return ax

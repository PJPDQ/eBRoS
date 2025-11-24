import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, ctx, dash_table
import plotly.express as px
import os
# Example Gantt data
df_gantt = pd.DataFrame({
    "bus_id": ["Bus1", "Bus1", "Bus1", "Bus1", "Bus1", "Bus2", "Bus2", "Bus2", "Bus2", "Bus2", "Bus3", "Bus3"],
    "trip_id": ["AC1", "CA2", "CS1", "CA4", "AC4", "AC2", "CA3", "CS1", "CA5", "AC5", "CA1", "AC3"],
    "dep_time_min": [555, 740, 880, 920, 1155, 675, 860, 1000, 1040, 1215, 570, 795],
    "arr_time_min": [695, 880, 930, 1060, 1295, 815, 1000, 1050, 1180, 1355, 710, 935]
})
base_date = pd.Timestamp("2025-01-01")
df_gantt["dep_time"] = base_date + pd.to_timedelta(df_gantt["dep_time_min"], unit="m")
df_gantt["arr_time"] = base_date + pd.to_timedelta(df_gantt["arr_time_min"], unit="m")
time_range = pd.date_range(base_date, base_date + pd.Timedelta("24h"), freq="10min")
df_gantt['energy_required'] = df_gantt['arr_time_min'] - df_gantt['dep_time_min']
D_MAX = 350
total_delay = 0
# -----------------------------------------------------------
# Initialize Dash App
# -----------------------------------------------------------
app = Dash(__name__)
app.title = "Electric Bus Rostering Dashboard"

instances = {"10T": "10Trips", "20T": "20Trips", "30T": "30Trips", "40T": "40Trips", "54T": "54Trips", "100T": "100Trips"}

# -----------------------------------------------------------
# Layout
# -----------------------------------------------------------
# main_content = html.Div(
#     [
#         html.Div([
#             dcc.Graph(id="gantt", style={"width": "48%", "display": "inline-block"}),
#             dcc.Graph(id="gantt_3cs", style={"width": "48%", "display": "inline-block"})
#         ]),
#         html.Hr(),
#         html.H4(f"1CS Battery State of Charge (SoC)"),
#         dcc.Graph(id="soc_line"),
#         html.H4(f"3CS Battery State of Charge (SoC)"),
#         dcc.Graph(id="soc_line_3")
#     ],
#     style={"width": "75%", "padding": "15px"}
# )
# side_bar = html.Div(
#     [
#         html.H4("Select Instance"),
#         dcc.Dropdown(
#             id="instance_selector",
#             options=[{"label": k, "value": k} for k in instances.keys()],
#             value="10T",
#             clearable=False
#         ),
#         html.Div(id="instance_desc", style={"marginTop": "20px", "fontStyle": "italic"}),
#         html.H4("Data Table 1CS"),
#         html.Div(id="df_gantt"),
#         html.H4("Data Table 3CS"),
#         html.Div(id="df_gantt_3")
#     ],
#     style={
#         "width": "20%",
#         "padding": "15px",
#         "backgroundColor": "#f5f5f5",
#         "borderRadius": "10px"
#     }
# )
app.layout = html.Div([
    html.H2("Electric Bus Rostering Dashboard", style={"textAlign": "center"}),

    html.Div([
        # LEFT PANEL – Control Panel
        html.Div([
            html.H4("Select Instance"),
            dcc.Dropdown(
                id="instance_selector",
                options=[{"label": k, "value": k} for k in instances.keys()],
                value="10T",
                clearable=False
            ),
            html.Div(id="instance_desc", style={"marginTop": "20px", "fontStyle": "italic"}),
            html.H4("Data Table 1CS"),
            html.H4([
                "Total Delay",
                html.Span(
                    id="total-delay-1",
                    style={"fontSize": "28px", "fontWeight": "bold", "marginBottom": "20px"}
                )
            ]),
            html.Div(id="df_gantt"),
            html.H4("Data Table 3CS"),
            html.H4([
                "Total Delay", 
                html.Span(
                    id="total-delay-3",
                    style={"fontSize": "28px", "fontWeight": "bold", "marginBottom": "20px"})
            ]),
            html.Div(id="df_gantt_3")
        ],
        style={
            "width": "20%",
            "padding": "15px",
            "backgroundColor": "#f5f5f5",
            "borderRadius": "10px"
        }),

        # RIGHT PANEL – Figures
        html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(id="gantt", style={"width": "48%", "display": "inline-block"}),
                    dcc.Graph(id="gantt_3cs", style={"width": "48%", "display": "inline-block"})
                ]),
                html.Hr(),
                html.H4(f"1CS Battery State of Charge (SoC)"),
                dcc.Graph(id="soc_line"),
                html.H4(f"3CS Battery State of Charge (SoC)"),
                dcc.Graph(id="soc_line_3")
            ])
        ],
        style={"width": "75%", "padding": "15px"})
    ],
    style={"display": "flex", "gap": "20px"})
])

# 1️⃣ Update Gantt Charts when instance changes
@app.callback(
    Output("df_gantt", "children"),
    Output("df_gantt_3", "children"),
    Output("total-delay-1", "children"),
    Output("total-delay-3", "children"),
    Output("gantt", "figure"),
    Output("gantt_3cs", "figure"),    
    Output("soc_line", "figure", allow_duplicate=True),
    Output("soc_line_3", "figure", allow_duplicate=True),
    Input("instance_selector", "value"),
    prevent_initial_call="initial_duplicate"
)
def test_instances(case):
    print(f"CASE!!! = {case}")
    work_dir = f"./FINAL_TimeLimit/{case}rips/"
    fn = f"{work_dir}CHSSA_{case}1CS_df_BEFORE.csv"
    if not os.path.isfile(fn):
        fn = f"{work_dir}CHSSA_{case}1CS_df.csv"
    single = pd.read_csv(fn)
    df_gantt = single.drop(columns="Unnamed: 0")
    df_gantt['dep_time_min'] = df_gantt['dep_time']
    df_gantt['arr_time_min'] = df_gantt['arr_time']
    df_gantt["dep_time"] = base_date + pd.to_timedelta(df_gantt["dep_time_min"], unit="m")
    df_gantt["arr_time"] = base_date + pd.to_timedelta(df_gantt["arr_time_min"], unit="m")
    df_gantt['energy_required'] = df_gantt['arr_time_min'] - df_gantt['dep_time_min']
    df_gantt['id'] = df_gantt['bus_id'].apply(lambda x: f"Bus{x}")

    fn = f"{work_dir}CHSSA_{case}3CS_df_BEFORE.csv"
    if not os.path.isfile(fn):
        fn = f"{work_dir}CHSSA_{case}3CS_df.csv"
    many = pd.read_csv(fn)
    df_gantt_3 = many.drop(columns="Unnamed: 0")
    df_gantt_3.head()
    df_gantt_3['dep_time_min'] = df_gantt_3['dep_time']
    df_gantt_3['arr_time_min'] = df_gantt_3['arr_time']
    df_gantt_3["dep_time"] = base_date + pd.to_timedelta(df_gantt_3["dep_time_min"], unit="m")
    df_gantt_3["arr_time"] = base_date + pd.to_timedelta(df_gantt_3["arr_time_min"], unit="m")
    df_gantt_3['energy_required'] = df_gantt_3['arr_time_min'] - df_gantt_3['dep_time_min']
    df_gantt_3['id'] = df_gantt_3['bus_id'].apply(lambda x: f"Bus{x}")
   
    soc_data = []
    for bus_id, group in df_gantt.groupby("bus_id"):
        soc = 100
        for id, trip in group.iterrows():
            trip_id = trip['trip_id']
            if trip_id.startswith("CS"):
                soc = 0
                time_range = pd.date_range(trip['dep_time'], trip['arr_time'] - pd.Timedelta("10m"), freq="10min")
                soc_data.append(pd.DataFrame({
                    "bus_id": bus_id,
                    "time": time_range,
                    "soc": np.linspace(soc, 100, len(time_range))
                }))
                soc = 100
            else:
                prev_soc = soc
                soc = soc - (trip['energy_required']/D_MAX * 100)
                time_range = pd.date_range(trip['dep_time'], trip['arr_time'], freq="10min")
                soc_data.append(pd.DataFrame({
                    "bus_id": bus_id,
                    "time": time_range,
                    "soc": np.linspace(soc, prev_soc, len(time_range))[::-1]
                }))
    df_soc = pd.concat(soc_data)


    soc_data_3 = []
    for bus_id, group in df_gantt_3.groupby("bus_id"):
        soc = 100
        for id, trip in group.iterrows():
            trip_id = trip['trip_id']
            if trip_id.startswith("CS"):
                soc = 0
                time_range = pd.date_range(trip['dep_time'], trip['arr_time'] - pd.Timedelta("10m"), freq="10min")
                soc_data_3.append(pd.DataFrame({
                    "bus_id": bus_id,
                    "time": time_range,
                    "soc": np.linspace(soc, 100, len(time_range))
                }))
                soc = 100
            else:
                prev_soc = soc
                soc = soc - (trip['energy_required']/D_MAX * 100)
                time_range = pd.date_range(trip['dep_time'], trip['arr_time'], freq="10min")
                soc_data_3.append(pd.DataFrame({
                    "bus_id": bus_id,
                    "time": time_range,
                    "soc": np.linspace(soc, prev_soc, len(time_range))[::-1]
                }))

    df_soc_3 = pd.concat(soc_data_3)  
    ########################################## 1 CS ##############################################
    # Create base color map dynamically
    trip_ids = df_gantt["trip_id"].unique()
    color_map = {t: "#1f77b4" for t in trip_ids}  # default all blue

    # Override "CS*" trip colors
    for t in trip_ids:
        if str(t).startswith("CS"):
            color_map[t] = "#ff7f0e"

    fig_gantt = px.timeline(
        df_gantt, 
        x_start="dep_time", x_end="arr_time", 
        y="id", color="trip_id",
        title=f"Electric Bus Timetable {case}1CS",
        color_discrete_map=color_map
    )

    fig_gantt.update_layout(
        yaxis=dict(
            tickmode="array",
            tickvals=df_gantt.id.unique(),
            ticktext=[f"{i}" for i in df_gantt.id.unique()]
        )
    )

    fig_gantt.update_yaxes(autorange="reversed")

    ########################################## 3 CS ##############################################
    trip_ids = df_gantt_3["trip_id"].unique()
    color_map = {t: "#1f77b4" for t in trip_ids}  # default all blue

    # Override "CS*" trip colors
    for t in trip_ids:
        if str(t).startswith("CS"):
            color_map[t] = "#ff7f0e"

    fig_gantt_3 = px.timeline(
        df_gantt_3, 
        x_start="dep_time", x_end="arr_time", 
        y="id", color="trip_id",
        title=f"Electric Bus Timetable {case}3CS",
        color_discrete_map=color_map
    )

    fig_gantt_3.update_layout(
        yaxis=dict(
            tickmode="array",
            tickvals=df_gantt_3.id.unique(),
            ticktext=[f"{i}" for i in df_gantt_3.id.unique()]
        )
    )
    
    fig = px.line(df_soc, x="time", y="soc", color="bus_id", title=f"{case}1CS State of Charge (All Buses)")
    fig.update_yaxes(title="State of Charge (%)", range=[0, 100])
    fig.update_xaxes(title="Time of Day", tickformat="%H:%M")
    
    fig2 = px.line(df_soc_3, x="time", y="soc", color="bus_id", title=f"{case}3CS State of Charge (All Buses)")
    fig2.update_yaxes(title="State of Charge (%)", range=[0, 100])
    fig2.update_xaxes(title="Time of Day", tickformat="%H:%M")

    fig_gantt_3.update_yaxes(autorange="reversed")

    # cols = ['bus_id', 'duration', 'dep_terminal', 'arr_terminal', 'dep_time_min', 'arr_time_min', 'difference']
    cols = ['bus_id', 'duration', 'difference']
    group_df = df_gantt.groupby("bus_id")[[ 
                'duration', 
                'difference'
            ]].sum().reset_index()
    group_df3 = df_gantt_3.groupby("bus_id")[[ 
                'duration', 
                'difference'
            ]].sum().reset_index()
    nbuses_1 = group_df.shape[0]
    nbuses_3 = group_df3.shape[0]
    delay_1 = group_df.difference.sum()
    delay_3 = group_df3.difference.sum()
    # return fig_gantt, fig_gantt_3, fig, fig2
    return \
        dash_table.DataTable(
            id="tab_gantt",
            columns=[{"name": i, "id": i} for i in cols],
            data=df_gantt.groupby('bus_id')[[ 
                'duration', 
                'difference'
            ]].sum().reset_index().to_dict("records"),
            style_table={"overflowY": "auto"}
        ), \
        dash_table.DataTable(
            id="tab_gantt_3",
            columns=[{"name": i, "id": i} for i in cols],
            data=df_gantt_3.groupby('bus_id')[[
                'duration', 
                'difference'
            ]].sum().reset_index().to_dict("records"),
            style_table={"overflowY": "auto"}
        ), \
        f"{nbuses_1}EBs and {delay_1}minutes", f"{nbuses_3}EBs and {delay_3}minutes", fig_gantt, fig_gantt_3, fig, fig2

# 2️⃣ Update SoC Line Chart based on clicks from either Gantt
@app.callback(
    Output("soc_line", "figure"),
    Output("soc_line_3", "figure"),
    [Input("gantt", "clickData"),
    Input("gantt_3cs", "clickData"),
    Input("instance_selector", "value")]
)
def show_soc_line(clickData, clickData3cs, case):
    triggered = ctx.triggered_id
    work_dir = f"./FINAL_TimeLimit/{case}rips/"
    fn = f"{work_dir}CHSSA_{case}1CS_df_BEFORE.csv"
    if not os.path.isfile(fn):
        fn = f"{work_dir}CHSSA_{case}1CS_df.csv"
    single = pd.read_csv(fn)
    df_gantt = single.drop(columns="Unnamed: 0")
    df_gantt['dep_time_min'] = df_gantt['dep_time']
    df_gantt['arr_time_min'] = df_gantt['arr_time']
    df_gantt["dep_time"] = base_date + pd.to_timedelta(df_gantt["dep_time_min"], unit="m")
    df_gantt["arr_time"] = base_date + pd.to_timedelta(df_gantt["arr_time_min"], unit="m")
    df_gantt['energy_required'] = df_gantt['arr_time_min'] - df_gantt['dep_time_min']
    df_gantt['id'] = df_gantt['bus_id'].apply(lambda x: f"Bus{x}")

    fn = f"{work_dir}CHSSA_{case}3CS_df_BEFORE.csv"
    if not os.path.isfile(fn):
        fn = f"{work_dir}CHSSA_{case}3CS_df.csv"
    many = pd.read_csv(fn)
    df_gantt_3 = many.drop(columns="Unnamed: 0")
    df_gantt_3.head()
    df_gantt_3['dep_time_min'] = df_gantt_3['dep_time']
    df_gantt_3['arr_time_min'] = df_gantt_3['arr_time']
    df_gantt_3["dep_time"] = base_date + pd.to_timedelta(df_gantt_3["dep_time_min"], unit="m")
    df_gantt_3["arr_time"] = base_date + pd.to_timedelta(df_gantt_3["arr_time_min"], unit="m")
    df_gantt_3['energy_required'] = df_gantt_3['arr_time_min'] - df_gantt_3['dep_time_min']
    df_gantt_3['id'] = df_gantt_3['bus_id'].apply(lambda x: f"Bus{x}")
    
    soc_data = []
    for bus_id, group in df_gantt.groupby("bus_id"):
        soc = 100
        for id, trip in group.iterrows():
            trip_id = trip['trip_id']
            if trip_id.startswith("CS"):
                soc = 0
                time_range = pd.date_range(trip['dep_time'], trip['arr_time'] - pd.Timedelta("10m"), freq="10min")
                soc_data.append(pd.DataFrame({
                    "bus_id": bus_id,
                    "time": time_range,
                    "soc": np.linspace(soc, 100, len(time_range))
                }))
                soc = 100
            else:
                prev_soc = soc
                soc = soc - (trip['energy_required']/D_MAX * 100)
                time_range = pd.date_range(trip['dep_time'], trip['arr_time'], freq="10min")
                soc_data.append(pd.DataFrame({
                    "bus_id": bus_id,
                    "time": time_range,
                    "soc": np.linspace(soc, prev_soc, len(time_range))[::-1]
                }))
    df_soc = pd.concat(soc_data)

    soc_data_3 = []
    for bus_id, group in df_gantt_3.groupby("bus_id"):
        soc = 100
        for id, trip in group.iterrows():
            trip_id = trip['trip_id']
            if trip_id.startswith("CS"):
                soc = 0
                time_range = pd.date_range(trip['dep_time'], trip['arr_time'] - pd.Timedelta("10m"), freq="10min")
                soc_data_3.append(pd.DataFrame({
                    "bus_id": bus_id,
                    "time": time_range,
                    "soc": np.linspace(soc, 100, len(time_range))
                }))
                soc = 100
            else:
                prev_soc = soc
                soc = soc - (trip['energy_required']/D_MAX * 100)
                time_range = pd.date_range(trip['dep_time'], trip['arr_time'], freq="10min")
                soc_data_3.append(pd.DataFrame({
                    "bus_id": bus_id,
                    "time": time_range,
                    "soc": np.linspace(soc, prev_soc, len(time_range))[::-1]
                }))

    df_soc_3 = pd.concat(soc_data_3)  
    bus_clicked = bus3_clicked = None
    if triggered == "gantt" and clickData:
        bus_clicked = clickData["points"][0]["y"]
    elif triggered == "gantt_3cs" and clickData3cs:
        bus3_clicked = clickData3cs["points"][0]["y"]
    else:
        print(f"CLIKED DATA = {clickData}")
        fig = px.line(df_soc, x="time", y="soc", color="bus_id", title="State of Charge (All Buses)")
        fig.update_yaxes(title="State of Charge (%)", range=[0, 100])
        fig.update_xaxes(title="Time of Day", tickformat="%H:%M")
        
        fig_3 = px.line(df_soc_3, x="time", y="soc", color="bus_id", title="State of Charge (All Buses)")
        fig_3.update_yaxes(title="State of Charge (%)", range=[0, 100])
        fig_3.update_xaxes(title="Time of Day", tickformat="%H:%M") 
        return fig, fig_3
    
    if bus_clicked:
        df_selected = df_soc[df_soc["bus_id"] == bus_clicked]
        df_selected_3 = df_soc_3[df_soc_3["bus_id"] == bus_clicked]
    elif bus3_clicked:
        df_selected = df_soc[df_soc["bus_id"] == bus3_clicked]
        df_selected_3 = df_soc_3[df_soc_3["bus_id"] == bus3_clicked]
        bus_clicked = bus3_clicked
     
    
    fig = px.line(df_soc, x="time", y="soc", color="bus_id", title="State of Charge (All Buses)")
    fig.update_yaxes(title="State of Charge (%)", range=[0, 100])
    fig.update_xaxes(title="Time of Day", tickformat="%H:%M")
    
    fig2 = px.line(df_soc_3, x="time", y="soc", color="bus_id", title="State of Charge (All Buses)")
    fig2.update_yaxes(title="State of Charge (%)", range=[0, 100])
    fig2.update_xaxes(title="Time of Day", tickformat="%H:%M")
    
    clickData = clickData3cs = None
    return fig, fig2

app.run(debug=False, host='0.0.0.0', port=8501)
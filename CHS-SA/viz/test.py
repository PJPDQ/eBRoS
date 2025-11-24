# # app.py
# from flask import Flask, render_template, request, jsonify
# import pandas as pd
# import plotly.express as px
# import plotly.io as pio

# app = Flask(__name__)

# # sample dataframe
# df = pd.DataFrame({
#     'bus_id': ['Bus1', 'Bus1', 'Bus2', 'Bus2'],
#     'trip_id': ['T1', 'T2', 'T3', 'T4'],
#     'dep_time': [0, 300, 100, 500],
#     'arr_time': [200, 450, 400, 700],
#     'SoC_start': [100, 60, 100, 70],
#     'SoC_end': [60, 80, 70, 90]
# })

# @app.route('/')
# def index():
#     # sample dataframe
#     data = {
#         "bus_id": ["Bus1", "Bus1", "Bus2", "Bus2"],
#         "trip_id": ["T1", "T2", "T3", "T4"],
#         "dep_time_min": [420, 600, 480, 720],  # departure time in minutes since midnight
#         "arr_time_min": [480, 660, 540, 780]   # arrival time in minutes since midnight
#     }

#     df = pd.DataFrame(data)
#     # Convert minutes to datetime (e.g., Jan 1, 2025)
#     base_date = pd.Timestamp("2025-01-01")
#     df["dep_time_dt"] = base_date + pd.to_timedelta(df["dep_time_min"], unit="m")
#     df["arr_time_dt"] = base_date + pd.to_timedelta(df["arr_time_min"], unit="m")
#     df = df[['bus_id', 'trip_id', 'dep_time_dt', 'arr_time_dt']]
#     print("-"*100)
#     print(df.dtypes)
#     fig = px.timeline(df, x_start="dep_time_dt", x_end="arr_time_dt", y="bus_id",
#                       color="trip_id", title="Electric Bus Rostering Overview")
    
    
#     print(df.head())
#     print("-"*100)
#     fig.update_yaxes(autorange="reversed")  # Gantt charts usually reversed
#     fig.update_xaxes(title="Time of Day", tickformat="%H:%M")
#     fig.write_html("my_interactive_plot.html")
#     gantt_html = pio.to_html(fig, full_html=False, include_plotlyjs=True)
#     return render_template("index.html", gantt=gantt_html)

# @app.route('/bus_soc/<bus_id>')
# def bus_soc(bus_id):
#     subset = df[df['bus_id'] == bus_id]
#     soc_df = pd.DataFrame({
#         'time': subset[['dep_time', 'arr_time']].values.flatten(),
#         'SoC': subset[['SoC_start', 'SoC_end']].values.flatten()
#     }).sort_values(by='time')
#     soc_fig = px.line(soc_df, x='time', y='SoC', title=f"SoC Over Time for {bus_id}")
#     return jsonify({'plot': pio.to_html(soc_fig, full_html=False)})

# if __name__ == '__main__':
#     app.run(debug=True)

from dash import Dash, dcc, html, Input, Output, ctx
import plotly.express as px
import pandas as pd

# -------------------------
# Example datasets
# -------------------------
datasets = {
    "10 Trips": {
        "1CS": pd.DataFrame({
            "bus_id": ["EB1", "EB1", "EB2", "EB2"],
            "dep_time": [0, 400, 100, 600],
            "arr_time": [300, 700, 500, 900],
            "trip_id": ["T1", "T2", "T3", "T4"],
            "soc_start": [100, 75, 100, 60],
            "soc_end":   [75, 50, 60, 85]
        }),
        "3CS": pd.DataFrame({
            "bus_id": ["EB3", "EB3", "EB4", "EB4"],
            "dep_time": [0, 500, 200, 800],
            "arr_time": [300, 900, 600, 1000],
            "trip_id": ["T5", "T6", "T7", "T8"],
            "soc_start": [100, 80, 100, 70],
            "soc_end":   [80, 95, 70, 90]
        })
    },
    "20 Trips": {
        "1CS": pd.DataFrame({
            "bus_id": ["EB5", "EB5", "EB6", "EB6"],
            "dep_time": [0, 300, 350, 700],
            "arr_time": [200, 600, 650, 950],
            "trip_id": ["T9", "T10", "T11", "T12"],
            "soc_start": [100, 80, 95, 65],
            "soc_end":   [80, 60, 70, 90]
        }),
        "3CS": pd.DataFrame({
            "bus_id": ["EB7", "EB7", "EB8", "EB8"],
            "dep_time": [0, 200, 400, 800],
            "arr_time": [150, 450, 650, 1000],
            "trip_id": ["T13", "T14", "T15", "T16"],
            "soc_start": [100, 78, 98, 72],
            "soc_end":   [78, 92, 72, 95]
        })
    }
}

# -------------------------
# App + layout
# -------------------------
app = Dash(__name__)
app.layout = html.Div([
    html.H2("EBRP Dashboard", style={"textAlign": "center"}),

    html.Div([
        # Control column
        html.Div([
            html.H4("Select Instance"),
            dcc.Dropdown(
                id="instance_selector",
                options=[{"label": k, "value": k} for k in datasets.keys()],
                value="10 Trips",
                clearable=False
            ),
            html.Div(id="instance_desc", style={"marginTop": "12px", "fontStyle": "italic"})
        ], style={"width": "20%", "padding": "10px", "backgroundColor": "#f7f7f7"}),

        # Figures column
        html.Div([
            html.Div([
                dcc.Graph(id="gantt_1cs", style={"width": "49%", "display": "inline-block"}),
                dcc.Graph(id="gantt_3cs", style={"width": "49%", "display": "inline-block"})
            ]),
            html.Hr(),
            html.H4("State of Charge (SoC)"),
            dcc.Graph(id="soc_line")
        ], style={"width": "78%", "padding": "10px"})
    ], style={"display": "flex", "gap": "2%"}),

    # Store to hold the user-selected instance (intermediate state)
    dcc.Store(id="selected_instance_store")
])


# -------------------------
# Callback 1: update gantts + store selected instance
# -------------------------
@app.callback(
    Output("gantt_1cs", "figure"),
    Output("gantt_3cs", "figure"),
    Output("instance_desc", "children"),
    Output("selected_instance_store", "data"),
    Input("instance_selector", "value")
)
def update_gantts(selected_instance):
    df1 = datasets[selected_instance]["1CS"]
    df2 = datasets[selected_instance]["3CS"]

    fig1 = px.timeline(df1, x_start="dep_time", x_end="arr_time", y="bus_id",
                       color="trip_id", title=f"{selected_instance} — 1CS")
    fig1.update_yaxes(autorange="reversed")
    fig1.update_layout(clickmode="event+select")

    fig2 = px.timeline(df2, x_start="dep_time", x_end="arr_time", y="bus_id",
                       color="trip_id", title=f"{selected_instance} — 3CS")
    fig2.update_yaxes(autorange="reversed")
    fig2.update_layout(clickmode="event+select")

    desc = f"Showing {selected_instance} (compare 1CS vs 3CS). Click any bar to view that bus SoC."
    # store the instance name (or you may store preprocessed figures/data)
    return fig1, fig2, desc, selected_instance


# -------------------------
# Callback 2: single callback that builds soc_line from clickData or store data
# -------------------------
@app.callback(
    Output("soc_line", "figure"),
    Input("gantt_1cs", "clickData"),
    Input("gantt_3cs", "clickData"),
    Input("selected_instance_store", "data"),
)
def update_soc(click1, click2, selected_instance):
    # get which component triggered the callback (None at initial page load)
    triggered = ctx.triggered_id

    # if no instance stored, return empty fig
    if not selected_instance:
        return px.line(title="No instance selected")

    # default: show aggregated SoC for all buses in selected instance (1CS data + 3CS data combined or whichever you prefer)
    if triggered is None:
        # show all buses combined (both 1CS and 3CS if desired)
        df_all_1 = datasets[selected_instance]["1CS"].copy()
        df_all_3 = datasets[selected_instance]["3CS"].copy()
        # construct simple per-bus lines from trip-level soc_start/soc_end
        def build_bus_soc_df(df):
            rows = []
            for bus, g in df.groupby("bus_id"):
                g = g.sort_values("dep_time")
                times = []
                socs = []
                for _, r in g.iterrows():
                    times.extend([r["dep_time"], r["arr_time"]])
                    socs.extend([r["soc_start"], r["soc_end"]])
                rows.append(pd.DataFrame({"bus_id": bus, "time": times, "soc": socs}))
            return pd.concat(rows, ignore_index=True)
        soc_all_1 = build_bus_soc_df(df_all_1)
        soc_all_3 = build_bus_soc_df(df_all_3)
        soc_all = pd.concat([soc_all_1, soc_all_3], ignore_index=True)
        fig = px.line(soc_all, x="time", y="soc", color="bus_id", title=f"SoC for all buses ({selected_instance})")
        fig.update_yaxes(range=[0, 100], title="SoC (%)")
        fig.update_xaxes(title="Time (minutes)")
        return fig

    # if triggered by gantt_1cs
    if triggered == "gantt_1cs" and click1:
        pt = click1["points"][0]
        bus = pt["y"]
        df = datasets[selected_instance]["1CS"]
    elif triggered == "gantt_3cs" and click2:
        pt = click2["points"][0]
        bus = pt["y"]
        df = datasets[selected_instance]["3CS"]
    else:
        # clicked elsewhere: fallback to aggregated view
        return px.line(title="Click on a Gantt bar to load a bus SoC")

    # Build SoC trace for the selected bus from trip-level soc_start/soc_end
    df_bus = df[df["bus_id"] == bus].sort_values("dep_time")
    times, socs = [], []
    for _, r in df_bus.iterrows():
        times += [r["dep_time"], r["arr_time"]]
        socs += [r["soc_start"], r["soc_end"]]

    if len(times) == 0:
        return px.line(title=f"No SoC data for {bus}")

    soc_df = pd.DataFrame({"time": times, "soc": socs})
    fig = px.line(soc_df, x="time", y="soc", title=f"SoC for {bus} ({selected_instance})", markers=True)
    fig.update_yaxes(range=[0, 100], title="SoC (%)")
    fig.update_xaxes(title="Time (minutes)")
    return fig


if __name__ == "__main__":
    app.run(debug=True)


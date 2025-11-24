import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
import pandas as pd

df = pd.DataFrame({
    "route": ["A", "B", "C"],
    "delay": [12, 8, 15]
})

app = dash.Dash(__name__)

sidebar = html.Div(
    [
        html.H3("Summary"),
        html.Label("Select Route"),
        dcc.Dropdown(
            id="route-dd",
            options=[{"label": r, "value": r} for r in df["route"].unique()],
            value="A"
        ),
        html.Div(id="table-container")    # 🟦 callback output goes here
    ],
    style={
        "position": "fixed",
        "width": "260px",
        "height": "100%",
        "padding": "20px",
        "backgroundColor": "#f8f9fa",
    }
)

main_content = html.Div(
    ["Main content here"],
    style={"marginLeft": "280px", "padding": "20px"}
)

app.layout = html.Div([sidebar, main_content])

@app.callback(
    Output("table-container", "children"),
    Input("route-dd", "value")
)
def update_sidebar_table(selected_route):
    filtered_df = df[df["route"] == selected_route]

    return dash_table.DataTable(
        id="dynamic-table",
        columns=[{"name": i, "id": i} for i in filtered_df.columns],
        data=filtered_df.to_dict("records"),
        style_cell={"padding": "6px"},
        style_table={"overflowY": "auto"}
    )

if __name__ == "__main__":
    app.run(debug=True)




# from dash import Dash, dcc, html, Input, Output
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go

# # ---------------------------------
# # EXAMPLE DATA
# # ---------------------------------
# base_date = pd.Timestamp("2025-01-01")

# # Gantt chart data
# df_gantt = pd.DataFrame({
#     "bus_id": ["Bus1", "Bus1", "Bus2", "Bus2"],
#     "trip_id": ["T1", "T2", "T3", "T4"],
#     "dep_time_min": [420, 600, 480, 720],
#     "arr_time_min": [480, 660, 540, 780]
# })
# df_gantt["dep_time"] = base_date + pd.to_timedelta(df_gantt["dep_time_min"], unit="m")
# df_gantt["arr_time"] = base_date + pd.to_timedelta(df_gantt["arr_time_min"], unit="m")

# # SoC data
# time_range = pd.date_range(base_date, base_date + pd.Timedelta("24h"), freq="10min")
# soc_data = []
# for bus in ["Bus1", "Bus2", "Bus3"]:
#     soc = 100 - np.linspace(0, 70, len(time_range)) + np.random.normal(0, 3, len(time_range))
#     soc_data.append(pd.DataFrame({
#         "bus_id": bus,
#         "time": time_range,
#         "soc": np.clip(soc, 0, 100)
#     }))
# df_soc = pd.concat(soc_data)

# # ---------------------------------
# # INITIAL FIGURES
# # ---------------------------------

# # 1️⃣ Gantt Chart
# fig_gantt = px.timeline(
#     df_gantt,
#     x_start="dep_time",
#     x_end="arr_time",
#     y="bus_id",
#     color="trip_id",
#     title="Electric Bus Timetable"
# )
# fig_gantt.update_yaxes(autorange="reversed")
# fig_gantt.update_layout(clickmode="event+select")

# # 2️⃣ SoC Chart (initial: all buses)
# fig_soc_all = px.line(df_soc, x="time", y="soc", color="bus_id", title="State of Charge (All Buses)")
# fig_soc_all.update_yaxes(title="State of Charge (%)", range=[0, 100])
# fig_soc_all.update_xaxes(title="Time of Day", tickformat="%H:%M")

# # ---------------------------------
# # DASH APP
# # ---------------------------------
# app = Dash(__name__)

# app.layout = html.Div([
#     html.H3("Electric Bus Rostering Dashboard"),
#     dcc.Graph(id="gantt", figure=fig_gantt),
#     html.Hr(),
#     html.H4("Battery State of Charge (SoC)"),
#     dcc.Graph(id="soc_line", figure=fig_soc_all)
# ])


# # ---------------------------------
# # CALLBACK: HANDLE CLICK EVENTS
# # ---------------------------------
# @app.callback(
#     Output("soc_line", "figure"),
#     Input("gantt", "clickData")
# )
# def update_soc_plot(clickData):
#     # No click or cleared selection: show ALL SoC lines
#     if not clickData or "points" not in clickData:
#         fig = px.line(df_soc, x="time", y="soc", color="bus_id", title="State of Charge (All Buses)")
#         fig.update_yaxes(title="State of Charge (%)", range=[0, 100])
#         fig.update_xaxes(title="Time of Day", tickformat="%H:%M")
#         return fig

#     # Extract the clicked bus_id
#     bus_clicked = clickData["points"][0]["y"]
#     df_selected = df_soc[df_soc["bus_id"] == bus_clicked]

#     # Filter for the selected bus
#     fig = px.line(df_selected, x="time", y="soc", color="bus_id", title=f"SoC for {bus_clicked}")
#     fig.update_yaxes(title="State of Charge (%)", range=[0, 100])
#     fig.update_xaxes(title="Time of Day", tickformat="%H:%M")

#     return fig


# if __name__ == "__main__":
#     app.run(debug=True)



# # from dash import Dash, dcc, html, Input, Output
# # import plotly.express as px
# # app = Dash(__name__)

# # import pandas as pd
# # import numpy as np
# # import plotly.express as px
# # import plotly.graph_objects as go
# # from plotly.subplots import make_subplots

# # # Example Gantt data
# # df_gantt = pd.DataFrame({
# #     "bus_id": ["Bus1", "Bus1", "Bus1", "Bus1", "Bus1", "Bus2", "Bus2", "Bus2", "Bus2", "Bus2", "Bus3", "Bus3"],
# #     "trip_id": ["AC1", "CA2", "CS1", "CA4", "AC4", "AC2", "CA3", "CS1", "CA5", "AC5", "CA1", "AC3"],
# #     "dep_time_min": [555, 740, 880, 920, 1155, 675, 860, 1000, 1040, 1215, 570, 795],
# #     "arr_time_min": [695, 880, 930, 1060, 1295, 815, 1000, 1050, 1180, 1355, 710, 935]
# # })
# # print(df_gantt.head())
# # base_date = pd.Timestamp("2025-01-01")
# # df_gantt["dep_time"] = base_date + pd.to_timedelta(df_gantt["dep_time_min"], unit="m")
# # df_gantt["arr_time"] = base_date + pd.to_timedelta(df_gantt["arr_time_min"], unit="m")
# # time_range = pd.date_range(base_date, base_date + pd.Timedelta("24h"), freq="10min")
# # print(f"time = {time_range}")
# # df_gantt['energy_required'] = df_gantt['arr_time_min'] - df_gantt['dep_time_min']
# # D_MAX = 350
# # soc_data = []
# # for bus_id, group in df_gantt.groupby("bus_id"):
# #     soc = 100
# #     for id, trip in group.iterrows():
# #         # print(f"trip = {trip}")
# #         trip_id = trip['trip_id']
# #         print(f"dep = {trip['dep_time']}")
# #         if trip_id.startswith("CS"):
# #             soc = 0
# #             time_range = pd.date_range(trip['dep_time'], trip['arr_time'] - pd.Timedelta("10m"), freq="10min")
# #             soc_data.append(pd.DataFrame({
# #                 "bus_id": bus_id,
# #                 "time": time_range,
# #                 "soc": np.linspace(soc, 100, len(time_range))
# #             }))
# #             soc = 100
# #         else:
# #             prev_soc = soc
# #             soc = soc - (trip['energy_required']/D_MAX * 100)
# #             time_range = pd.date_range(trip['dep_time'], trip['arr_time'], freq="10min")
# #             print(f"soc = {soc}... prev={prev_soc}")
# #             soc_data.append(pd.DataFrame({
# #                 "bus_id": bus_id,
# #                 "time": time_range,
# #                 "soc": np.linspace(soc, prev_soc, len(time_range))[::-1]
# #             }))
# #             print(f"{trip['trip_id']} ...\nNEW! ---- {soc_data}")
# # df_soc = pd.concat(soc_data)

# # print(f"SOC={df_soc.head(20)}")

# # # Create the Gantt chart
# # fig_gantt = px.timeline(
# #     df_gantt, 
# #     x_start="dep_time", x_end="arr_time", 
# #     y="bus_id", color="trip_id",
# #     title="Electric Bus Timetable"
# # )
# # fig_gantt.update_yaxes(autorange="reversed")

# # # Layout
# # app.layout = html.Div([
# #     html.H3("Electric Bus Rostering Dashboard"),
# #     dcc.Graph(id="gantt", figure=fig_gantt),
# #     html.Hr(),
# #     html.H4("Battery State of Charge (SoC)"),
# #     dcc.Graph(id="soc_line")
# # ])

# # # Callback: when user clicks a trip block on the Gantt
# # @app.callback(
# #     Output("soc_line", "figure"),
# #     Input("gantt", "clickData")
# # )
# # def show_soc_line(clickData):
# #     if not clickData:
# #         return go.Figure()

# #     # Get clicked bus_id
# #     bus_clicked = clickData["points"][0]["y"]

# #     # Filter the SoC data
# #     df_selected = df_soc[df_soc["bus_id"] == bus_clicked]

# #     # Create line chart
# #     fig = px.line(df_selected, x="time", y="soc", title=f"SoC over Time - {bus_clicked}")
# #     fig.update_yaxes(title="State of Charge (%)", range=[0, 100])
# #     fig.update_xaxes(title="Time of Day", tickformat="%H:%M")

# #     return fig


# # if __name__ == "__main__":
# #     app.run(debug=True)

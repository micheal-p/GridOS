import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import pandapower.plotting.plotly as pplotly
import json
import plotly
import plotly.graph_objs as go
import datetime
import traceback
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --- FLEET SPECS ---
EV_MODELS = [
    {"name": "Tesla Model 3", "capacity": 60, "max_rate": 11},      
    {"name": "Tesla Model S", "capacity": 100, "max_rate": 17},     
    {"name": "Cybertruck", "capacity": 123, "max_rate": 22}   
]

grid_state = {
    "battery_soc_kvarh": 8000, # Start at 80%
    "max_battery_kvarh": 10000, 
    "transformer_temp": 45.0, 
    "is_blackout": False, 
    "history": [],
    "start_time": None
}

IEEE_33_COORDS = {
    0: (0, 0),    1: (1, 0),    2: (2, 0),    3: (3, 0),    4: (4, 0),
    5: (5, 0),    6: (6, 0),    7: (7, 0),    8: (8, 0),    9: (9, 0),
    10: (10, 0),  11: (11, 0),  12: (12, 0),  13: (13, 0),  14: (14, 0),
    15: (15, 0),  16: (16, 0),  17: (17, 0),  
    18: (1, 1),   19: (2, 1),   20: (3, 1),   21: (4, 1),   
    22: (2, -1),  23: (3, -1),  24: (4, -1),                
    25: (5, 1),   26: (6, 1),   27: (7, 1),   28: (8, 1),   29: (9, 1), 
    30: (10, -1), 31: (11, -1), 32: (12, -1)                
}

def get_live_map(net, car_data, is_blackout):
    if not hasattr(net, 'bus_geodata') or net.bus_geodata is None or len(net.bus_geodata) == 0:
        net.bus_geodata = pd.DataFrame(index=net.bus.index, columns=["x", "y"])
    for bus_idx, (x, y) in IEEE_33_COORDS.items():
        if bus_idx in net.bus_geodata.index: net.bus_geodata.loc[bus_idx, "x"] = x; net.bus_geodata.loc[bus_idx, "y"] = y

    edge_x = []; edge_y = []
    line_color = '#bbb'
    if grid_state["transformer_temp"] > 100: line_color = '#FF9500' 
    if is_blackout: line_color = '#000'

    for _, line in net.line.iterrows():
        try:
            x0 = IEEE_33_COORDS[line.from_bus][0]; y0 = IEEE_33_COORDS[line.from_bus][1]
            x1 = IEEE_33_COORDS[line.to_bus][0]; y1 = IEEE_33_COORDS[line.to_bus][1]
            edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
        except KeyError: continue 

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=3, color=line_color), hoverinfo='none', mode='lines')

    node_x = []; node_y = []; node_text = []; node_color = []; node_size = []
    bus_counts = {i: 0 for i in net.bus.index}
    for car in car_data:
        if car['bus_loc'] in bus_counts: bus_counts[car['bus_loc']] += 1

    for i in net.bus.index:
        if i in IEEE_33_COORDS:
            node_x.append(IEEE_33_COORDS[i][0]); node_y.append(IEEE_33_COORDS[i][1])
            v_pu = 0.0 if is_blackout else 1.0
            if not is_blackout:
                try: v_pu = net.res_bus.at[i, 'vm_pu']
                except: v_pu = 1.0 
            
            count = bus_counts.get(i, 0)
            node_size.append(15 + (count * 4))
            node_text.append(f"<b>Bus {i}</b><br>Voltage: {v_pu:.2f}<br>EVs: {count}")
            
            if is_blackout: node_color.append('#000')
            elif v_pu < 0.95: node_color.append('#FF3B30')
            elif v_pu < 0.97: node_color.append('#FF9500')
            else: node_color.append('#34C759')

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
        marker=dict(showscale=not is_blackout, colorscale='RdYlGn', reversescale=False, color=node_color, size=node_size, colorbar=dict(thickness=15, title=dict(text='Voltage', side='right'), xanchor='left'), line=dict(width=2, color='#fff')))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(autosize=True, showlegend=False, hovermode='closest', margin=dict(b=20,l=20,r=20,t=20), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'))
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def run_simulation(num_cars, solar_intensity, strategy):
    net = pn.case33bw()
    
    if grid_state["start_time"] is None:
        grid_state["start_time"] = datetime.datetime.now().timestamp()

    # RESET
    if grid_state["is_blackout"]:
        grid_state["transformer_temp"] -= 10
        if grid_state["transformer_temp"] < 50: grid_state["is_blackout"] = False
        else: return { "min_voltage": 0.0, "total_solar_mw": 0, "grid_battery_percent": 0, "grid_status_text": "BLACKOUT", "time_estimate": "Offline", "net_flow_mw": 0, "transformer_temp": int(grid_state["transformer_temp"]), "cars": [], "history": grid_state["history"], "map": get_live_map(net, [], True), "start_ts": grid_state["start_time"] }

    # DEPLOY
    load_buses = net.load.bus.values
    active_buses = np.random.choice(load_buses, max(1, int(num_cars)), replace=True)
    car_data = []
    
    for bus in active_buses:
        model = np.random.choice(EV_MODELS)
        pp.create_load(net, bus=bus, p_mw=model['max_rate']/1000.0, q_mvar=0)
        car_data.append({
            "model": model['name'], "capacity": model['capacity'], "soc": np.random.randint(10, 80),
            "rate": model['max_rate'], "max_rate": model['max_rate'], "status": "charging", "time_to_full": "...", "penalty": 0, "bus_loc": int(bus)
        })

    # SOLAR (3MW Max Capacity for this grid size)
    total_solar_mw = 3.0 * (solar_intensity / 100.0)
    for bus in [18, 22, 25, 32]: pp.create_sgen(net, bus, p_mw=total_solar_mw/4, q_mvar=0)
    try: pp.runpp(net, numba=False)
    except: pass

    # PHYSICS
    min_voltage = min(net.res_bus.vm_pu)
    total_ev_load_mw = sum([c['rate'] for c in car_data]) / 1000.0
    
    # THERMAL
    # If load > 2MW, heat rises rapidly
    heat_gain = (total_ev_load_mw / 2.0) * 8.0 
    grid_state["transformer_temp"] += (heat_gain - 4.0) # -4 is cooling
    grid_state["transformer_temp"] = max(25, grid_state["transformer_temp"])

    status_msg = "Stable"
    if strategy == "smart":
        if grid_state["transformer_temp"] > 100:
            status_msg = "Cooling Active"
            grid_state["transformer_temp"] -= 10
            for car in car_data: car['rate'] *= 0.2; car['status'] = 'slow'; car['penalty'] = 999
        elif min_voltage < 0.95:
            status_msg = "Voltage Opt."
            for car in car_data: car['rate'] *= 0.5; car['status'] = 'slow'; car['penalty'] = 30
    elif grid_state["transformer_temp"] > 150:
        grid_state["is_blackout"] = True
        return { "min_voltage": 0.0, "total_solar_mw": 0, "grid_battery_percent": 0, "grid_status_text": "EXPLODED", "time_estimate": "Offline", "net_flow_mw": 0, "transformer_temp": int(grid_state["transformer_temp"]), "cars": [], "history": grid_state["history"], "map": get_live_map(net, [], True), "start_ts": grid_state["start_time"] }

    # BATTERY PHYSICS
    # Negative flow = Drain
    net_flow_mw = total_solar_mw - total_ev_load_mw
    
    # Update Server State
    grid_state["battery_soc_kvarh"] += (net_flow_mw * 1000 * 0.5) 
    grid_state["battery_soc_kvarh"] = max(0, min(grid_state["battery_soc_kvarh"], grid_state["max_battery_kvarh"]))
    batt_percent = int((grid_state["battery_soc_kvarh"] / grid_state["max_battery_kvarh"]) * 100)
    
    time_est = "Stable"
    if net_flow_mw < 0:
        drain_rate = abs(net_flow_mw * 1000)
        if drain_rate > 0: time_est = f"{int(grid_state['battery_soc_kvarh']/drain_rate)}h Left"
    elif net_flow_mw > 0 and batt_percent < 100:
        charge_rate = net_flow_mw * 1000
        time_est = f"Full in {int((grid_state['max_battery_kvarh']-grid_state['battery_soc_kvarh'])/charge_rate)}h"

    # TIME CALC
    for car in car_data:
        if car['rate'] > 0.1:
            h = int((car['capacity']*(100-car['soc'])/100)/car['rate'])
            car['time_to_full'] = f"{h}h"
        else: car['time_to_full'] = "Paused"

    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    grid_state["history"].insert(0, {"time": timestamp, "status": status_msg, "volts": round(min_voltage,3)})
    if len(grid_state["history"]) > 5: grid_state["history"].pop()

    return {
        "min_voltage": round(min_voltage, 3), 
        "total_solar_mw": round(total_solar_mw, 2),
        "grid_battery_percent": batt_percent,
        "grid_status_text": status_msg,
        "time_estimate": time_est, 
        "net_flow_mw": float(round(net_flow_mw, 3)), 
        "start_ts": grid_state["start_time"], 
        "congestion_count": sum(1 for c in car_data if c['status']!='charging'),
        "transformer_temp": int(grid_state["transformer_temp"]),
        "cars": car_data,
        "history": grid_state["history"],
        "map": get_live_map(net, car_data, False)
    }

@app.route('/')
def home(): return render_template('index.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        data = request.json
        return jsonify(run_simulation(int(data.get('num_cars')), float(data.get('solar')), data.get('strategy')))
    except Exception as e: return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
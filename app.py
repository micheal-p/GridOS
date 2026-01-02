import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import pandapower.plotting.plotly as pplotly
import json
import plotly
import plotly.graph_objs as go
import datetime
import time
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --- FLEET SPECS ---
EV_MODELS = [
    {"name": "Tesla Model 3", "capacity": 60, "max_rate": 11},      
    {"name": "Tesla Model S", "capacity": 100, "max_rate": 17},     
    {"name": "Cybertruck", "capacity": 123, "max_rate": 22}   
]

# --- PHYSICS STATE (PERSISTENT) ---
grid_state = {
    "battery_kwh": 8000.0,      # Real Energy
    "max_kwh": 10000.0, 
    "transformer_temp": 45.0,   # Celsius
    "ambient_temp": 30.0,
    "is_blackout": False, 
    "last_update": None,        # For dT calculations
    "last_update": None,        # For dT calculations
    "history": [],
    "fleet": []                 # Persistent Fleet State
}
# Standard IEEE 33-Bus Radial Topology for Visualization
# Main Feeder: 0 -> 1 -> ... -> 17
# Lateral 1 (at Bus 1): 1 -> 18 -> 19 -> 20 -> 21
# Lateral 2 (at Bus 2): 2 -> 22 -> 23 -> 24
# Lateral 3 (at Bus 5): 5 -> 25 -> ... -> 32
IEEE_33_COORDS = {
    0: (0, 0),     # Source
    1: (1, 0),     2: (2, 0),     3: (3, 0),     4: (4, 0),
    5: (5, 0),     6: (6, 0),     7: (7, 0),     8: (8, 0),     9: (9, 0),
    10: (10, 0),   11: (11, 0),   12: (12, 0),   13: (13, 0),   14: (14, 0),
    15: (15, 0),   16: (16, 0),   17: (17, 0),  
    
    # Lateral at Bus 1 (Nodes 19-22 locally named as 18-21 in 0-index often, check pandapower)
    # Pandapower case33bw indices are 0-32.
    # Connections (from standard diagram):
    # 1-18, 18-19, 19-20, 20-21
    18: (1, 1),    19: (1, 2),    20: (1, 3),    21: (1, 4),
    
    # Lateral at Bus 2 (Nodes 23-25) -> 22-24
    # 2-22, 22-23, 23-24
    22: (2, -1),   23: (2, -2),   24: (2, -3),
    
    # Lateral at Bus 5 (Nodes 26-33) -> 25-32
    # 5-25, 25-26, 26-27...
    # Also sub-lateral at 25? Standard IEEE 33 is complex.
    # Simplified visual layout for "Lateral 3"
    25: (5, 1),    26: (5, 2),    27: (5, 3),    28: (5, 4),    29: (5, 5),
    30: (6, 5),    31: (7, 5),    32: (8, 5)     # Wrapping around
}

def get_live_map(net, car_data, is_blackout):
    if not hasattr(net, 'bus_geodata') or net.bus_geodata is None or len(net.bus_geodata) == 0:
        net.bus_geodata = pd.DataFrame(index=net.bus.index, columns=["x", "y"])
        
    for bus_idx, (x, y) in IEEE_33_COORDS.items():
        if bus_idx in net.bus_geodata.index:
            net.bus_geodata.loc[bus_idx, "x"] = x
            net.bus_geodata.loc[bus_idx, "y"] = y

    edge_x = []
    edge_y = []
    line_color = '#bbb'
    if grid_state["transformer_temp"] > 90:
        line_color = '#FF9500' # Orange Hot
    if grid_state["transformer_temp"] > 150:
        line_color = '#FF3B30' # Red Hot
    if is_blackout:
        line_color = '#333' # Dark Grey (Dead)

    for _, line in net.line.iterrows():
        try:
            x0 = IEEE_33_COORDS[line.from_bus][0]
            y0 = IEEE_33_COORDS[line.from_bus][1]
            x1 = IEEE_33_COORDS[line.to_bus][0]
            y1 = IEEE_33_COORDS[line.to_bus][1]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        except KeyError:
            continue 

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=3, color=line_color),
        hoverinfo='none', mode='lines'
    )

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    bus_counts = {i: 0 for i in net.bus.index}
    for car in car_data:
        if car['bus_loc'] in bus_counts:
            bus_counts[car['bus_loc']] += 1

    for i in net.bus.index:
        if i in IEEE_33_COORDS:
            node_x.append(IEEE_33_COORDS[i][0])
            node_y.append(IEEE_33_COORDS[i][1])
            
            v_pu = 0.0 if is_blackout else 1.0
            if not is_blackout:
                try:
                    v_pu = net.res_bus.at[i, 'vm_pu']
                except (KeyError, AttributeError):
                    v_pu = 1.0 
            
            count = bus_counts.get(i, 0)
            node_size.append(15 + (count * 4))
            node_text.append(f"<b>Bus {i}</b><br>Voltage: {v_pu:.4f} p.u.<br>EVs: {count}")
            
            if is_blackout:
                node_color.append('#000')
            elif v_pu < 0.90:
                node_color.append('#FF0000') # Critical Low
            elif v_pu < 0.95:
                node_color.append('#FF9500') # Low
            elif v_pu > 1.05:
                node_color.append('#0000FF') # High
            else:
                node_color.append('#34C759') # Good

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
        marker=dict(
            showscale=not is_blackout,
            colorscale='RdYlGn',
            reversescale=False,
            color=node_color,
            size=node_size,
            colorbar=dict(
                thickness=15,
                title=dict(text='Voltage (p.u.)', side='right'),
                xanchor='left'
            ),
            line=dict(width=2, color='#fff')
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            autosize=True,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=20,r=20,t=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def run_simulation(num_cars, solar_intensity, strategy):
    net = pn.case33bw()
    current_time = time.time()
    
    # Calculate Delta Time (dt) since last simulation step
    if grid_state["last_update"] is None:
        dt = 1.0 
    else:
        dt = current_time - grid_state["last_update"]
        if dt > 10:
            dt = 1.0 
    
    grid_state["last_update"] = current_time

    # 1. RESET Logic / Cooling
    if grid_state["is_blackout"]:
        # Natural Cooling
        grid_state["transformer_temp"] += (grid_state["ambient_temp"] - grid_state["transformer_temp"]) * 0.1
        if grid_state["transformer_temp"] < 60:
            grid_state["is_blackout"] = False
        else:
            return { 
                "min_voltage": 0.0, "total_solar_mw": 0, "grid_battery_percent": 0, 
                "grid_status_text": "SYSTEM FAILURE", "time_estimate": "Offline", "net_flow_mw": 0, 
                "transformer_temp": int(grid_state["transformer_temp"]), "cars": [], 
                "history": grid_state["history"], "map": get_live_map(net, [], True),
                "losses_mw": 0, "line_loading_percent": 0
            }

    # 2. MANAGE FLEET (Persistent State)
    # If number of cars changed significantly, regenerate. Otherwise, update physics.
    # Note: For simple UI slider adjustments, we'll authoritative reset if count mismatches.
    
    target_num_cars = max(1, int(num_cars))
    current_fleet = grid_state["fleet"]
    
    if len(current_fleet) != target_num_cars:
        # REGENERATE FLEET
        load_buses = net.load.bus.values
        if len(load_buses) == 0:
            load_buses = net.bus.index.values
            
        active_buses = np.random.choice(load_buses, target_num_cars, replace=True)
        new_fleet = []
        
        for bus in active_buses:
            model = np.random.choice(EV_MODELS)
            new_fleet.append({
                "model": model['name'], 
                "capacity": model['capacity'], 
                "soc": float(np.random.randint(20, 80)), # Float for smooth updates
                "max_rate": model['max_rate'],
                "rate": 0.0,
                "requested_rate": model['max_rate'], 
                "status": "charging", 
                "time_to_full": "...", 
                "bus_loc": int(bus)
            })
        grid_state["fleet"] = new_fleet
    else:
        # UPDATE PHYSICS (Charge Cars)
        # TIME ACCELERATION: 300x Real-time (1 sec real = 5 min sim)
        # This allows users to actually see charging happen in a reasonable demo time.
        TIME_ACCELERATION = 300.0 
        hours_step = (dt * TIME_ACCELERATION) / 3600.0
        
        for car in current_fleet:
            if car['soc'] < 100 and car['status'] != 'stopped':
                energy_added_kwh = car['rate'] * hours_step
                percent_added = (energy_added_kwh / car['capacity']) * 100.0
                car['soc'] += percent_added
                
                if car['soc'] >= 100:
                    car['soc'] = 100.0
                    car['status'] = "done"
                    car['rate'] = 0.0

    # Local reference for logic
    car_data = grid_state["fleet"]

    # 3. GENERATE SUPPLY (Solar)
    total_solar_mw = 3.0 * (solar_intensity / 100.0)
    for bus in [18, 22, 25, 32]:
        pp.create_sgen(net, bus, p_mw=total_solar_mw/4, q_mvar=0)

    # 4. INITIAL ENERGY BALANCE 
    total_requested_mw = sum([c['requested_rate'] for c in car_data]) / 1000.0
    battery_discharge_cap_mw = 10.0 if grid_state["battery_kwh"] > 10 else 0.0
    available_mw = total_solar_mw + battery_discharge_cap_mw
    
    # 5. SMART STRATEGY (Optimization / Droop Control)
    # Applied BEFORE Power Flow (Logic Simulation) or iteratively.
    # Here we simulate an iterative approach by applying limits based on 'Grid Stress' assumptions
    # or by running a baseline PF and then correcting.
    # For efficiency, we will run logic based on available capacity first (Brownout prevention)
    
    status_msg = "Stable"
    # Global Capacity Check (Brownout)
    global_ratio = 1.0
    if total_requested_mw > available_mw:
        if grid_state["battery_kwh"] <= 10:
             # Solar Only Limit
             if total_solar_mw <= 0.1:
                 global_ratio = 0
                 status_msg = "GRID COLLAPSE (No Power)"
             else:
                 global_ratio = total_solar_mw / total_requested_mw
                 status_msg = "BROWNOUT (Power Limited)"
    
    for car in car_data:
        # Reset requested rate if not fully charged
        if car['soc'] < 100:
            car['requested_rate'] = car['max_rate']
            car['status'] = 'charging'
            
        car['rate'] = car['requested_rate'] * global_ratio
        if global_ratio < 0.05 and car['soc'] < 100:
            car['status'] = 'stopped'
            
    # Apply Loads to Net
    net.load.drop(net.load.index, inplace=True) 
    for car in car_data:
        if car['rate'] > 0:
            pp.create_load(net, bus=car['bus_loc'], p_mw=car['rate']/1000.0, q_mvar=0)

    # 6. RUN POWER FLOW (Physics)
    try:
        pp.runpp(net, numba=False)
        converged = True
    except pp.LoadflowNotConverged:
        converged = False
        status_msg = "UNSTABLE (Diverged)"
    except Exception as e:
        print(f"Powerflow error: {e}")
        converged = False
        status_msg = "ERROR (Solver)"

    # 7. POST-FLOW SMART CORRECTION (Droop Control)
    # If strategy is smart, we check voltages and throttle SPECIFIC cars to fix local issues
    # This simulates a local smart inverter controller (Volt-Watt)
    if strategy == "smart" and converged:
        intervention_needed = False
        new_status = status_msg
        
        # Check every car's bus voltage
        for car in car_data:
            bus_idx = car['bus_loc']
            try:
                vm_pu = net.res_bus.at[bus_idx, 'vm_pu']
                # DROOP CURVE:
                # V > 0.95: 100% Rate
                # 0.90 < V < 0.95: Linear reduction
                # V < 0.90: 0% Rate (Cutoff)
                
                throttle_factor = 1.0
                if vm_pu < 0.90:
                    throttle_factor = 0.0
                elif vm_pu < 0.95:
                    # Linear interpolation: at 0.95 -> 1.0, at 0.90 -> 0.0
                    throttle_factor = (vm_pu - 0.90) / 0.05
                    
                if throttle_factor < 1.0 and car['soc'] < 100:
                    intervention_needed = True
                    car['rate'] *= throttle_factor
                    car['status'] = f"smart-curtailed {int(throttle_factor*100)}%"
                    
            except:
                pass
        
        if intervention_needed:
            new_status = "Smart Voltage Optimization"
            # Re-run Power Flow with new rates?
            # Strictly speaking we should, to get final metrics.
            # Reset loads
            net.load.drop(net.load.index, inplace=True) 
            for car in car_data:
                if car['rate'] > 0.001:
                    pp.create_load(net, bus=car['bus_loc'], p_mw=car['rate']/1000.0, q_mvar=0)
            try:
                pp.runpp(net, numba=False)
                status_msg = new_status
            except:
                converged = False # Optimization failed

    # 8. METRICS & THERMODYNAMICS
    actual_delivered_mw = sum([c['rate'] for c in car_data]) / 1000.0
    
    losses_mw = 0.0
    max_line_loading = 0.0
    min_voltage = 0.0
    vdi = 0.0 # Voltage Deviation Index
    
    if converged:
        # Calculate Engineering Metrics
        losses_mw = net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum()
        max_line_loading = max(net.res_line.loading_percent.max(), net.res_trafo.loading_percent.max())
        min_voltage = net.res_bus.vm_pu.min()
        
        # Calculate VDI (Sum of squared deviations from 1.0)
        # Often defined as Root Mean Square deviation or Accumulative deviation
        # Here we use Sum of Deviations for simplicity in visualization trend
        vdi = sum(abs(1.0 - net.res_bus.vm_pu)) 
        
        power_factor = (actual_delivered_mw / 2.5) ** 2
    else:
        power_factor = 20.0
        losses_mw = 0
        min_voltage = 0.0
        max_line_loading = 999.0
    
    # Thermal Integration
    cooling_factor = (grid_state["transformer_temp"] - grid_state["ambient_temp"]) * 0.05
    dT = (power_factor * 2.0) - cooling_factor
    grid_state["transformer_temp"] += dT
    
    # Safety
    if grid_state["transformer_temp"] > 200:
        grid_state["is_blackout"] = True
        status_msg = "TRANSFORMER EXPLODED (>200°C)"

    if grid_state["is_blackout"] or not converged:
        voltage_display = min_voltage if converged else 0.0
        return { 
            "min_voltage": round(voltage_display, 3), 
            "total_solar_mw": total_solar_mw, 
            "grid_battery_percent": 0.0, 
            "grid_status_text": status_msg, 
            "time_estimate": "Offline", 
            "net_flow_mw": 0, 
            "transformer_temp": int(grid_state["transformer_temp"]), 
            "cars": [], 
            "history": grid_state["history"], 
            "map": get_live_map(net, [], True),
            "losses_mw": 0, "line_loading_percent": 0
        }

    # 9. BATTERY & TIME
    net_flow_mw = total_solar_mw - actual_delivered_mw # Actually delivered (includes losses? no, net flow at PCC usually measures import/export + solar. Let's simplify: Supply - Demand)
    # Correct accounting: Net Flow = Solar - (Load + Losses)
    net_flow_mw = total_solar_mw - (actual_delivered_mw + losses_mw)

    sim_step_hours = 1.0 / 60.0 
    grid_state["battery_kwh"] += (net_flow_mw * 1000 * sim_step_hours)
    grid_state["battery_kwh"] = max(0, min(grid_state["battery_kwh"], grid_state["max_kwh"]))
    batt_percent = (grid_state["battery_kwh"] / grid_state["max_kwh"]) * 100

    time_est = "Stable"
    if batt_percent <= 0.1 and net_flow_mw < 0:
        time_est = "DEPLETED"
    elif net_flow_mw < 0:
        drain_rate = abs(net_flow_mw * 1000)
        if drain_rate > 0:
            time_est = f"{int(grid_state['battery_kwh']/drain_rate)}h Left"
    elif net_flow_mw > 0 and batt_percent < 100:
        charge_rate = net_flow_mw * 1000
        time_est = f"Full in {int((grid_state['max_kwh']-grid_state['battery_kwh'])/charge_rate)}h"

    # 10. CAR ETA
    # 10. CAR ETA & ENERGY DETAILS
    for car in car_data:
        kwh_needed = (100.0 - car['soc']) / 100.0 * car['capacity']
        car['kwh_needed'] = round(kwh_needed, 1) # Send to UI
        
        if car['rate'] > 0.1 and car['soc'] < 100:
            hours_left = kwh_needed / car['rate']
            if hours_left < 1.0:
                 car['time_to_full'] = f"{int(hours_left * 60)} min"
            else:
                 car['time_to_full'] = f"{hours_left:.1f} hrs"
        elif car['soc'] >= 100:
            car['time_to_full'] = "Full"
        else:
            car['time_to_full'] = "Paused"

    # Sort fleets by Bus Location for stable UI
    car_data.sort(key=lambda x: x['bus_loc'])

    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    grid_state["history"].insert(0, {"time": timestamp, "status": status_msg, "volts": round(min_voltage,3), "losses": round(losses_mw, 3)})
    if len(grid_state["history"]) > 5:
        grid_state["history"].pop()

    return {
        "min_voltage": round(min_voltage, 4), 
        "total_solar_mw": round(total_solar_mw, 2),
        "grid_battery_percent": round(batt_percent, 1),
        "grid_status_text": status_msg,
        "time_estimate": time_est, 
        "net_flow_mw": float(round(net_flow_mw, 3)),
        "losses_mw": float(round(losses_mw, 4)),
        "line_loading_percent": float(round(max_line_loading, 1)),
        "vdi": float(round(vdi, 4)),
        "congestion_count": sum(1 for c in car_data if c['status']!='charging'),
        "transformer_temp": int(grid_state["transformer_temp"]),
        "cars": car_data,
        "history": grid_state["history"],
        "map": get_live_map(net, car_data, False)
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        data = request.json
        return jsonify(run_simulation(int(data.get('num_cars')), float(data.get('solar')), data.get('strategy')))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
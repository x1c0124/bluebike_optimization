# -*- coding: utf-8 -*-
"""
Bluebikes Station Capacity Optimization Model

This module provides a complete optimization pipeline for determining optimal
dock capacity expansion/reduction for Bluebikes stations based on predicted demand,
subject to budget constraints.

Author: Operations Research Expert
Date: 2024
"""

import pandas as pd
import numpy as np
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import requests
import json

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_data(path="modified data/full_year_agg_outbound.csv", use_predicted=True):
    """
    Load and preprocess the aggregated Bluebikes data.

    Parameters:
    -----------
    path : str
        Path to the CSV file containing aggregated data
    use_predicted : bool
        If True, use 'predicted' column for demand. If False or column doesn't exist,
        use 'trip_count' as demand.

    Returns:
    --------
    df : pd.DataFrame
        Loaded and preprocessed dataframe
    """
    df = pd.read_csv(path, parse_dates=["date"])

    # Create time column combining date and hour
    df["time"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")

    # If predicted column doesn't exist, use trip_count as demand
    if use_predicted and 'predicted' not in df.columns:
        print("Warning: 'predicted' column not found. Using 'trip_count' as demand.")
        df['predicted'] = df['trip_count']
    elif not use_predicted:
        df['predicted'] = df['trip_count']

    # Ensure predicted values are non-negative
    df['predicted'] = df['predicted'].clip(lower=0)

    print(f"Loaded {len(df)} records from {path}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Stations: {df['start_station_name'].nunique()}")

    return df


def build_demand_df(df, use_inbound=True):
    """
    Build demand dataframe for optimization model with both inbound and outbound demand.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with station, time, and predicted demand
    use_inbound : bool
        If True, build both inbound and outbound demand. If False, only outbound.

    Returns:
    --------
    demand_df : pd.DataFrame
        DataFrame with columns: station, time, demand_out, demand_in
    """
    # Outbound demand (trips starting at station)
    outbound_df = df[["start_station_name", "time", "predicted"]].copy()
    outbound_df.rename(columns={
        "start_station_name": "station",
        "predicted": "demand_out"
    }, inplace=True)

    # Ensure demand is non-negative
    outbound_df['demand_out'] = outbound_df['demand_out'].clip(lower=0)

    if use_inbound and 'end_station_name' in df.columns:
        # Inbound demand (trips ending at station)
        # We need to aggregate trips ending at each station
        df['ended_at'] = pd.to_datetime(df['ended_at'], format='mixed', errors='coerce')
        df['time_end'] = df['ended_at']

        # Filter for target stations only
        allowed_stations = outbound_df['station'].unique()
        inbound_raw = df[df['end_station_name'].isin(allowed_stations)].copy()

        if len(inbound_raw) > 0:
            # Aggregate inbound trips by end station and time
            inbound_agg = (
                inbound_raw.groupby(['end_station_name', 'time_end'])
                .size()
                .reset_index(name='demand_in')
            )
            inbound_agg.rename(columns={'end_station_name': 'station', 'time_end': 'time'}, inplace=True)

            # Merge with outbound to ensure same time periods
            demand_df = pd.merge(
                outbound_df[['station', 'time', 'demand_out']],
                inbound_agg[['station', 'time', 'demand_in']],
                on=['station', 'time'],
                how='left'
            )
            demand_df['demand_in'] = demand_df['demand_in'].fillna(0)
        else:
            # No inbound data, use outbound as proxy (scaled)
            print("  Warning: No inbound data found. Using outbound demand as proxy for inbound.")
            demand_df = outbound_df.copy()
            demand_df['demand_in'] = demand_df['demand_out'] * 0.8  # Assume 80% return rate
    else:
        # No inbound data available, use outbound as proxy
        if use_inbound:
            print("  Warning: end_station_name not found. Using outbound demand as proxy for inbound.")
        demand_df = outbound_df.copy()
        demand_df['demand_in'] = demand_df['demand_out'] * 0.8  # Assume 80% return rate

    # Ensure demands are non-negative
    demand_df['demand_out'] = demand_df['demand_out'].clip(lower=0)
    demand_df['demand_in'] = demand_df['demand_in'].clip(lower=0)

    # Remove any rows with missing values
    demand_df = demand_df.dropna()

    print(f"\nDemand DataFrame created:")
    print(f"  Total records: {len(demand_df)}")
    print(f"  Stations: {demand_df['station'].nunique()}")
    print(f"  Time periods: {demand_df['time'].nunique()}")
    print(f"  Total outbound demand: {demand_df['demand_out'].sum():.2f}")
    print(f"  Total inbound demand: {demand_df['demand_in'].sum():.2f}")
    print(f"  Average outbound per station-hour: {demand_df['demand_out'].mean():.2f}")
    print(f"  Average inbound per station-hour: {demand_df['demand_in'].mean():.2f}")

    return demand_df


def fetch_bluebikes_capacities():
    """
    Fetch real station capacities from Bluebikes GBFS API.

    Returns:
    --------
    capacity_dict : dict
        Dictionary mapping station names to their real capacities
        Format: {"Station Name": capacity}
    """
    url = "https://gbfs.bluebikes.com/gbfs/en/station_information.json"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        capacity_dict = {}
        for station in data["data"]["stations"]:
            name = station["name"]
            capacity = station.get("capacity", None)
            if capacity is not None:
                capacity_dict[name] = capacity

        print(f"✓ Successfully fetched {len(capacity_dict)} station capacities from Bluebikes API")
        return capacity_dict

    except requests.exceptions.RequestException as e:
        print(f"⚠ Warning: Failed to fetch capacities from Bluebikes API: {e}")
        print("  Using default capacities instead")
        return {}
    except (KeyError, json.JSONDecodeError) as e:
        print(f"⚠ Warning: Error parsing API response: {e}")
        print("  Using default capacities instead")
        return {}


def fetch_bluebikes_station_status():
    """
    Fetch real-time station status (current inventory) from Bluebikes GBFS API.

    Returns:
    --------
    status_dict : dict
        Dictionary mapping station names to their current status
        Format: {"Station Name": {"num_bikes_available": X, "num_docks_available": Y}}
    """
    url = "https://gbfs.bluebikes.com/gbfs/en/station_status.json"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Get station information to map station_id to name
        info_url = "https://gbfs.bluebikes.com/gbfs/en/station_information.json"
        info_response = requests.get(info_url, timeout=10)
        info_data = info_response.json()

        # Create mapping from station_id to name
        id_to_name = {}
        for station in info_data["data"]["stations"]:
            id_to_name[station["station_id"]] = station["name"]

        status_dict = {}
        for station in data["data"]["stations"]:
            station_id = station["station_id"]
            name = id_to_name.get(station_id, None)
            if name:
                status_dict[name] = {
                    "num_bikes_available": station.get("num_bikes_available", 0),
                    "num_docks_available": station.get("num_docks_available", 0)
                }

        print(f"✓ Successfully fetched {len(status_dict)} station statuses from Bluebikes API")
        return status_dict

    except requests.exceptions.RequestException as e:
        print(f"⚠ Warning: Failed to fetch station status from Bluebikes API: {e}")
        print("  Using default initial inventory instead")
        return {}
    except (KeyError, json.JSONDecodeError) as e:
        print(f"⚠ Warning: Error parsing API response: {e}")
        print("  Using default initial inventory instead")
        return {}


def build_station_info(df, custom_capacities=None, use_real_capacities=True):
    """
    Build station information table with old capacities.
    Uses real capacities from Bluebikes API if available.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing station names
    custom_capacities : dict, optional
        Dictionary mapping station names to custom capacities.
        These override both API and default values.
        Format: {"Station Name": capacity}
    use_real_capacities : bool
        If True, fetch real capacities from Bluebikes API

    Returns:
    --------
    station_info : pd.DataFrame
        DataFrame with columns: station, old_capacity
    """
    stations = sorted(df["start_station_name"].unique())
    default_capacity = 19

    # Initialize with default capacity
    capacities = {station: default_capacity for station in stations}

    # Fetch real capacities from Bluebikes API
    if use_real_capacities:
        print("\nFetching real station capacities from Bluebikes API...")
        real_caps = fetch_bluebikes_capacities()

        if real_caps:
            # Update capacities with real data
            for station in stations:
                if station in real_caps:
                    capacities[station] = real_caps[station]
                    print(f"  ✓ {station}: {real_caps[station]} docks (from API)")
                else:
                    print(f"  ⚠ {station}: Not found in API, using default {default_capacity}")
        else:
            print("  Using default capacities (API fetch failed)")

    # Override with custom capacities if provided (highest priority)
    if custom_capacities:
        print("\nApplying custom capacity overrides:")
        for station, capacity in custom_capacities.items():
            if station in capacities:
                capacities[station] = capacity
                print(f"  → {station}: {capacity} docks (custom override)")
            else:
                print(f"  ⚠ Warning: Station '{station}' not found in data")

    station_info = pd.DataFrame({
        "station": stations,
        "old_capacity": [capacities[s] for s in stations]
    })

    print(f"\nStation Info created:")
    print(f"  Total stations: {len(station_info)}")
    print(f"  Total old capacity: {station_info['old_capacity'].sum()}")
    print(f"  Average capacity per station: {station_info['old_capacity'].mean():.2f}")
    print("\nFinal station capacities:")
    for _, row in station_info.iterrows():
        print(f"  {row['station']}: {row['old_capacity']} docks")

    return station_info


def build_cost_table(df, custom_costs=None, electrified_stations=None):
    """
    Build cost table for dock addition and removal.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing station names
    custom_costs : dict, optional
        Dictionary with custom costs. Format:
        {
            "cost_per_dock_add": 3947,
            "cost_per_dock_remove": -200
        }
    electrified_stations : list, optional
        List of station names that are electrified (30% cost premium)

    Returns:
    --------
    cost_table : pd.DataFrame
        DataFrame with columns: station, cost_per_dock_add, cost_per_dock_remove
    """
    stations = sorted(df["start_station_name"].unique())

    # Default costs
    # Standard: $75,000 per 19 docks = ~$3947 per dock
    # Removal: -$200 savings per dock
    base_cost_add = custom_costs.get("cost_per_dock_add", 3947) if custom_costs else 3947
    base_cost_remove = custom_costs.get("cost_per_dock_remove", -200) if custom_costs else -200

    # Initialize costs
    cost_add = {}
    cost_remove = {}

    for station in stations:
        # Check if station is electrified (30% premium)
        if electrified_stations and station in electrified_stations:
            cost_add[station] = base_cost_add * 1.3
            print(f"  Electrified station: {station} (+30% cost)")
        else:
            cost_add[station] = base_cost_add

        cost_remove[station] = base_cost_remove

    cost_table = pd.DataFrame({
        "station": stations,
        "cost_per_dock_add": [cost_add[s] for s in stations],
        "cost_per_dock_remove": [cost_remove[s] for s in stations]
    })

    print(f"\nCost Table created:")
    print(f"  Base cost per dock add: ${base_cost_add:,.2f}")
    print(f"  Cost per dock remove: ${base_cost_remove:,.2f}")
    print(f"  Electrified stations: {len(electrified_stations) if electrified_stations else 0}")

    return cost_table


def build_model(demand_df, station_info, cost_table, budget=200000, initial_inventory_dict=None, initial_inventory_ratio=0.5, lambda_penalty=0.001):
    """
    Build Pyomo optimization model for dynamic bike inventory optimization.

    Model Formulation (Dynamic Inventory Model):
    ---------------------------------------------
    Decision Variables:
        - add[s]: Number of docks to add at station s (non-negative integer)
        - remove[s]: Number of docks to remove at station s (non-negative integer)
        - capacity[s]: New capacity at station s = old_capacity + add[s] - remove[s]
        - inv[s,t]: Inventory (bikes available) at station s at time t (non-negative)
        - x_out[s,t]: Satisfied outbound demand at station s at time t (non-negative)
        - x_in[s,t]: Satisfied inbound demand at station s at time t (non-negative)

    Objective:
        Maximize: Σ_s,t x_out[s,t] - λ * total_cost
        Where λ (lambda_penalty) balances performance vs. cost
        This represents successful bike rentals minus cost penalty, balancing performance and budget.

    Constraints:
        1. Dynamic inventory: inv[s,t+1] = inv[s,t] + x_in[s,t] - x_out[s,t] for all s, t
        2. Inventory bounds: 0 ≤ inv[s,t] ≤ capacity[s] for all s, t
        3. Outbound constraint: x_out[s,t] ≤ demand_out[s,t] for all s, t
        4. Inbound constraint: x_in[s,t] ≤ demand_in[s,t] for all s, t
        5. Outbound limited by inventory: x_out[s,t] ≤ inv[s,t] for all s, t
        6. Inbound limited by space: x_in[s,t] ≤ capacity[s] - inv[s,t] for all s, t
        7. Initial inventory: inv[s,0] = initial_inventory_ratio * old_capacity[s]
        8. Budget: Σ_s (cost_add[s]·add[s] + cost_remove[s]·remove[s]) ≤ BUDGET
        9. Capacity definition: capacity[s] = old_capacity[s] + add[s] - remove[s]

    Key Improvements:
        - Capacity now represents parking spaces (not throughput)
        - Inventory dynamics reflect real bike sharing operations
        - Dock changes affect maximum inventory, not direct demand satisfaction

    Parameters:
    -----------
    demand_df : pd.DataFrame
        DataFrame with columns: station, time, demand_out, demand_in
    station_info : pd.DataFrame
        DataFrame with columns: station, old_capacity
    cost_table : pd.DataFrame
        DataFrame with columns: station, cost_per_dock_add, cost_per_dock_remove
    budget : float
        Total budget available for capacity changes
    initial_inventory_ratio : float
        Initial inventory as fraction of old capacity (default 0.5 = 50%)
    lambda_penalty : float
        Cost penalty coefficient in objective function (default 0.001)
        Larger values → more cost-conscious, fewer docks added

    Returns:
    --------
    model : pyo.ConcreteModel
        Pyomo optimization model
    """
    print(f"\n{'='*80}")
    print("Building Pyomo Optimization Model")
    print(f"{'='*80}")

    # Create model
    model = pyo.ConcreteModel()

    # Sets
    stations = list(station_info['station'].unique())
    times = sorted(list(demand_df['time'].unique()))  # Sort times for proper ordering

    # Create demand dictionaries for faster lookup
    demand_out_dict = {}
    demand_in_dict = {}
    for _, row in demand_df.iterrows():
        demand_out_dict[(row['station'], row['time'])] = row.get('demand_out', 0.0)
        demand_in_dict[(row['station'], row['time'])] = row.get('demand_in', 0.0)

    # Create cost dictionaries
    cost_add_dict = dict(zip(cost_table['station'], cost_table['cost_per_dock_add']))
    cost_remove_dict = dict(zip(cost_table['station'], cost_table['cost_per_dock_remove']))

    # Create old capacity dictionary
    old_capacity_dict = dict(zip(station_info['station'], station_info['old_capacity']))

    # Sets
    model.S = pyo.Set(initialize=stations, doc="Set of stations")
    model.T = pyo.Set(initialize=times, doc="Set of time periods")

    # Create ordered time set for inventory dynamics
    model.T_ordered = pyo.Set(initialize=times, ordered=True, doc="Ordered set of time periods")

    # Parameters
    model.budget = pyo.Param(initialize=budget, doc="Total budget")

    # Initial inventory: use real data if provided, otherwise use ratio
    if initial_inventory_dict:
        def initial_inventory_init(model, s):
            """Initialize with real inventory from API"""
            return initial_inventory_dict.get(s, initial_inventory_ratio * old_capacity_dict.get(s, 19))
        model.initial_inventory = pyo.Param(model.S, initialize=initial_inventory_init,
                                           doc="Initial inventory (real data from API)")
        print(f"  Using real initial inventory from Bluebikes API")
    else:
        def initial_inventory_init(model, s):
            """Initialize with ratio of capacity"""
            return initial_inventory_ratio * old_capacity_dict.get(s, 19)
        model.initial_inventory = pyo.Param(model.S, initialize=initial_inventory_init,
                                           doc="Initial inventory (ratio of capacity)")
        print(f"  Using initial inventory ratio: {initial_inventory_ratio:.1%}")

    def demand_out_init(model, s, t):
        """Initialize outbound demand parameter"""
        return demand_out_dict.get((s, t), 0.0)

    model.demand_out = pyo.Param(model.S, model.T, initialize=demand_out_init,
                                 doc="Outbound demand at station s at time t")

    def demand_in_init(model, s, t):
        """Initialize inbound demand parameter"""
        return demand_in_dict.get((s, t), 0.0)

    model.demand_in = pyo.Param(model.S, model.T, initialize=demand_in_init,
                                doc="Inbound demand at station s at time t")

    def old_capacity_init(model, s):
        """Initialize old capacity parameter"""
        return old_capacity_dict.get(s, 19)

    model.old_capacity = pyo.Param(model.S, initialize=old_capacity_init,
                                    doc="Old capacity at station s")

    def cost_add_init(model, s):
        """Initialize cost to add dock"""
        return cost_add_dict.get(s, 3947)

    model.cost_add = pyo.Param(model.S, initialize=cost_add_init,
                              doc="Cost to add one dock at station s")

    def cost_remove_init(model, s):
        """Initialize cost to remove dock (negative = savings)"""
        return cost_remove_dict.get(s, -200)

    model.cost_remove = pyo.Param(model.S, initialize=cost_remove_init,
                                  doc="Cost (savings) to remove one dock at station s")

    # Decision Variables
    model.add = pyo.Var(model.S, domain=pyo.NonNegativeIntegers,
                       doc="Number of docks to add at station s")
    model.remove = pyo.Var(model.S, domain=pyo.NonNegativeIntegers,
                          doc="Number of docks to remove at station s")
    model.capacity = pyo.Var(model.S, domain=pyo.NonNegativeReals,
                            doc="New capacity (parking spaces) at station s")
    model.inv = pyo.Var(model.S, model.T, domain=pyo.NonNegativeReals,
                       doc="Inventory (bikes available) at station s at time t")
    model.x_out = pyo.Var(model.S, model.T, domain=pyo.NonNegativeReals,
                         doc="Satisfied outbound demand at station s at time t")
    model.x_in = pyo.Var(model.S, model.T, domain=pyo.NonNegativeReals,
                        doc="Satisfied inbound demand at station s at time t")

    # Objective: Maximize total successful outbound trips minus cost penalty
    # Formula: maximize (Σ x_out[s,t]) - λ * total_cost
    def objective_rule(model):
        # Total successful outbound trips
        total_trips = sum(model.x_out[s, t] for s in model.S for t in model.T)
        # Total cost: sum of (cost_add * add + cost_remove * remove) for all stations
        total_cost = sum(model.cost_add[s] * model.add[s] + model.cost_remove[s] * model.remove[s]
                        for s in model.S)
        # Objective: maximize trips minus cost penalty
        return total_trips - lambda_penalty * total_cost

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize,
                                    doc="Maximize (total successful outbound trips) - λ * total_cost")

    # Constraints

    # 1. Capacity definition: new capacity = old capacity + add - remove
    def capacity_definition_rule(model, s):
        return model.capacity[s] == model.old_capacity[s] + model.add[s] - model.remove[s]

    model.capacity_definition = pyo.Constraint(model.S, rule=capacity_definition_rule,
                                                doc="New capacity definition")

    # 2. Initial inventory: inv[s,0] = initial_inventory[s] (real data or ratio-based)
    def initial_inventory_rule(model, s):
        t0 = times[0]  # First time period
        return model.inv[s, t0] == model.initial_inventory[s]

    model.initial_inventory_constraint = pyo.Constraint(model.S, rule=initial_inventory_rule,
                                                       doc="Initial inventory constraint")

    # 3. Dynamic inventory: inv[s,t+1] = inv[s,t] + x_in[s,t] - x_out[s,t]
    def inventory_dynamics_rule(model, s, t):
        if t == times[-1]:  # Last time period, no next period
            return pyo.Constraint.Skip
        t_idx = times.index(t)
        t_next = times[t_idx + 1]
        return model.inv[s, t_next] == model.inv[s, t] + model.x_in[s, t] - model.x_out[s, t]

    model.inventory_dynamics = pyo.Constraint(model.S, model.T, rule=inventory_dynamics_rule,
                                              doc="Dynamic inventory balance")

    # 4. Inventory bounds: 0 ≤ inv[s,t] ≤ capacity[s]
    def inventory_lower_bound_rule(model, s, t):
        return model.inv[s, t] >= 0

    model.inventory_lower_bound = pyo.Constraint(model.S, model.T,
                                                rule=inventory_lower_bound_rule,
                                                doc="Inventory non-negative")

    def inventory_upper_bound_rule(model, s, t):
        return model.inv[s, t] <= model.capacity[s]

    model.inventory_upper_bound = pyo.Constraint(model.S, model.T,
                                                 rule=inventory_upper_bound_rule,
                                                 doc="Inventory cannot exceed capacity")

    # 5. Outbound demand constraint: x_out[s,t] ≤ demand_out[s,t]
    def outbound_demand_rule(model, s, t):
        return model.x_out[s, t] <= model.demand_out[s, t]

    model.outbound_demand = pyo.Constraint(model.S, model.T, rule=outbound_demand_rule,
                                          doc="Outbound cannot exceed demand")

    # 6. Inbound demand constraint: x_in[s,t] ≤ demand_in[s,t]
    def inbound_demand_rule(model, s, t):
        return model.x_in[s, t] <= model.demand_in[s, t]

    model.inbound_demand = pyo.Constraint(model.S, model.T, rule=inbound_demand_rule,
                                         doc="Inbound cannot exceed demand")

    # 7. Outbound limited by inventory: x_out[s,t] ≤ inv[s,t]
    # (Can't rent more bikes than available)
    def outbound_inventory_rule(model, s, t):
        return model.x_out[s, t] <= model.inv[s, t]

    model.outbound_inventory = pyo.Constraint(model.S, model.T, rule=outbound_inventory_rule,
                                             doc="Outbound limited by available inventory")

    # 8. Inbound limited by available space: x_in[s,t] ≤ capacity[s] - inv[s,t]
    # (Can't return more bikes than parking spaces available)
    def inbound_space_rule(model, s, t):
        return model.x_in[s, t] <= model.capacity[s] - model.inv[s, t]

    model.inbound_space = pyo.Constraint(model.S, model.T, rule=inbound_space_rule,
                                        doc="Inbound limited by available parking space")

    # 9. Budget constraint: total cost of changes cannot exceed budget
    def budget_constraint_rule(model):
        return (sum(model.cost_add[s] * model.add[s] + model.cost_remove[s] * model.remove[s]
                   for s in model.S) <= model.budget)

    model.budget_constraint = pyo.Constraint(rule=budget_constraint_rule, doc="Budget constraint")

    print(f"Model created (Dynamic Inventory Model):")
    print(f"  Stations: {len(model.S)}")
    print(f"  Time periods: {len(model.T)}")
    print(f"  Decision variables:")
    print(f"    - Dock changes: {len(model.S) * 2}")
    print(f"    - Capacity: {len(model.S)}")
    print(f"    - Inventory: {len(model.S) * len(model.T)}")
    print(f"    - Outbound trips: {len(model.S) * len(model.T)}")
    print(f"    - Inbound trips: {len(model.S) * len(model.T)}")
    print(f"    Total: {len(model.S) * (2 + 1 + 3 * len(model.T))}")
    print(f"  Constraints: {len(model.S) * (1 + 1 + 6 * len(model.T)) + 1}")
    print(f"  Budget: ${budget:,.2f}")
    if initial_inventory_dict:
        print(f"  Initial inventory: Real data from Bluebikes API")
        for s in model.S:
            inv_val = pyo.value(model.initial_inventory[s])
            cap_val = old_capacity_dict.get(s, 19)
            print(f"    {s}: {inv_val:.1f} bikes ({inv_val/cap_val*100:.1f}% of capacity)")
    else:
        print(f"  Initial inventory ratio: {initial_inventory_ratio:.1%}")

    return model


def solve_model(model, solver_name="glpk", verbose=True):
    """
    Solve the Pyomo optimization model.

    Parameters:
    -----------
    model : pyo.ConcreteModel
        Pyomo optimization model
    solver_name : str
        Solver name ('glpk', 'cbc', etc.). If None, will try to find any available solver.
    verbose : bool
        Whether to print solver output

    Returns:
    --------
    result : SolverResults
        Solver results object
    """
    print(f"\n{'='*80}")

    # Try to find available solver
    available_solvers = []
    solver_to_try = [solver_name] if solver_name else ["glpk", "cbc", "cplex", "gurobi"]

    for s in solver_to_try:
        solver = pyo.SolverFactory(s)
        if solver.available():
            available_solvers.append(s)
            if s == solver_name or solver_name is None:
                print(f"Solving Model with {s.upper()} Solver")
                print(f"{'='*80}")
                break
    else:
        # No solver found
        print("ERROR: No solver available!")
        print(f"{'='*80}")
        print("\nTo install a solver, choose one of the following options:")
        print("\n1. Install GLPK (recommended for macOS):")
        print("   brew install glpk")
        print("\n2. Install CBC:")
        print("   brew install cbc")
        print("\n3. Or download GLPK from: https://www.gnu.org/software/glpk/")
        print("\n4. Or use Pyomo's built-in solvers (if available)")
        raise ValueError(f"No solver available. Please install GLPK or CBC.")

    # Use the first available solver
    if available_solvers:
        solver_name = available_solvers[0]
        solver = pyo.SolverFactory(solver_name)

    # Solve model
    if verbose:
        result = solver.solve(model, tee=True)
    else:
        result = solver.solve(model, tee=False)

    # Check solution status
    if result.solver.termination_condition == pyo.TerminationCondition.optimal:
        print(f"\n✓ Optimal solution found!")
        print(f"  Solver status: {result.solver.status}")
        print(f"  Termination condition: {result.solver.termination_condition}")
    elif result.solver.termination_condition == pyo.TerminationCondition.feasible:
        print(f"\n⚠ Feasible solution found (may not be optimal)")
        print(f"  Solver status: {result.solver.status}")
        print(f"  Termination condition: {result.solver.termination_condition}")
    else:
        print(f"\n✗ Solution not found")
        print(f"  Solver status: {result.solver.status}")
        print(f"  Termination condition: {result.solver.termination_condition}")
        print(f"  Message: {result.solver.message if hasattr(result.solver, 'message') else 'N/A'}")

    return result


def summarize_results(model, station_info, demand_df, cost_table):
    """
    Summarize optimization results.

    Parameters:
    -----------
    model : pyo.ConcreteModel
        Solved Pyomo model
    station_info : pd.DataFrame
        Station information with old capacities
    demand_df : pd.DataFrame
        Demand dataframe
    cost_table : pd.DataFrame
        Cost table

    Returns:
    --------
    results_summary : pd.DataFrame
        Summary of results with columns: station, old_capacity, add, remove,
        new_capacity, cost_add, cost_remove, total_cost
    """
    print(f"\n{'='*80}")
    print("Optimization Results Summary")
    print(f"{'='*80}")

    # Extract results
    results = []
    for s in model.S:
        add_val = pyo.value(model.add[s])
        remove_val = pyo.value(model.remove[s])
        capacity_val = pyo.value(model.capacity[s])
        cost_add_val = pyo.value(model.cost_add[s])
        cost_remove_val = pyo.value(model.cost_remove[s])

        total_cost = cost_add_val * add_val + cost_remove_val * remove_val

        results.append({
            'station': s,
            'old_capacity': pyo.value(model.old_capacity[s]),
            'add': add_val,
            'remove': remove_val,
            'new_capacity': capacity_val,
            'cost_per_dock_add': cost_add_val,
            'cost_per_dock_remove': cost_remove_val,
            'total_cost': total_cost
        })

    results_summary = pd.DataFrame(results)

    # Calculate objective value
    objective_value = pyo.value(model.objective)

    # Calculate total cost
    total_cost = results_summary['total_cost'].sum()

    # Calculate demand statistics for dynamic model
    total_outbound_demand = demand_df['demand_out'].sum() if 'demand_out' in demand_df.columns else 0
    total_inbound_demand = demand_df['demand_in'].sum() if 'demand_in' in demand_df.columns else 0

    # Calculate satisfied trips from model
    total_satisfied_outbound = 0
    total_satisfied_inbound = 0
    for s in model.S:
        for t in model.T:
            total_satisfied_outbound += pyo.value(model.x_out[s, t])
            total_satisfied_inbound += pyo.value(model.x_in[s, t])

    # Calculate unmet demand
    unmet_outbound = total_outbound_demand - total_satisfied_outbound
    unmet_inbound = total_inbound_demand - total_satisfied_inbound

    # Calculate average inventory utilization
    avg_inventory = {}
    max_inventory = {}
    for s in model.S:
        inv_values = [pyo.value(model.inv[s, t]) for t in model.T]
        avg_inventory[s] = np.mean(inv_values)
        max_inventory[s] = max(inv_values)

    # Print summary
    # Calculate total cost for display
    total_cost = sum(pyo.value(model.cost_add[s] * model.add[s] + model.cost_remove[s] * model.remove[s])
                    for s in model.S)

    # The objective value includes the cost penalty, so we need to extract the actual trips
    # Objective = total_trips - lambda_penalty * total_cost
    # We can get total_trips from the model variables
    total_trips = sum(pyo.value(model.x_out[s, t]) for s in model.S for t in model.T)

    print(f"\nObjective Value (with cost penalty): {objective_value:,.2f}")
    print(f"Total Successful Outbound Trips: {total_trips:,.2f}")
    print(f"Total Cost: ${total_cost:,.2f}")
    print(f"Budget Used: {total_cost / pyo.value(model.budget) * 100:.2f}%")

    print(f"\nDemand Statistics:")
    print(f"  Total Outbound Demand: {total_outbound_demand:,.2f}")
    print(f"  Satisfied Outbound: {total_satisfied_outbound:,.2f}")
    print(f"  Unmet Outbound: {unmet_outbound:,.2f}")
    print(f"  Outbound Satisfaction Rate: {(total_satisfied_outbound/total_outbound_demand*100) if total_outbound_demand > 0 else 0:.2f}%")

    print(f"\n  Total Inbound Demand: {total_inbound_demand:,.2f}")
    print(f"  Satisfied Inbound: {total_satisfied_inbound:,.2f}")
    print(f"  Unmet Inbound: {unmet_inbound:,.2f}")
    print(f"  Inbound Satisfaction Rate: {(total_satisfied_inbound/total_inbound_demand*100) if total_inbound_demand > 0 else 0:.2f}%")

    print(f"\nCapacity Changes:")
    print(f"  Total docks to add: {results_summary['add'].sum():.0f}")
    print(f"  Total docks to remove: {results_summary['remove'].sum():.0f}")
    print(f"  Net capacity change: {results_summary['new_capacity'].sum() - results_summary['old_capacity'].sum():.0f}")

    print(f"\nAverage Inventory Utilization:")
    for s in model.S:
        cap = pyo.value(model.capacity[s])
        avg_inv = avg_inventory[s]
        util = (avg_inv / cap * 100) if cap > 0 else 0
        print(f"  {s}: {avg_inv:.1f}/{cap:.0f} bikes ({util:.1f}% utilization)")

    print(f"\nStation-Level Results:")
    print(results_summary.to_string(index=False))

    return results_summary, {
        'objective_value': objective_value,
        'total_cost': total_cost,
        'budget_used_pct': total_cost / pyo.value(model.budget) * 100,
        'total_outbound_demand': total_outbound_demand,
        'total_inbound_demand': total_inbound_demand,
        'satisfied_outbound': total_satisfied_outbound,
        'satisfied_inbound': total_satisfied_inbound,
        'unmet_outbound': unmet_outbound,
        'unmet_inbound': unmet_inbound,
        'avg_inventory': avg_inventory,
        'max_inventory': max_inventory
    }


def visualize_changes(results_summary, stats, output_dir="modified data"):
    """
    Create visualizations of optimization results.

    Parameters:
    -----------
    results_summary : pd.DataFrame
        Results summary from summarize_results()
    stats : dict
        Statistics dictionary from summarize_results()
    output_dir : str
        Directory to save plots
    """
    print(f"\n{'='*80}")
    print("Creating Visualizations")
    print(f"{'='*80}")

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Prepare data for plotting
    results_summary = results_summary.sort_values('station')

    # 1. Bar chart: Old vs New Capacity
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results_summary))
    width = 0.35

    bars1 = ax.bar(x - width/2, results_summary['old_capacity'], width,
                   label='Old Capacity', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, results_summary['new_capacity'], width,
                   label='New Capacity', color='#2ecc71', alpha=0.8)

    ax.set_xlabel('Station', fontsize=12, fontweight='bold')
    ax.set_ylabel('Capacity (Docks)', fontsize=12, fontweight='bold')
    ax.set_title('Old vs New Capacity by Station', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_summary['station'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/capacity_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir}/capacity_comparison.png")
    plt.close()

    # 2. Bar chart: Add vs Remove
    fig, ax = plt.subplots(figsize=(12, 6))

    # Only show stations with changes
    changes_df = results_summary[(results_summary['add'] > 0) | (results_summary['remove'] > 0)].copy()

    if len(changes_df) > 0:
        x = np.arange(len(changes_df))
        width = 0.35

        bars1 = ax.bar(x - width/2, changes_df['add'], width,
                       label='Add Docks', color='#27ae60', alpha=0.8)
        bars2 = ax.bar(x + width/2, changes_df['remove'], width,
                       label='Remove Docks', color='#e74c3c', alpha=0.8)

        ax.set_xlabel('Station', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Docks', fontsize=12, fontweight='bold')
        ax.set_title('Dock Additions and Removals by Station', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(changes_df['station'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No capacity changes recommended',
               ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title('Dock Additions and Removals by Station', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/add_remove_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir}/add_remove_comparison.png")
    plt.close()

    # 3. Summary statistics table
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')

    # Build table data based on available stats
    table_data = [
        ['Metric', 'Value'],
        ['Total Satisfied Outbound', f"{stats.get('satisfied_outbound', stats.get('satisfied_after', 0)):,.2f}"],
        ['Total Satisfied Inbound', f"{stats.get('satisfied_inbound', 0):,.2f}"],
        ['Unmet Outbound', f"{stats.get('unmet_outbound', stats.get('unmet_after', 0)):,.2f}"],
        ['Unmet Inbound', f"{stats.get('unmet_inbound', 0):,.2f}"],
        ['Outbound Satisfaction Rate', f"{(stats.get('satisfied_outbound', 0)/stats.get('total_outbound_demand', 1)*100) if stats.get('total_outbound_demand', 0) > 0 else 0:.2f}%"],
        ['Inbound Satisfaction Rate', f"{(stats.get('satisfied_inbound', 0)/stats.get('total_inbound_demand', 1)*100) if stats.get('total_inbound_demand', 0) > 0 else 0:.2f}%"],
        ['Total Cost', f"${stats['total_cost']:,.2f}"],
        ['Budget Used', f"{stats['budget_used_pct']:.2f}%"],
        ['Total Docks Added', f"{results_summary['add'].sum():.0f}"],
        ['Total Docks Removed', f"{results_summary['remove'].sum():.0f}"],
        ['Net Capacity Change', f"{results_summary['new_capacity'].sum() - results_summary['old_capacity'].sum():.0f}"]
    ]

    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                    colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')

    plt.title('Optimization Results Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(f'{output_dir}/results_summary_table.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir}/results_summary_table.png")
    plt.close()

    print(f"\n✓ All visualizations saved to {output_dir}/")


# Main execution
if __name__ == "__main__":
    print("="*80)
    print("Bluebikes Station Capacity Optimization Model")
    print("="*80)

    # Configuration
    DATA_PATH = "modified data/full_year_agg_outbound.csv"
    BUDGET = 200000  # Total budget in dollars
    # Solver name - will auto-detect if None
    # Options: "glpk", "cbc", None (auto-detect)
    SOLVER = None  # Auto-detect available solver

    # Custom station capacities (optional)
    # Set to None to use real capacities from Bluebikes API
    # Or provide custom overrides if needed
    CUSTOM_CAPACITIES = None  # Using real API data instead
    # Example custom override (if needed):
    # CUSTOM_CAPACITIES = {
    #     "MIT at Mass Ave / Amherst St": 31,
    #     "Harvard Square at Mass Ave/ Dunster": 19,
    #     "Central Square at Mass Ave / Essex St": 19
    # }

    # Electrified stations (30% cost premium) - optional
    ELECTRIFIED_STATIONS = None  # Example: ["MIT at Mass Ave / Amherst St"]

    # Custom costs (optional)
    CUSTOM_COSTS = None  # Example: {"cost_per_dock_add": 4000, "cost_per_dock_remove": -200}

    try:
        # Step 1: Load data
        df = load_data(DATA_PATH, use_predicted=True)

        # Step 2: Build demand dataframe (with both inbound and outbound)
        demand_df = build_demand_df(df, use_inbound=True)

        # Step 3: Build station info
        station_info = build_station_info(df, custom_capacities=CUSTOM_CAPACITIES)

        # Step 4: Build cost table
        cost_table = build_cost_table(df, custom_costs=CUSTOM_COSTS,
                                      electrified_stations=ELECTRIFIED_STATIONS)

        # Step 5: Fetch real-time initial inventory from Bluebikes API
        print("\n" + "="*80)
        print("Fetching Real-Time Initial Inventory from Bluebikes API")
        print("="*80)
        station_status = fetch_bluebikes_station_status()

        # Build initial inventory dictionary from API data
        initial_inventory_dict = {}
        if station_status:
            for _, row in station_info.iterrows():
                station_name = row['station']
                if station_name in station_status:
                    num_bikes = station_status[station_name]['num_bikes_available']
                    initial_inventory_dict[station_name] = num_bikes
                    print(f"  ✓ {station_name}: {num_bikes} bikes available (from API)")
                else:
                    # Fallback to ratio if station not found in API
                    fallback_inv = 0.5 * row['old_capacity']
                    initial_inventory_dict[station_name] = fallback_inv
                    print(f"  ⚠ {station_name}: Not found in API, using {fallback_inv:.1f} (50% of capacity)")
        else:
            # API failed, use ratio-based initial inventory
            print("  Using ratio-based initial inventory (API fetch failed)")
            initial_inventory_dict = None

        # Step 6: Build optimization model (Dynamic Inventory Model)
        # Use real inventory data if available, otherwise use 50% ratio
        INITIAL_INVENTORY_RATIO = 0.5  # Fallback ratio
        LAMBDA_PENALTY = 0.2  # Cost penalty coefficient

        print("\n" + "="*80)
        print("Building Pyomo Optimization Model")
        print("="*80)
        model = build_model(demand_df, station_info, cost_table, budget=BUDGET,
                           initial_inventory_dict=initial_inventory_dict,
                           initial_inventory_ratio=INITIAL_INVENTORY_RATIO,
                           lambda_penalty=LAMBDA_PENALTY)

        # Step 7: Solve model
        result = solve_model(model, solver_name=SOLVER, verbose=True)

        # Step 8: Summarize results
        results_summary, stats = summarize_results(model, station_info, demand_df, cost_table)

        # Step 9: Visualize results
        visualize_changes(results_summary, stats, output_dir="modified data")

        # Save results to CSV
        results_summary.to_csv("modified data/optimization_results.csv", index=False)
        print(f"\n✓ Results saved to: modified data/optimization_results.csv")

        print("\n" + "="*80)
        print("Optimization Complete!")
        print("="*80)

    except Exception as e:
        print(f"\n✗ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

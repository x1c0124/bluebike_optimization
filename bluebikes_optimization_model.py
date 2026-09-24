"""
Bluebikes Station Capacity Optimization Model

Decides how many docks to add or remove at each station, using an hourly
inventory model of bikes at each station and a fixed budget.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

DEFAULT_CAPACITY = 19
# Dock counts at the time of the study (from the Bluebikes GBFS station feed)
STATION_CAPACITIES = {
    "MIT at Mass Ave / Amherst St": 31,
    "Harvard Square at Mass Ave/ Dunster": 19,
    "Central Square at Mass Ave / Essex St": 19,
}
COST_PER_DOCK_ADD = 3947      # $75,000 per 19-dock station
COST_PER_DOCK_REMOVE = -200   # negative = savings
ELECTRIFIED_PREMIUM = 1.3
INBOUND_RATIO = 0.8           # inbound demand assumed to be 80% of outbound


def section(title):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def load_data(path="modified data/full_year_agg_outbound.csv", use_predicted=True):
    """
    Load the hourly aggregated data and add a `time` column.

    Demand is taken from the `predicted` column when present and
    `use_predicted` is True; otherwise from `trip_count`.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df["time"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")

    if not (use_predicted and 'predicted' in df.columns):
        if use_predicted:
            print("Note: 'predicted' column not found. Using 'trip_count' as demand.")
        df['predicted'] = df['trip_count']
    df['predicted'] = df['predicted'].clip(lower=0)

    print(f"Loaded {len(df)} records from {path}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Stations: {df['start_station_name'].nunique()}")
    return df


def build_demand_df(df, inbound_ratio=INBOUND_RATIO):
    """
    Build hourly demand per station: columns station, time, demand_out, demand_in.

    The aggregated data only records trips starting at each station, so inbound
    demand (bikes being returned) is approximated as `inbound_ratio` × outbound.
    """
    demand_df = df.rename(columns={"start_station_name": "station", "predicted": "demand_out"})
    demand_df = demand_df[["station", "time", "demand_out"]].dropna()
    demand_df['demand_in'] = demand_df['demand_out'] * inbound_ratio

    print("\nDemand DataFrame created:")
    print(f"  Total records: {len(demand_df)}")
    print(f"  Stations: {demand_df['station'].nunique()}")
    print(f"  Time periods: {demand_df['time'].nunique()}")
    print(f"  Total outbound demand: {demand_df['demand_out'].sum():.2f}")
    print(f"  Total inbound demand: {demand_df['demand_in'].sum():.2f} ({inbound_ratio:.0%} of outbound)")
    return demand_df


def build_station_info(df, custom_capacities=None):
    """
    Current dock count per station: columns station, old_capacity.

    Uses STATION_CAPACITIES, falling back to DEFAULT_CAPACITY; `custom_capacities`
    overrides both.
    """
    stations = sorted(df["start_station_name"].unique())
    capacities = {s: STATION_CAPACITIES.get(s, DEFAULT_CAPACITY) for s in stations}

    for station, capacity in (custom_capacities or {}).items():
        if station in capacities:
            capacities[station] = capacity
        else:
            print(f"  Warning: station '{station}' not found in data")

    station_info = pd.DataFrame({"station": stations, "old_capacity": [capacities[s] for s in stations]})

    print("\nStation capacities:")
    for s, cap in capacities.items():
        print(f"  {s}: {cap} docks")
    return station_info


def build_cost_table(df, custom_costs=None, electrified_stations=None):
    """
    Cost per dock added/removed at each station. Electrified stations cost 30% more to expand.
    """
    custom_costs = custom_costs or {}
    electrified_stations = set(electrified_stations or [])
    cost_add = custom_costs.get("cost_per_dock_add", COST_PER_DOCK_ADD)
    cost_remove = custom_costs.get("cost_per_dock_remove", COST_PER_DOCK_REMOVE)

    stations = sorted(df["start_station_name"].unique())
    cost_table = pd.DataFrame({
        "station": stations,
        "cost_per_dock_add": [cost_add * (ELECTRIFIED_PREMIUM if s in electrified_stations else 1)
                              for s in stations],
        "cost_per_dock_remove": cost_remove,
    })

    print("\nCost table:")
    print(f"  Cost per dock added: ${cost_add:,.2f}")
    print(f"  Cost per dock removed: ${cost_remove:,.2f}")
    print(f"  Electrified stations: {len(electrified_stations)}")
    return cost_table


def build_model(demand_df, station_info, cost_table, budget=200000,
                initial_inventory_ratio=0.5, lambda_penalty=0.001):
    """
    Build the Pyomo dynamic inventory model.

    Decision variables
        add[s], remove[s]   docks added / removed at station s (integer)
        capacity[s]         new dock count = old_capacity + add - remove
        inv[s,t]            bikes at station s at the start of hour t
        x_out[s,t]          bikes rented (satisfied outbound demand)
        x_in[s,t]           bikes returned (satisfied inbound demand)

    Objective
        maximize Σ x_out[s,t] − λ · total_cost

    Constraints
        inv[s,t+1] = inv[s,t] + x_in[s,t] − x_out[s,t]
        inv[s,0]   = initial_inventory_ratio · old_capacity[s]
        inv[s,t] ≤ capacity[s]
        x_out[s,t] ≤ demand_out[s,t],  x_out[s,t] ≤ inv[s,t]
        x_in[s,t]  ≤ demand_in[s,t],   x_in[s,t]  ≤ capacity[s] − inv[s,t]
        Σ (cost_add[s]·add[s] + cost_remove[s]·remove[s]) ≤ budget

    lambda_penalty trades off trips against spending: larger values add fewer docks.
    """
    section("Building Pyomo Optimization Model")

    stations = list(station_info['station'])
    times = sorted(set(demand_df['time']))  # pd.Timestamp, matching the demand dict keys
    next_time = dict(zip(times[:-1], times[1:]))

    demand = demand_df.set_index(['station', 'time'])
    old_capacity = dict(zip(station_info['station'], station_info['old_capacity']))

    m = pyo.ConcreteModel()
    m.S = pyo.Set(initialize=stations)
    m.T = pyo.Set(initialize=times, ordered=True)

    m.budget = pyo.Param(initialize=budget)
    m.old_capacity = pyo.Param(m.S, initialize=old_capacity)
    m.initial_inventory = pyo.Param(m.S, initialize={s: initial_inventory_ratio * old_capacity[s] for s in stations})
    m.demand_out = pyo.Param(m.S, m.T, initialize=demand['demand_out'].to_dict(), default=0.0)
    m.demand_in = pyo.Param(m.S, m.T, initialize=demand['demand_in'].to_dict(), default=0.0)
    m.cost_add = pyo.Param(m.S, initialize=dict(zip(cost_table['station'], cost_table['cost_per_dock_add'])))
    m.cost_remove = pyo.Param(m.S, initialize=dict(zip(cost_table['station'], cost_table['cost_per_dock_remove'])))

    m.add = pyo.Var(m.S, domain=pyo.NonNegativeIntegers)
    m.remove = pyo.Var(m.S, domain=pyo.NonNegativeIntegers)
    m.capacity = pyo.Var(m.S, domain=pyo.NonNegativeReals)
    m.inv = pyo.Var(m.S, m.T, domain=pyo.NonNegativeReals)
    m.x_out = pyo.Var(m.S, m.T, domain=pyo.NonNegativeReals)
    m.x_in = pyo.Var(m.S, m.T, domain=pyo.NonNegativeReals)

    m.total_cost = pyo.Expression(expr=sum(m.cost_add[s] * m.add[s] + m.cost_remove[s] * m.remove[s] for s in m.S))
    m.total_trips = pyo.Expression(expr=sum(m.x_out[s, t] for s in m.S for t in m.T))
    m.objective = pyo.Objective(expr=m.total_trips - lambda_penalty * m.total_cost, sense=pyo.maximize)

    m.capacity_definition = pyo.Constraint(
        m.S, rule=lambda m, s: m.capacity[s] == m.old_capacity[s] + m.add[s] - m.remove[s])
    m.initial_inventory_constraint = pyo.Constraint(
        m.S, rule=lambda m, s: m.inv[s, times[0]] == m.initial_inventory[s])

    def inventory_dynamics_rule(m, s, t):
        if t not in next_time:
            return pyo.Constraint.Skip
        return m.inv[s, next_time[t]] == m.inv[s, t] + m.x_in[s, t] - m.x_out[s, t]
    m.inventory_dynamics = pyo.Constraint(m.S, m.T, rule=inventory_dynamics_rule)

    m.inventory_upper_bound = pyo.Constraint(m.S, m.T, rule=lambda m, s, t: m.inv[s, t] <= m.capacity[s])
    m.outbound_demand = pyo.Constraint(m.S, m.T, rule=lambda m, s, t: m.x_out[s, t] <= m.demand_out[s, t])
    m.inbound_demand = pyo.Constraint(m.S, m.T, rule=lambda m, s, t: m.x_in[s, t] <= m.demand_in[s, t])
    # Can't rent more bikes than are at the station
    m.outbound_inventory = pyo.Constraint(m.S, m.T, rule=lambda m, s, t: m.x_out[s, t] <= m.inv[s, t])
    # Can't return more bikes than there are empty docks
    m.inbound_space = pyo.Constraint(m.S, m.T, rule=lambda m, s, t: m.x_in[s, t] <= m.capacity[s] - m.inv[s, t])
    m.budget_constraint = pyo.Constraint(expr=m.total_cost <= m.budget)

    print(f"  Stations: {len(stations)} | Time periods: {len(times)}")
    print(f"  Variables: {m.nvariables()} | Constraints: {m.nconstraints()}")
    print(f"  Budget: ${budget:,.2f} | Initial inventory: {initial_inventory_ratio:.0%} of capacity"
          f" | λ = {lambda_penalty}")
    return m


def solve_model(model, solver_name=None, verbose=True):
    """Solve with `solver_name`, or the first available of GLPK, CBC, CPLEX, Gurobi."""
    candidates = [solver_name] if solver_name else ["glpk", "cbc", "cplex", "gurobi"]
    solver_name = next((s for s in candidates if pyo.SolverFactory(s).available(exception_flag=False)), None)
    if solver_name is None:
        raise RuntimeError(f"No solver available (tried {', '.join(candidates)}). "
                           "Install one with `brew install glpk`; see INSTALL.md.")

    section(f"Solving Model with {solver_name.upper()}")
    result = pyo.SolverFactory(solver_name).solve(model, tee=verbose)

    condition = result.solver.termination_condition
    if condition == pyo.TerminationCondition.optimal:
        print("\n✓ Optimal solution found")
    elif condition == pyo.TerminationCondition.feasible:
        print("\n⚠ Feasible solution found (may not be optimal)")
    else:
        print(f"\n✗ No solution found: {result.solver.status}, {condition}")
    return result


def summarize_results(model, demand_df):
    """Return per-station dock changes and a dict of overall statistics."""
    section("Optimization Results Summary")
    m = model

    results_summary = pd.DataFrame([{
        'station': s,
        'old_capacity': pyo.value(m.old_capacity[s]),
        'add': pyo.value(m.add[s]),
        'remove': pyo.value(m.remove[s]),
        'new_capacity': pyo.value(m.capacity[s]),
        'cost_per_dock_add': pyo.value(m.cost_add[s]),
        'cost_per_dock_remove': pyo.value(m.cost_remove[s]),
        'total_cost': pyo.value(m.cost_add[s] * m.add[s] + m.cost_remove[s] * m.remove[s]),
    } for s in m.S])

    total_cost = pyo.value(m.total_cost)
    stats = {
        'objective_value': pyo.value(m.objective),
        'total_cost': total_cost,
        'budget_used_pct': total_cost / pyo.value(m.budget) * 100,
        'total_outbound_demand': demand_df['demand_out'].sum(),
        'total_inbound_demand': demand_df['demand_in'].sum(),
        'satisfied_outbound': pyo.value(m.total_trips),
        'satisfied_inbound': sum(pyo.value(m.x_in[s, t]) for s in m.S for t in m.T),
        'avg_inventory': {s: np.mean([pyo.value(m.inv[s, t]) for t in m.T]) for s in m.S},
    }
    stats['unmet_outbound'] = stats['total_outbound_demand'] - stats['satisfied_outbound']
    stats['unmet_inbound'] = stats['total_inbound_demand'] - stats['satisfied_inbound']
    for direction in ['outbound', 'inbound']:
        total = stats[f'total_{direction}_demand']
        stats[f'{direction}_rate'] = stats[f'satisfied_{direction}'] / total * 100 if total > 0 else 0

    print(f"\nObjective value (with cost penalty): {stats['objective_value']:,.2f}")
    print(f"Total cost: ${total_cost:,.2f} ({stats['budget_used_pct']:.2f}% of budget)")
    for direction in ['outbound', 'inbound']:
        print(f"\n{direction.capitalize()} demand: {stats[f'total_{direction}_demand']:,.2f}")
        print(f"  Satisfied: {stats[f'satisfied_{direction}']:,.2f} ({stats[f'{direction}_rate']:.2f}%)")
        print(f"  Unmet: {stats[f'unmet_{direction}']:,.2f}")

    print("\nAverage inventory:")
    for s, avg_inv in stats['avg_inventory'].items():
        cap = pyo.value(m.capacity[s])
        print(f"  {s}: {avg_inv:.1f}/{cap:.0f} bikes ({avg_inv / cap * 100 if cap else 0:.1f}%)")

    print("\nStation-level results:")
    print(results_summary.to_string(index=False))
    return results_summary, stats


def _bar_chart(ax, labels, series, title, ylabel, hide_zero=False):
    """Grouped bar chart with value labels. `series` is a list of (label, values, color)."""
    x = np.arange(len(labels))
    width = 0.35
    for i, (label, values, color) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color, alpha=0.8)
        for bar in bars:
            height = bar.get_height()
            if height > 0 or not hide_zero:
                ax.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}',
                        ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('Station', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def visualize_changes(results_summary, stats, output_dir="modified data"):
    """Save the capacity comparison, dock change and summary table charts."""
    section("Creating Visualizations")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results_summary = results_summary.sort_values('station')

    # 1. Old vs new capacity
    fig, ax = plt.subplots(figsize=(12, 6))
    _bar_chart(ax, results_summary['station'],
               [('Old Capacity', results_summary['old_capacity'], '#3498db'),
                ('New Capacity', results_summary['new_capacity'], '#2ecc71')],
               'Old vs New Capacity by Station', 'Capacity (Docks)')
    _save(fig, f'{output_dir}/capacity_comparison.png')

    # 2. Docks added / removed, only for stations that change
    fig, ax = plt.subplots(figsize=(12, 6))
    changes = results_summary[(results_summary['add'] > 0) | (results_summary['remove'] > 0)]
    if len(changes):
        _bar_chart(ax, changes['station'],
                   [('Add Docks', changes['add'], '#27ae60'),
                    ('Remove Docks', changes['remove'], '#e74c3c')],
                   'Dock Additions and Removals by Station', 'Number of Docks', hide_zero=True)
    else:
        ax.text(0.5, 0.5, 'No capacity changes recommended',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title('Dock Additions and Removals by Station', fontsize=14, fontweight='bold')
    _save(fig, f'{output_dir}/add_remove_comparison.png')

    # 3. Summary statistics table
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    table_data = [
        ['Metric', 'Value'],
        ['Total Satisfied Outbound', f"{stats['satisfied_outbound']:,.2f}"],
        ['Total Satisfied Inbound', f"{stats['satisfied_inbound']:,.2f}"],
        ['Unmet Outbound', f"{stats['unmet_outbound']:,.2f}"],
        ['Unmet Inbound', f"{stats['unmet_inbound']:,.2f}"],
        ['Outbound Satisfaction Rate', f"{stats['outbound_rate']:.2f}%"],
        ['Inbound Satisfaction Rate', f"{stats['inbound_rate']:.2f}%"],
        ['Total Cost', f"${stats['total_cost']:,.2f}"],
        ['Budget Used', f"{stats['budget_used_pct']:.2f}%"],
        ['Total Docks Added', f"{results_summary['add'].sum():.0f}"],
        ['Total Docks Removed', f"{results_summary['remove'].sum():.0f}"],
        ['Net Capacity Change',
         f"{results_summary['new_capacity'].sum() - results_summary['old_capacity'].sum():.0f}"],
    ]
    table = ax.table(cellText=table_data, cellLoc='left', loc='center', colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#34495e')
            cell.set_text_props(weight='bold', color='white')
        elif row % 2 == 0:
            cell.set_facecolor('#ecf0f1')
    ax.set_title('Optimization Results Summary', fontsize=14, fontweight='bold', pad=20)
    _save(fig, f'{output_dir}/results_summary_table.png')


if __name__ == "__main__":
    section("Bluebikes Station Capacity Optimization Model")

    DATA_PATH = "modified data/full_year_agg_outbound.csv"
    OUTPUT_DIR = "modified data"
    BUDGET = 200000
    SOLVER = None                   # None = first available of glpk, cbc, cplex, gurobi
    INITIAL_INVENTORY_RATIO = 0.5   # stations start half full
    LAMBDA_PENALTY = 0.2            # cost penalty in the objective
    CUSTOM_CAPACITIES = None        # e.g. {"MIT at Mass Ave / Amherst St": 35}
    ELECTRIFIED_STATIONS = None     # e.g. ["MIT at Mass Ave / Amherst St"]
    CUSTOM_COSTS = None             # e.g. {"cost_per_dock_add": 4000, "cost_per_dock_remove": -200}

    df = load_data(DATA_PATH)
    demand_df = build_demand_df(df)
    station_info = build_station_info(df, custom_capacities=CUSTOM_CAPACITIES)
    cost_table = build_cost_table(df, custom_costs=CUSTOM_COSTS, electrified_stations=ELECTRIFIED_STATIONS)

    model = build_model(demand_df, station_info, cost_table, budget=BUDGET,
                        initial_inventory_ratio=INITIAL_INVENTORY_RATIO, lambda_penalty=LAMBDA_PENALTY)
    solve_model(model, solver_name=SOLVER, verbose=False)

    results_summary, stats = summarize_results(model, demand_df)
    visualize_changes(results_summary, stats, output_dir=OUTPUT_DIR)
    results_summary.to_csv(f"{OUTPUT_DIR}/optimization_results.csv", index=False)
    print(f"\n✓ Results saved to: {OUTPUT_DIR}/optimization_results.csv")
    section("Optimization Complete!")

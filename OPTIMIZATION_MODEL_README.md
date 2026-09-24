# Bluebikes Station Capacity Optimization Model

## Overview

This module provides a complete optimization pipeline for determining optimal dock capacity expansion/reduction for Bluebikes stations based on predicted demand, subject to budget constraints.

## Installation Requirements

```bash
pip install pandas numpy pyomo matplotlib seaborn
```

For the solver (choose one):
- **GLPK**: `brew install glpk` (macOS) or download from https://www.gnu.org/software/glpk/
- **CBC**: Included with Pyomo or install separately

## Quick Start

```python
from bluebikes_optimization_model import *

# Load data
df = load_data("modified data/full_year_agg_outbound.csv")

# Build components
demand_df = build_demand_df(df)
station_info = build_station_info(df, custom_capacities={
    "MIT at Mass Ave / Amherst St": 25,
    "Harvard Square at Mass Ave/ Dunster": 30,
    "Central Square at Mass Ave / Essex St": 23
})
cost_table = build_cost_table(df)

# Build and solve model
model = build_model(demand_df, station_info, cost_table, budget=200000)
result = solve_model(model, solver_name="glpk")

# Get results
results_summary, stats = summarize_results(model, station_info, demand_df, cost_table)
visualize_changes(results_summary, stats)
```

## Run Complete Pipeline

Simply run:
```bash
python bluebikes_optimization_model.py
```

## Model Formulation

**Decision Variables:**
- `add[s]`: Number of docks to add at station s
- `remove[s]`: Number of docks to remove at station s
- `capacity[s]`: New capacity at station s
- `x[s,t]`: Satisfied demand at station s at time t

**Objective:**
Maximize total satisfied demand: Σ_s,t x[s,t]

**Constraints:**
1. Capacity: x[s,t] ≤ capacity[s]
2. Demand: x[s,t] ≤ demand[s,t]
3. Budget: Σ_s (cost_add[s]·add[s] + cost_remove[s]·remove[s]) ≤ BUDGET
4. Capacity definition: capacity[s] = old_capacity[s] + add[s] - remove[s]

## Output Files

- `optimization_results.csv`: Detailed results per station
- `capacity_comparison.png`: Old vs new capacity visualization
- `add_remove_comparison.png`: Dock additions/removals visualization
- `results_summary_table.png`: Summary statistics table

## Customization

### Custom Station Capacities
```python
custom_capacities = {
    "Station Name 1": 25,
    "Station Name 2": 30
}
station_info = build_station_info(df, custom_capacities=custom_capacities)
```

### Electrified Stations (30% cost premium)
```python
electrified_stations = ["MIT at Mass Ave / Amherst St"]
cost_table = build_cost_table(df, electrified_stations=electrified_stations)
```

### Custom Costs
```python
custom_costs = {
    "cost_per_dock_add": 4000,
    "cost_per_dock_remove": -200
}
cost_table = build_cost_table(df, custom_costs=custom_costs)
```

## Default Parameters

- **Default capacity per station**: 19 docks
- **Cost per dock add**: $3,947 (based on $75,000 per 19 docks)
- **Cost per dock remove**: -$200 (savings)
- **Default budget**: $200,000
- **Default solver**: GLPK

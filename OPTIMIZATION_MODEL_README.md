# Bluebikes Station Capacity Optimization Model

`bluebikes_optimization_model.py` decides how many docks to add or remove at each station. It simulates the number of bikes at every station hour by hour over the year and works within a fixed budget.

## Installation

```bash
pip install pandas numpy pyomo matplotlib seaborn
brew install glpk
```

See [INSTALL.md](INSTALL.md) for other solver options.

## Run

```bash
python bluebikes_optimization_model.py
```

The settings are at the bottom of the file under `if __name__ == "__main__":`. They cover the budget, solver, initial inventory ratio, cost penalty λ, custom capacities, electrified stations and custom costs.

## Use as a module

```python
from bluebikes_optimization_model import *

df = load_data("modified data/full_year_agg_outbound.csv")
demand_df = build_demand_df(df)
station_info = build_station_info(df)
cost_table = build_cost_table(df)

model = build_model(demand_df, station_info, cost_table,
                    budget=200000, initial_inventory_ratio=0.5, lambda_penalty=0.2)
solve_model(model, solver_name="glpk")

results_summary, stats = summarize_results(model, demand_df)
visualize_changes(results_summary, stats)
```

## Model formulation

**Decision variables**
- `add[s]`, `remove[s]`: docks added or removed at station s (integer)
- `capacity[s]`: new dock count
- `inv[s,t]`: bikes at station s at the start of hour t
- `x_out[s,t]`: bikes rented (satisfied outbound demand)
- `x_in[s,t]`: bikes returned (satisfied inbound demand)

**Objective**

Maximize Σ x_out[s,t] − λ · total_cost

**Constraints**
1. Inventory balance: inv[s,t+1] = inv[s,t] + x_in[s,t] − x_out[s,t]
2. Initial inventory: inv[s,0] = initial_inventory_ratio · old_capacity[s]
3. Inventory ≤ capacity: inv[s,t] ≤ capacity[s]
4. Rentals: x_out[s,t] ≤ demand_out[s,t] and x_out[s,t] ≤ inv[s,t]
5. Returns: x_in[s,t] ≤ demand_in[s,t] and x_in[s,t] ≤ capacity[s] − inv[s,t]
6. Capacity: capacity[s] = old_capacity[s] + add[s] − remove[s]
7. Budget: Σ (cost_add[s]·add[s] + cost_remove[s]·remove[s]) ≤ budget

## Inputs and assumptions

| Setting | Default |
|---|---|
| Demand | `predicted` column if present, otherwise the historical `trip_count` |
| Inbound demand | 80% of outbound (the data only records trip starts) |
| Current docks | MIT 31, Harvard Square 19, Central Square 19 (`STATION_CAPACITIES`) |
| Initial inventory | 50% of current docks |
| Cost to add a dock | $3,947 ($75,000 per 19 docks); 30% more at electrified stations |
| Cost to remove a dock | −$200 (savings) |
| Budget | $200,000 |
| Cost penalty λ | 0.2 |

## Customization

```python
# Override dock counts
station_info = build_station_info(df, custom_capacities={"MIT at Mass Ave / Amherst St": 35})

# Electrified stations (30% higher cost to add docks)
cost_table = build_cost_table(df, electrified_stations=["MIT at Mass Ave / Amherst St"])

# Custom costs
cost_table = build_cost_table(df, custom_costs={"cost_per_dock_add": 4000, "cost_per_dock_remove": -200})
```

## Output files (in `modified data/`)

- `optimization_results.csv`: dock changes and cost per station
- `capacity_comparison.png`: old vs new capacity
- `add_remove_comparison.png`: docks added and removed
- `results_summary_table.png`: summary statistics

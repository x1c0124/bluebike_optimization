# Installation

## 1. Python packages

```bash
pip install pandas numpy scikit-learn xgboost statsmodels pyomo matplotlib seaborn tqdm
```

## 2. LP solver

The optimization model uses Pyomo, which needs an external linear programming solver. Install one of the following.

**GLPK (recommended)**

```bash
brew install glpk                     # macOS with Homebrew
conda install -c conda-forge glpk     # or with Conda
```

You can also download it from https://www.gnu.org/software/glpk/ and add it to your `PATH`.

**CBC (alternative)**

```bash
brew install cbc
```

## 3. Check the solver

```bash
glpsol --version
```

Or from Python:

```python
import pyomo.environ as pyo
print(pyo.SolverFactory('glpk').available())   # should print True
```

## 4. Run

```bash
python analyze_full_year_data.py          # demand prediction
python bluebikes_optimization_model.py    # capacity optimization
```

The optimization script loads the data, builds and solves the model, and saves the results and charts to `modified data/`.

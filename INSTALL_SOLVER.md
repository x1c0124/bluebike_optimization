# 安装求解器 (Solver Installation Guide)

## 问题
运行优化模型需要安装一个线性规划求解器。目前系统没有找到可用的求解器。

## 解决方案

### 选项 1: 安装 Homebrew 然后安装 GLPK (推荐 macOS)

1. **安装 Homebrew** (如果还没有):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **安装 GLPK**:
   ```bash
   brew install glpk
   ```

3. **验证安装**:
   ```bash
   which glpsol
   glpsol --version
   ```

### 选项 2: 手动下载 GLPK

1. 访问: https://www.gnu.org/software/glpk/
2. 下载适合 macOS 的版本
3. 解压并添加到 PATH

### 选项 3: 使用 Conda (如果已安装)

```bash
conda install -c conda-forge glpk
```

### 选项 4: 使用 CBC

```bash
brew install cbc
```

## 验证安装

安装后，运行以下命令验证:

```python
import pyomo.environ as pyo
solver = pyo.SolverFactory('glpk')
if solver.available():
    print("✓ GLPK solver is available!")
else:
    print("✗ GLPK solver not found")
```

## 运行优化模型

安装求解器后，运行:

```bash
python bluebikes_optimization_model.py
```

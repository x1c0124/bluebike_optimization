# 快速安装指南

## 问题
优化模型需要 GLPK 求解器，但系统尚未安装。

## 解决方案（3个步骤）

### 步骤 1: 安装 Homebrew
在终端中运行以下命令（需要输入密码）：
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 步骤 2: 安装 GLPK
安装 Homebrew 后，运行：
```bash
brew install glpk
```

### 步骤 3: 运行优化模型
```bash
cd bluebike_optimization
python3 bluebikes_optimization_model.py
```

## 验证安装
安装后，验证 GLPK 是否可用：
```bash
which glpsol
glpsol --version
```

## 如果不想安装 Homebrew

### 选项 A: 使用 Conda
```bash
conda install -c conda-forge glpk
```

### 选项 B: 手动下载 GLPK
1. 访问: https://www.gnu.org/software/glpk/
2. 下载 macOS 版本
3. 解压并添加到 PATH

## 安装完成后
运行脚本将自动：
- 加载数据
- 构建优化模型
- 求解优化问题
- 生成结果和可视化图表

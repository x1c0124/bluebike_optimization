# -*- coding: utf-8 -*-
"""
分析全年数据，参考 bluebikes_top3_opt.py 的处理方式
使用已有的天气数据和事件类型特征

依赖库:
- xgboost: 用于XGBoost模型
- statsmodels: 用于OLS回归详细统计
"""

import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        return iterable
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# XGBoost - 必须安装
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
# statsmodels - 必须安装，用于OLS回归详细统计
import statsmodels.api as sm

# Path to data folder
DATA_DIR = '/Users/charlotte/Downloads/project/modified data'

print("=" * 80)
print("Step 1: 加载和探索数据")
print("=" * 80)

file_path = f'{DATA_DIR}/merged_all_months.csv'
df = pd.read_csv(file_path)

print(f"数据形状: {df.shape}")

print("\n" + "=" * 80)
print("Step 2: 聚合出行数据（出站/起始站点）")
print("=" * 80)

# 解析时间戳
df['started_at'] = pd.to_datetime(df['started_at'], format='mixed', errors='coerce')
df['date'] = df['started_at'].dt.date

# 过滤3个目标站点
allowed = [
    "MIT at Mass Ave / Amherst St",
    "Harvard Square at Mass Ave/ Dunster",
    "Central Square at Mass Ave / Essex St"
]
df_f = df[df['start_station_name'].isin(allowed)].copy()

print(f"过滤后的数据形状: {df_f.shape}")
print(f"站点列表: {df_f['start_station_name'].unique()}")

# 添加时间相关特征
df_f['day_of_week'] = df_f['started_at'].dt.day_name()
df_f['is_weekend'] = df_f['day_of_week'].isin(['Saturday', 'Sunday'])
df_f['week_of_year'] = df_f['started_at'].dt.isocalendar().week
df_f['month'] = df_f['started_at'].dt.month_name()

# 按站点、日期、小时等分组统计出行次数
# 同时保留天气和事件特征（取平均值，因为同一时间点的值应该相同）
outbound = (
    df_f.groupby(['start_station_name', 'date', 'hour',
                  'day_of_week', 'is_weekend', 'week_of_year', 'month'])
        .agg({
            'ride_id': 'count',  # 出行次数
            'PRCP': 'mean',      # 天气特征取平均值
            'SNOW': 'mean',
            'TAVG': 'mean',
            'extreme_weather': 'mean',
            'event_Basketball': 'max',  # 事件特征取最大值（0或1）
            'event_Comedy': 'max',
            'event_Concert': 'max',
            'event_Fair': 'max',
            'event_Hockey': 'max'
        })
        .reset_index()
        .rename(columns={'ride_id': 'trip_count'})
)

print(f"\n聚合后的数据形状: {outbound.shape}")
print(f"\n前20行:")
print(outbound.head(20))

# === 1. 构建完整网格 ===
stations = outbound['start_station_name'].unique()
dates = outbound['date'].unique()
hours = sorted(outbound['hour'].unique())

print(f"\n站点数: {len(stations)}")
print(f"日期数: {len(dates)}")
print(f"小时数: {len(hours)}")
print(f"完整网格大小: {len(stations) * len(dates) * len(hours)}")

full_grid = pd.MultiIndex.from_product(
    [stations, dates, hours],
    names=['start_station_name', 'date', 'hour']
).to_frame(index=False)

# === 2. 合并出行次数 ===
merged = pd.merge(
    full_grid,
    outbound[['start_station_name', 'date', 'hour', 'trip_count',
              'PRCP', 'SNOW', 'TAVG', 'extreme_weather',
              'event_Basketball', 'event_Comedy', 'event_Concert', 'event_Fair', 'event_Hockey']],
    on=['start_station_name', 'date', 'hour'],
    how='left'
)

# 填充缺失值
merged['trip_count'] = merged['trip_count'].fillna(0)
# 天气特征：对于缺失值，使用前一个有效值填充，如果都没有则用0
merged['PRCP'] = merged['PRCP'].ffill().fillna(0)
merged['SNOW'] = merged['SNOW'].ffill().fillna(0)
merged['TAVG'] = merged['TAVG'].ffill().fillna(0)
merged['extreme_weather'] = merged['extreme_weather'].fillna(0)
# 事件特征：缺失值填充为0
for col in ['event_Basketball', 'event_Comedy', 'event_Concert', 'event_Fair', 'event_Hockey']:
    merged[col] = merged[col].fillna(0)

# === 3. 重新计算所有时间特征 ===
merged['date'] = pd.to_datetime(merged['date'])
merged['day_of_week'] = merged['date'].dt.day_name()
merged['is_weekend'] = merged['day_of_week'].isin(['Saturday', 'Sunday'])
merged['week_of_year'] = merged['date'].dt.isocalendar().week
merged['month'] = merged['date'].dt.month_name()

print(f"\n合并后的数据形状: {merged.shape}")
print(f"\n前10行:")
print(merged.head(10))

# 保存聚合数据
output_file = f'{DATA_DIR}/full_year_agg_outbound.csv'
merged.to_csv(output_file, index=False)
print(f"\n已保存聚合数据到: {output_file}")

print("\n" + "=" * 80)
print("Step 3: One-hot编码分类变量")
print("=" * 80)

# One-hot编码
complete_df = pd.get_dummies(
    merged,
    columns=['start_station_name', 'day_of_week', 'week_of_year', 'month', 'hour'],
    prefix=['station', 'day', 'week', 'month', 'hour'],
    drop_first=False,   # 保留所有虚拟变量
    dtype=int
)

print(f"编码后的数据形状: {complete_df.shape}")
print(f"\n列数: {len(complete_df.columns)}")
print(f"\n前10行:")
print(complete_df.head(10))

# 用于存储所有模型结果的字典
all_model_results = []

print("\n" + "=" * 80)
print("Step 4: 模型训练和评估")
print("=" * 80)

# 准备特征和目标变量
df_model = complete_df.copy()
y = df_model['trip_count']
X = df_model.drop(columns=['trip_count', 'date', 'is_weekend'], errors='ignore')

# 删除月份特征（避免与周的多重共线性）
X = X.drop(columns=[c for c in X.columns if c.startswith('month_')], errors='ignore')

print(f"特征数: {X.shape[1]}")
print(f"样本数: {X.shape[0]}")

# 删除基线虚拟变量（避免多重共线性）
baseline_dummies = [
    'day_Monday',                                    # day_of_week的基线
    'month_January',                                 # month的基线（如果存在）
    'station_MIT at Mass Ave / Amherst St',          # station的基线
    'week_1',                                        # week_of_year的基线
    'hour_0'                                         # hour的基线
]

X = X.drop(columns=baseline_dummies, errors='ignore')

print(f"删除基线变量后的特征数: {X.shape[1]}")

# 训练测试集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

print("\n" + "-" * 80)
print("4.1 线性回归")
print("-" * 80)

# 训练线性回归
lr = LinearRegression()
lr.fit(X_train, y_train)

# 预测
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

# 计算指标
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("线性回归性能:")
print(f"训练集 RMSE: {train_rmse:.3f} | 训练集 R²: {train_r2:.3f}")
print(f"测试集 RMSE: {test_rmse:.3f} | 测试集 R²: {test_r2:.3f}")

# 查看系数
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr.coef_
}).sort_values('Coefficient', ascending=False)

print("\n前10个正系数特征:")
print(coef_df.head(10))
print("\n前10个负系数特征:")
print(coef_df.tail(10))

# 存储线性回归结果
all_model_results.append({
    'Model': 'Linear Regression',
    'Train_R2': train_r2,
    'Test_R2': test_r2,
    'OSR2': test_r2,  # Out-of-sample R²
    'Train_RMSE': train_rmse,
    'Test_RMSE': test_rmse,
    'Best_Params': 'None',
    'Significant_Features': 'N/A (no p-values from sklearn)',
    'Top_Features': ', '.join(coef_df.head(10)['Feature'].tolist())
})

# OLS模型（带统计信息）
print("\n使用statsmodels进行OLS回归:")
ols_model = sm.OLS(y_train, X_train).fit()
y_train_pred_ols = ols_model.predict(X_train)
y_test_pred_ols = ols_model.predict(X_test)

train_rmse_ols = np.sqrt(mean_squared_error(y_train, y_train_pred_ols))
test_rmse_ols = np.sqrt(mean_squared_error(y_test, y_test_pred_ols))
train_r2_ols = r2_score(y_train, y_train_pred_ols)
test_r2_ols = r2_score(y_test, y_test_pred_ols)

print(f"训练集 RMSE: {train_rmse_ols:.3f}")
print(f"训练集 R²: {train_r2_ols:.3f}")
print(f"测试集 RMSE: {test_rmse_ols:.3f}")
print(f"测试集 R²: {test_r2_ols:.3f}")

# 打印详细的统计信息
print("\nOLS回归详细统计:")
print(ols_model.summary())

# 提取显著特征（p < 0.05）
ols_pvalues = ols_model.pvalues
significant_features = ols_pvalues[ols_pvalues < 0.05].sort_values()
significant_features_str = ', '.join([f"{feat} (p={pval:.4f})"
                                      for feat, pval in significant_features.items()])

# 存储OLS结果
all_model_results.append({
    'Model': 'OLS Regression',
    'Train_R2': train_r2_ols,
    'Test_R2': test_r2_ols,
    'OSR2': test_r2_ols,
    'Train_RMSE': train_rmse_ols,
    'Test_RMSE': test_rmse_ols,
    'Best_Params': 'None',
    'Significant_Features': significant_features_str if len(significant_features) > 0 else 'None',
    'Top_Features': ', '.join(significant_features.head(10).index.tolist()) if len(significant_features) > 0 else 'N/A'
})

print("\n" + "-" * 80)
print("4.2 Ridge 和 Lasso 回归")
print("-" * 80)

# CV设置
kf = KFold(n_splits=5, shuffle=True, random_state=42)
r2_scorer = make_scorer(r2_score)

# 模型参数网格
param_grid = {
    'Ridge': {'model': Ridge(), 'params': {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 5, 10]}},
    'Lasso': {'model': Lasso(max_iter=5000), 'params': {'alpha': [0.0001, 0.001, 0.01, 0.1]}},
}

# 交叉验证 → 最佳参数 → 训练/测试评估
for name, entry in tqdm(param_grid.items(), desc="正则化模型"):
    print(f"\n===== {name} =====")
    base_model = entry['model']
    params = entry['params']
    param_results = []

    best_cv_r2 = -np.inf
    best_param = None
    best_model = None

    # 步骤1: 对所有参数进行交叉验证
    for param_name, param_values in params.items():
        for val in param_values:
            model = base_model.set_params(**{param_name: val})
            cv_r2 = cross_val_score(model, X_train, y_train, cv=kf, scoring=r2_scorer, n_jobs=-1).mean()
            param_results.append([f"{param_name}={val}", cv_r2])
            if cv_r2 > best_cv_r2:
                best_cv_r2 = cv_r2
                best_param = {param_name: val}
                best_model = model

    # 步骤2: 显示所有参数的CV R²
    param_df = pd.DataFrame(param_results, columns=['Parameter', 'CV_R2'])
    print(param_df.round(3))
    print(f"\n{name}最佳参数: {best_param} (CV_R²={best_cv_r2:.3f})")

    # 步骤3: 在全部训练数据上重新训练最佳模型
    best_model.fit(X_train, y_train)
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print(f"训练集 R²: {train_r2:.3f} | 测试集 R²: {test_r2:.3f}")

    # 存储Ridge/Lasso结果
    all_model_results.append({
        'Model': name,
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'OSR2': test_r2,
        'Train_RMSE': train_rmse,
        'Test_RMSE': test_rmse,
        'Best_Params': str(best_param),
        'Significant_Features': 'N/A (regularized model)',
        'Top_Features': 'N/A (regularized model)'
    })

print("\n" + "-" * 80)
print("4.3 其他模型 (RandomForest, XGBoost, GBM)")
print("-" * 80)

# 模型参数网格
param_grid_other = {
    'RandomForest': {'model': RandomForestRegressor(random_state=42, n_jobs=-1),
                     'params': {'n_estimators': [50, 100, 150]}},
    'GBM': {'model': GradientBoostingRegressor(random_state=42, n_estimators=100),
            'params': {'learning_rate': [0.05, 0.1]}}
}

# 添加XGBoost到模型列表
param_grid_other['XGBoost'] = {'model': xgb.XGBRegressor(random_state=42, n_jobs=-1, n_estimators=100),
                               'params': {'max_depth': [3, 5, 7]}}

# CV → 最佳参数 → 训练/测试评估
for name, entry in tqdm(param_grid_other.items(), desc="其他模型"):
    print(f"\n\n===== {name} =====")
    base_model = entry['model']
    params = entry['params']
    param_results = []

    best_cv_r2 = -np.inf
    best_param = None
    best_model = None

    # 步骤1: 对所有参数进行交叉验证
    for param_name, param_values in params.items():
        for val in param_values:
            model = base_model.set_params(**{param_name: val})
            cv_r2 = cross_val_score(model, X_train, y_train, cv=kf, scoring=r2_scorer, n_jobs=-1).mean()
            param_results.append([f"{param_name}={val}", cv_r2])
            if cv_r2 > best_cv_r2:
                best_cv_r2 = cv_r2
                best_param = {param_name: val}
                best_model = model

    # 步骤2: 显示所有参数的CV R²
    param_df = pd.DataFrame(param_results, columns=['Parameter', 'CV_R2'])
    print(param_df.round(3))
    print(f"\n{name}最佳参数: {best_param} (CV_R²={best_cv_r2:.3f})")

    # 步骤3: 在全部训练数据上重新训练最佳模型
    best_model.fit(X_train, y_train)
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print(f"训练集 R²: {train_r2:.3f} | 测试集 R²: {test_r2:.3f}")

    # 提取特征重要性（对于树模型）
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        top_features = ', '.join(feature_importance.head(10)['Feature'].tolist())
    else:
        top_features = 'N/A'

    # 存储树模型结果
    all_model_results.append({
        'Model': name,
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'OSR2': test_r2,
        'Train_RMSE': train_rmse,
        'Test_RMSE': test_rmse,
        'Best_Params': str(best_param),
        'Significant_Features': 'N/A (tree model - use feature importance)',
        'Top_Features': top_features
    })

# XGBoost 网格搜索
print("\n" + "-" * 80)
print("4.4 XGBoost 网格搜索")
print("-" * 80)

param_grid_xgb = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.5],
    'n_estimators': [100, 150, 200, 300]
}

# 生成所有参数组合
param_combinations = list(itertools.product(
    param_grid_xgb['max_depth'],
    param_grid_xgb['learning_rate'],
    param_grid_xgb['n_estimators']
))

results = []

# 网格搜索循环
for (depth, lr, n_est) in tqdm(param_combinations, desc="XGBoost网格搜索"):
    params = {'max_depth': depth, 'learning_rate': lr, 'n_estimators': n_est}
    model = xgb.XGBRegressor(random_state=42, n_jobs=-1, **params)

    # 交叉验证 R²
    cv_r2 = cross_val_score(model, X_train, y_train, cv=kf, scoring=r2_scorer, n_jobs=1).mean()

    # 在全部训练集上训练
    model.fit(X_train, y_train)
    y_train_pred = np.maximum(model.predict(X_train), 0)  # 确保非负
    y_test_pred = np.maximum(model.predict(X_test), 0)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    results.append({
        'max_depth': depth,
        'learning_rate': lr,
        'n_estimators': n_est,
        'Train_R2': train_r2,
        'CV_R2': cv_r2,
        'Test_R2': test_r2
    })

# 编译结果
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('CV_R2', ascending=False).reset_index(drop=True)

print("\n=== XGBoost 网格搜索结果 ===")
print(results_df.round(5))
print("\n=== Full results_df (detailed) ===")
print(results_df)

# 识别最佳配置
best_params = results_df.iloc[0][['max_depth', 'learning_rate', 'n_estimators']].to_dict()
print(f"\n最佳参数: {best_params}")

# 使用最佳参数训练最终模型并存储结果
best_xgb_final = xgb.XGBRegressor(
    random_state=42,
    n_jobs=-1,
    max_depth=int(best_params['max_depth']),
    learning_rate=float(best_params['learning_rate']),
    n_estimators=int(best_params['n_estimators'])
)
best_xgb_final.fit(X_train, y_train)
y_train_pred_xgb = np.maximum(best_xgb_final.predict(X_train), 0)
y_test_pred_xgb = np.maximum(best_xgb_final.predict(X_test), 0)

train_r2_xgb = r2_score(y_train, y_train_pred_xgb)
test_r2_xgb = r2_score(y_test, y_test_pred_xgb)
train_rmse_xgb = np.sqrt(mean_squared_error(y_train, y_train_pred_xgb))
test_rmse_xgb = np.sqrt(mean_squared_error(y_test, y_test_pred_xgb))

# 提取XGBoost特征重要性
xgb_feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_xgb_final.feature_importances_
}).sort_values('Importance', ascending=False)
xgb_top_features = ', '.join(xgb_feature_importance.head(10)['Feature'].tolist())

# 存储XGBoost网格搜索最佳结果
all_model_results.append({
    'Model': 'XGBoost (Grid Search Best)',
    'Train_R2': train_r2_xgb,
    'Test_R2': test_r2_xgb,
    'OSR2': test_r2_xgb,
    'Train_RMSE': train_rmse_xgb,
    'Test_RMSE': test_rmse_xgb,
    'Best_Params': str(best_params),
    'Significant_Features': 'N/A (tree model - use feature importance)',
    'Top_Features': xgb_top_features
})

print("\n" + "=" * 80)
print("Step 5: 可视化预测结果")
print("=" * 80)

import matplotlib.dates as mdates

# 使用最佳XGBoost模型参数重新训练
best_xgb_params = results_df.iloc[0][['max_depth', 'learning_rate', 'n_estimators']].to_dict()
# 确保参数类型正确（n_estimators和max_depth必须是整数）
best_xgb_params['max_depth'] = int(best_xgb_params['max_depth'])
best_xgb_params['n_estimators'] = int(best_xgb_params['n_estimators'])
best_xgb_params['learning_rate'] = float(best_xgb_params['learning_rate'])
best_model = xgb.XGBRegressor(
    random_state=42,
    n_jobs=-1,
    **best_xgb_params
)
best_model.fit(X, y)

# 在整个数据集上预测
y_pred_full = np.maximum(best_model.predict(X), 0)

# 合并原始数据
plot_df = complete_df.copy()
plot_df['predicted'] = y_pred_full
plot_df['actual'] = plot_df['trip_count']

# 恢复站点名称（从one-hot列）
station_cols = [c for c in plot_df.columns if c.startswith('station_')]
plot_df['station'] = plot_df[station_cols].idxmax(axis=1).str.replace('station_', '')

# 按日期排序并平滑
plot_df = plot_df.sort_values('date')
plot_df['actual_smooth'] = plot_df['actual'].rolling(24, center=True).mean()
plot_df['predicted_smooth'] = plot_df['predicted'].rolling(24, center=True).mean()

# 绘图设置
stations = plot_df['station'].unique()
n = len(stations)
plt.figure(figsize=(14, 3.5 * n))

for i, station in enumerate(stations, 1):
    subset = plot_df[plot_df['station'] == station]
    plt.subplot(n, 1, i)

    try:
        # 尝试使用新版本的seaborn参数
        sns.lineplot(
            data=subset, x='date', y='actual_smooth',
            color='tab:blue', label='Actual', linewidth=1.5, errorbar=None
        )
        sns.lineplot(
            data=subset, x='date', y='predicted_smooth',
            color='tab:orange', label='Predicted', linewidth=1.5, errorbar=None, alpha=0.7
        )
    except TypeError:
        # 如果errorbar参数不支持，使用旧版本的方式
        sns.lineplot(
            data=subset, x='date', y='actual_smooth',
            color='tab:blue', label='Actual', linewidth=1.5, ci=None
        )
        sns.lineplot(
            data=subset, x='date', y='predicted_smooth',
            color='tab:orange', label='Predicted', linewidth=1.5, ci=None, alpha=0.7
        )

    plt.title(station)
    plt.ylabel("Trip Count")
    plt.xlabel(None)
    plt.legend()

    # 月度刻度，旋转标签
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.xticks(rotation=45, ha='right')

    # 移除背景网格线
    ax.grid(False)

plt.suptitle("Actual vs Predicted Trip Counts by Station (Full Year Data, Time Smoothed)",
             fontsize=14, y=0.95)
plt.xlabel("Time (by Month)")
plt.tight_layout(rect=[0, 0, 1, 0.94])

# 保存图片
output_plot = f'{DATA_DIR}/full_year_prediction_plot.png'
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"Saved visualization to: {output_plot}")
plt.close()

# 计算残差
plot_df['residual'] = plot_df['actual'] - plot_df['predicted']

# 显示残差最大的行
cols_to_show = [c for c in ['date', 'hour', 'station', 'actual', 'predicted', 'residual']
                if c in plot_df.columns]

largest_positive = plot_df.sort_values('residual', ascending=False).head(20)
largest_negative = plot_df.sort_values('residual', ascending=True).head(20)

print("\n=== Top 20 Underpredicted Cases (Actual >> Predicted) ===")
print(largest_positive[cols_to_show])

print("\n=== Top 20 Overpredicted Cases (Predicted >> Actual) ===")
print(largest_negative[cols_to_show])

print("\n" + "=" * 80)
print("Step 6: 生成模型结果总结")
print("=" * 80)

# 创建结果DataFrame
results_summary_df = pd.DataFrame(all_model_results)

# 按Test_R2 (OSR2)排序
results_summary_df = results_summary_df.sort_values('OSR2', ascending=False).reset_index(drop=True)

print("\n所有模型结果总结（按OSR2排序）:")
print("=" * 80)
print(results_summary_df.to_string(index=False))

# 保存到CSV
summary_csv_path = f'{DATA_DIR}/model_results_summary.csv'
results_summary_df.to_csv(summary_csv_path, index=False)
print(f"\n✓ 模型结果总结已保存到: {summary_csv_path}")

# 创建Markdown格式的详细报告
markdown_report = f"""# Bluebikes Demand Prediction - Model Comparison Report

## Summary

This report compares the performance of different machine learning models for predicting Bluebikes trip counts.

**Best Model (by OSR²):** {results_summary_df.iloc[0]['Model']} (OSR² = {results_summary_df.iloc[0]['OSR2']:.4f})

## Model Performance Comparison

| Model | Train R² | Test R² (OSR²) | Train RMSE | Test RMSE | Best Parameters |
|-------|----------|---------------|------------|----------|-----------------|
"""

for _, row in results_summary_df.iterrows():
    markdown_report += f"| {row['Model']} | {row['Train_R2']:.4f} | {row['OSR2']:.4f} | {row['Train_RMSE']:.2f} | {row['Test_RMSE']:.2f} | {row['Best_Params']} |\n"

markdown_report += "\n## Significant Features / Top Features by Model\n\n"

for _, row in results_summary_df.iterrows():
    markdown_report += f"### {row['Model']}\n\n"
    markdown_report += f"- **Significant Features:** {row['Significant_Features']}\n\n"
    markdown_report += f"- **Top 10 Features:** {row['Top_Features']}\n\n"
    markdown_report += "---\n\n"

# 保存Markdown报告
markdown_path = f'{DATA_DIR}/model_results_report.md'
with open(markdown_path, 'w', encoding='utf-8') as f:
    f.write(markdown_report)
print(f"✓ Markdown报告已保存到: {markdown_path}")

print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)

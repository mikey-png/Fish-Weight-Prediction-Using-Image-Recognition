import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import numpy as np

# 第一步：读取Excel文件中的数据
data = pd.read_excel('./datanew/20240319转换/fish_measurements-20240319.xlsx')

# 准备特征（面积和比例）和目标变量（重量）
X = data[['fish_length', 'fish_height', 'fish_area']]
y = data['Weight']

# 第二步：使用Z-Score方法去除异常值
# 计算特征和目标变量的Z-score
z_scores = np.abs(stats.zscore(data[['fish_length', 'fish_height', 'fish_area', 'Weight']]))

# 设置Z-score阈值（常用为3）来识别异常值
threshold = 3
filtered_entries = (z_scores < threshold).all(axis=1)

# 过滤掉异常值的数据
data_filtered = data[filtered_entries]

# 准备过滤后的特征和目标变量
X_filtered = data_filtered[['fish_length', 'fish_height', 'fish_area']]
y_filtered = data_filtered['Weight']

# 第三步：使用随机森林回归模型拟合过滤后的数据
model_rf = RandomForestRegressor(
    n_estimators=500,  # 增加树的数量
    max_depth=10,  # 限制树的深度
    min_samples_split=10,  # 内部节点拆分所需的最小样本数
    min_samples_leaf=2,  # 叶节点所需的最小样本数
    max_features='sqrt',  # 最佳分割时要考虑的特征数量
    random_state=42,
    n_jobs=-1  # 使用所有可用的CPU核心进行训练
)
model_rf.fit(X_filtered, y_filtered)

# 计算随机森林模型的预测误差
rf_predictions = model_rf.predict(X_filtered)
rf_mse = mean_squared_error(y_filtered, rf_predictions)
rf_r2 = r2_score(y_filtered, rf_predictions)
rf_percentage_errors = np.abs((y_filtered - rf_predictions) / y_filtered) * 100
rf_mean_percentage_error = np.mean(rf_percentage_errors)

print(f"随机森林 - 均方误差: {rf_mse}")
print(f"随机森林 - R平方: {rf_r2}")
print(f"随机森林 - 平均百分比误差: {rf_mean_percentage_error:.2f}%")

# 第四步：使用梯度提升回归模型拟合过滤后的数据
model_gb = GradientBoostingRegressor(
    n_estimators=500,  # 增加弱学习器数量
    learning_rate=0.05,  # 学习率
    max_depth=5,  # 每棵树的最大深度
    random_state=42
)
model_gb.fit(X_filtered, y_filtered)

# 计算梯度提升模型的预测误差
gb_predictions = model_gb.predict(X_filtered)
gb_mse = mean_squared_error(y_filtered, gb_predictions)
gb_r2 = r2_score(y_filtered, gb_predictions)
gb_percentage_errors = np.abs((y_filtered - gb_predictions) / y_filtered) * 100
gb_mean_percentage_error = np.mean(gb_percentage_errors)

print(f"梯度提升 - 均方误差: {gb_mse}")
print(f"梯度提升 - R平方: {gb_r2}")
print(f"梯度提升 - 平均百分比误差: {gb_mean_percentage_error:.2f}%")


# 读取新上传的 CSV 文件
import pandas as pd
new_file_path = "lightning_logs/tb_logs/version_1/solution_data.csv"
df_new = pd.read_csv(new_file_path)
columns_needed = ["y","t","v_pred","v_exact"]
# 选择需要的列
df_new_selected = df_new[columns_needed]

# 按 epoch 进行合并，取均值（忽略 NaN 值）
# df_new_merged = df_new_selected.groupby("epoch", as_index=False).mean()

# 获取前五条和后五条数据
df_new_final = pd.concat([df_new_selected.head(5), df_new_selected.tail(5)], ignore_index=True)
print(df_new_final)
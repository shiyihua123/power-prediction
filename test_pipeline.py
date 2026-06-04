path = "/home/syh/workplace/PythonProject/power-prediction/outputs/NHITS/cv_results_20260604.csv"




import pandas as pd
df = pd.read_csv(path)
print(df)

# 打印出begin_utc列所有的unique值
print(df['begin_utc'].unique())

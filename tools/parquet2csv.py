import pandas as pd


df = pd.read_parquet("C:\\Users\\YF\\Downloads\\0000.parquet")

# 过滤空image字段（模型不是多模态的，在纯文本子集上评估）
filtered_df = df[df['image'] == '']


filtered_df = filtered_df.assign(score=1)
filtered_df[['question', 'answer', 'score']].to_csv('out.csv', index=False)

print("CSV文件生成完成，包含 {} 条数据".format(len(filtered_df)))
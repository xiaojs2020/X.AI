import pandas as pd

# 文件路径
docs_path = "docs.tsv"
predictions_path = "predictions.tsv"

# 加载 TSV 文件
docs_df = pd.read_csv(docs_path, sep='\t', header=None, names=['doc_id', 'text'])
predictions_df = pd.read_csv(predictions_path, sep='\t', header=None, names=['query_id', 'query', 'ground_truth'])

# 查看文件内容的前几行
docs_sample = docs_df.head()
predictions_sample = predictions_df.head()

# 返回样本数据以确认文件内容正确加载
print(docs_sample, predictions_sample)

#
import faiss
import numpy as np
import cohere
import time

# Cohere API 客户端初始化
co = cohere.Client('hcmOSywWJNyRDmLqK2M1ScyVDOCwh3AnFK1hrCLB')

# 提取文档内容
doc_texts = docs_df['text'].tolist()
doc_ids = docs_df['doc_id'].tolist()

# 获取嵌入向量的函数
def get_embeddings(texts):
    response = co.embed(texts=texts, model='embed-english-v2.0')
    return np.array(response.embeddings)

# 分批处理以避免请求限制，每 5 个文档请求一次，防止超出速率限制
batch_size = 5
all_embeddings = []

for i in range(0, len(doc_texts), batch_size):
    batch_texts = doc_texts[i:i + batch_size]
    print(batch_texts)
    embeddings = get_embeddings(batch_texts)
    all_embeddings.extend(embeddings)
    time.sleep(5)  # 每批处理后休眠 5 秒

# 将所有嵌入转换为 numpy 数组
doc_embeddings = np.array(all_embeddings)

# 使用 FAISS 创建索引
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

# 保存索引和文档内容
index_file_path = 'document_index_faiss.index'
documents_file_path = 'documents_faiss.txt'

faiss.write_index(index, index_file_path)

# 保存文档文本与对应的 ID（保持顺序一致）
with open(documents_file_path, 'w', encoding='utf-8') as f:
    for doc_id, text in zip(doc_ids, doc_texts):
        f.write(f"{doc_id}\t{text}\n")
#
# "Index and documents saved successfully."

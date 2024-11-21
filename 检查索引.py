import faiss
import numpy as np
import cohere
import openai

# 加载 API 密钥
openai.api_key = 'sk-proj-your-openai-key'
co = cohere.Client('hcmOSywWJNyRDmLqK2M1ScyVDOCwh3AnFK1hrCLB')

# 加载索引和文档
index = faiss.read_index('document_index_faiss.index')
with open('documents_faiss.txt', 'r', encoding='utf-8') as f:
    documents = [line.strip().split('\t', 1)[1] for line in f if '\t' in line]

# 获取嵌入并检索
def get_embeddings(docs):
    response = co.embed(texts=docs, model='embed-english-v2.0')
    return np.array(response.embeddings)

def search(query, k=5):
    query_embedding = get_embeddings([query])
    _, I = index.search(query_embedding, k)
    return [documents[i] for i in I[0]]

# 示例查询
query = "What is the main cause of HIV-1 infection in children?"
retrieved_docs = search(query, k=5)
print("Retrieved documents:", retrieved_docs)

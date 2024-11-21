import faiss
import numpy as np
import cohere
import openai

# 设置 OpenAI API 密钥
openai.api_key = '/token'
co = cohere.Client('hcmOSywWJNyRDmLqK2M1ScyVDOCwh3AnFK1hrCLB')


# 加载 FAISS 索引
index_file_path = 'document_index_faiss.index'
index = faiss.read_index(index_file_path)

# 加载文档内容
documents_file_path = 'documents_faiss.txt'
with open(documents_file_path, 'r', encoding='utf-8') as f:
    documents = f.readlines()


# 获取 Cohere 嵌入并打印 token 消耗和价格
def get_embeddings(docs):
    response = co.embed(texts=docs, model='embed-english-v2.0')  # 使用 Cohere 的嵌入模型
    embeddings = np.array(response.embeddings)
    # token_usage = response.token_usage  # 获取 token 使用情况
    # print(f"Cohere Token usage: {token_usage} tokens")
    return embeddings


# 文档检索
def search(query, k=5):
    query_embedding = get_embeddings([query])  # 获取查询文本的嵌入
    _, I = index.search(query_embedding, k)  # 搜索最相似的 k 个文档
    return [documents[i] for i in I[0]]


# 构建问答的 prompt 并使用 OpenAI GPT 生成答案
def generate_answer(query, retrieved_docs):
    prompt = f"Context:\n{retrieved_docs}\n\nQuestion: {query}\nAnswer:"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}]
    )

    # 打印 token 消耗和价格
    token_usage = response['usage']['total_tokens']  # 获取 OpenAI 的 token 使用量
    print(f"OpenAI Token usage: {token_usage} tokens")

    # OpenAI 的价格计算可以参考官网的定价信息（每千个 token 的价格）
    # 例如：gpt-3.5-turbo 每千个 token 约 $0.002
    price_per_1000_tokens = 0.002
    cost = token_usage / 1000 * price_per_1000_tokens
    print(f"OpenAI cost for this request: ${cost:.4f}")

    return response['choices'][0]['message']['content']


# 示例查询
query = "What is the main cause of HIV-1 infection in children?"
retrieved_docs = search(query, k=5)  # 检索与查询相关的文档
print("Retrieved documents:", retrieved_docs)

# 使用检索到的文档生成答案
response = generate_answer(query, "\n".join(retrieved_docs))
print("Generated Answer:", response)

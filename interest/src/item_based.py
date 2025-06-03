import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 从 content_pool 提取完整标签全集
def extract_all_tags(content_pool):
    tags = set()
    for item in content_pool:
        tags.update(item.get("tags", []))
    return sorted(list(tags))

# 将单个内容向量化 (One-hot)
def content_to_vector(content_item: dict, all_tags: list) -> np.ndarray:
    vec = np.zeros(len(all_tags))
    for tag in content_item.get("tags", []):
        if tag in all_tags:
            vec[all_tags.index(tag)] = 1.0
    return vec

# 构建内容之间的相似度矩阵
def build_item_similarity(content_pool, all_tags):
    content_ids = [item['id'] for item in content_pool]
    content_vecs = np.array([content_to_vector(item, all_tags) for item in content_pool])
    sim_matrix = cosine_similarity(content_vecs)
    return content_ids, sim_matrix

def item_based_recommend(user_id, user_history, content_pool, all_tags, top_k=5):
    content_ids, sim_matrix = build_item_similarity(content_pool, all_tags)
    id_to_idx = {cid: idx for idx, cid in enumerate(content_ids)}

    history = user_history.get(user_id, [])
    scores = np.zeros(len(content_ids))

    for h_cid in history:
        if h_cid in id_to_idx:
            h_idx = id_to_idx[h_cid]
            scores += sim_matrix[h_idx]  # 累加相似度得分

    for h_cid in history:
        if h_cid in id_to_idx:
            scores[id_to_idx[h_cid]] = 0.0

    sorted_idx = np.argsort(scores)[::-1]
    results = [{"content_id": content_ids[idx], "score": round(scores[idx], 4)} for idx in sorted_idx[:top_k]]
    return results

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 构建所有用户的行为向量
def build_user_behavior_matrix(user_history, content_pool):
    content_ids = [item['id'] for item in content_pool]
    id_to_idx = {cid: idx for idx, cid in enumerate(content_ids)}
    user_ids = list(user_history.keys())

    user_vecs = np.zeros((len(user_ids), len(content_ids)))

    for i, uid in enumerate(user_ids):
        for cid in user_history.get(uid, []):
            if cid in id_to_idx:
                user_vecs[i][id_to_idx[cid]] = 1.0  # 看过的标1
    return user_ids, content_ids, user_vecs

# 主推荐函数
def user_cf_recommend(user_id, user_history, content_pool, top_k=5):
    user_ids, content_ids, user_vecs = build_user_behavior_matrix(user_history, content_pool)
    id_to_idx = {cid: idx for idx, cid in enumerate(content_ids)}
    uid_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}

    if user_id not in uid_to_idx:
        return []  # 新用户，无法推荐

    target_idx = uid_to_idx[user_id]
    sim_scores = cosine_similarity([user_vecs[target_idx]], user_vecs)[0]

    content_scores = {}

    for idx, uid in enumerate(user_ids):
        if uid == user_id:
            continue  # 不和自己算
        sim = sim_scores[idx]
        for cid in user_history.get(uid, []):
            content_scores[cid] = content_scores.get(cid, 0.0) + sim

    # 排除掉用户已经看过的内容
    user_seen = set(user_history.get(user_id, []))
    results = []
    for cid, score in content_scores.items():
        if cid not in user_seen:
            results.append({"content_id": cid, "score": round(score, 4)})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
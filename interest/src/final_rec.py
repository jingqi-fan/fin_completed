from collections import defaultdict

from src.item_based import item_based_recommend
from src.user_cf import user_cf_recommend
from src.utils import extract_all_tags

# 主融合推荐函数 (不使用 user_tags，也不使用 label_match)
def final_recommend(
    user_id: str,
    content_pool: list,
    user_history: dict,
    top_k: int = 5,
    weights: dict = None
) -> list:
    if weights is None:
        weights = {
            "item_based": 0.5,
            "user_cf": 0.5
        }

    # 动态生成统一标签空间
    all_tags = extract_all_tags(content_pool)

    rec2 = item_based_recommend(user_id, user_history, content_pool, all_tags, top_k=5)
    rec3 = user_cf_recommend(user_id, user_history, content_pool, top_k=5)

    score_map = defaultdict(lambda: {"score": 0.0, "content_id": None, "source": []})

    for recs, key, weight in [
        (rec2, "item_based", weights["item_based"]),
        (rec3, "user_cf", weights["user_cf"])
    ]:
        for item in recs:
            cid = item["content_id"]
            score_map[cid]["score"] += item["score"] * weight
            score_map[cid]["content_id"] = cid
            score_map[cid]["source"].append(key)

    final = sorted(score_map.values(), key=lambda x: x["score"], reverse=True)
    return final[:top_k]

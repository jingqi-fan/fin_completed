import json
from src.final_rec import final_recommend

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main(user_id="u123"):
    # 加载数据
    user_history = load_json("data/user_history.json")
    content_pool = load_json("data/content_pool.json")

    # 检查用户历史是否存在
    if user_id not in user_history:
        print(f"❌ 用户 {user_id} 不存在或无历史记录")
        return

    # 推荐融合权重
    weights = {
        "item_based": 0.5,
        "user_cf": 0.5
    }

    # 执行推荐
    results = final_recommend(
        user_id=user_id,
        content_pool=content_pool,
        user_history=user_history,
        top_k=5,
        weights=weights
    )

    # 打印结果
    print(f"\n🔥 推荐结果 for 用户 {user_id}：")
    for idx, item in enumerate(results, 1):
        print(f"{idx}. 内容ID: {item['content_id']} (得分: {item['score']:.4f}, 来源: {','.join(item['source'])})")

if __name__ == "__main__":
    main(user_id="u001")

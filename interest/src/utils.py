def extract_all_tags(content_pool: list) -> list:
    """
    只从内容池中自动提取所有出现的标签，生成统一标签空间。
    """
    tag_set = set()
    for item in content_pool:
        tag_set.update(item.get("tags", []))
    return sorted(list(tag_set))

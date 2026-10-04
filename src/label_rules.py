LABEL_KEYWORDS = {
    "经济": ["经济", "企业", "市场", "金融", "银行", "投资", "贸易", "股市", "工业", "农业"],
    "体育": ["比赛", "足球", "篮球", "奥运", "运动员", "世界杯", "体育", "乒乓", "排球", "网球"],
    "文化": ["文化", "文学", "艺术", "音乐", "电影", "电视", "戏剧", "出版", "书法"],
    "科技": ["科技", "科学", "科研", "电脑", "计算机", "航天", "卫星", "互联网"],
    "法治": ["法制", "法律", "司法", "法院", "检察", "立法", "法规", "法治"],
    "农业": ["农业", "农民", "农村", "粮食", "种植", "畜牧", "农田"],
    "国际": ["国际", "美国", "俄罗斯", "联合国", "日本", "韩国", "印度", "巴基斯坦", "法国", "英国"],
}


def assign_label(title: str) -> str:
    if not title:
        return ""
    title_text = title
    hits = []
    for label, keywords in LABEL_KEYWORDS.items():
        if any(keyword in title_text for keyword in keywords):
            hits.append(label)
    if len(hits) == 1:
        return hits[0]
    return ""

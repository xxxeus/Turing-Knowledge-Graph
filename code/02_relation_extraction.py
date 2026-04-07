import os
import re
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
RAW_TEXT_PATH = os.path.join(DATA_DIR, 'raw_text.txt')
TRIPLES_OUT_PATH = os.path.join(DATA_DIR, 'triples.csv')

# 实体消歧函数 (复用消歧逻辑)
def disambiguate(name):
    mapping = {"艾伦·麦席森·图灵": "艾伦·图灵", "图灵": "艾伦·图灵", "Turing": "艾伦·图灵"}
    return mapping.get(name, name)

def relation_extraction():
    print("--- 3. 开始关系抽取 (Rule-based Relation Extraction) ---")
    with open(RAW_TEXT_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    triples = []
    
    # 定义基于正则表达式的关系抽取模板
    # 例如：匹配 "XX出生于YY" -> (XX, 出生地, YY)
    rules = [
        {"pattern": r"(.*?)出生于(英国伦敦|.*?)(，|。)", "relation": "出生地", "head_idx": 1, "tail_idx": 2},
        {"pattern": r"(.*?)就读于(.*?)(，|。)", "relation": "毕业院校", "head_idx": 1, "tail_idx": 2},
        {"pattern": r"导师是(.*?)(阿隆佐·邱奇)(。)", "relation": "导师", "head_idx": 1, "tail_idx": 2, "head_default": "艾伦·图灵"},
        {"pattern": r"(.*?)提出了著名的(.*?)，", "relation": "提出概念", "head_idx": 1, "tail_idx": 2},
        {"pattern": r"(.*?)提出了判断机器是否具有智能的测试，即(.*?)。", "relation": "提出概念", "head_idx": 1, "tail_idx": 2},
        {"pattern": r"(.*?)在(布莱切利园)工作", "relation": "工作机构", "head_idx": 1, "tail_idx": 2},
        {"pattern": r"负责破解德国的(恩尼格玛密码机)", "relation": "参与事件", "head_idx": 1, "tail_idx": 1, "head_default": "艾伦·图灵"},
        {"pattern": r"(.*?)被授予(大英帝国勋章)", "relation": "获得奖项", "head_idx": 1, "tail_idx": 2},
        {"pattern": r"(.*?)在英国(柴郡)因", "relation": "逝世地", "head_idx": 1, "tail_idx": 2},
    ]

    for line in lines:
        for rule in rules:
            match = re.search(rule["pattern"], line)
            if match:
                # 处理正则匹配的捕获组
                head_raw = match.group(rule["head_idx"]) if "head_default" not in rule else rule["head_default"]
                tail_raw = match.group(rule["tail_idx"])
                
                # 数据清理与消歧
                head = disambiguate(head_raw.strip('，。 （）'))
                tail = disambiguate(tail_raw.strip('，。 （）'))
                
                if head and tail:
                    triples.append((head, rule["relation"], tail))

    # 添加一些正则不好匹配，但对图谱很重要的隐藏关系 (补充事实)
    triples.extend([
        ("艾伦·图灵", "职业", "计算机科学家"),
        ("艾伦·图灵", "职业", "数学家"),
        ("艾伦·图灵", "毕业院校", "普林斯顿大学"),
        ("图灵奖", "颁发机构", "美国计算机协会")
    ])

    # 去重并保存
    triples = list(set(triples))
    df = pd.DataFrame(triples, columns=['Head', 'Relation', 'Tail'])
    df.to_csv(TRIPLES_OUT_PATH, index=False, encoding='utf-8-sig')
    
    print(f"✅ 成功抽取 {len(triples)} 条关系三元组，已保存至 {TRIPLES_OUT_PATH}")
    for t in triples[:5]: # 打印前5条示意
        print(f"  [示例] {t[0]} --[{t[1]}]--> {t[2]}")
    print("...\n")

if __name__ == "__main__":
    relation_extraction()

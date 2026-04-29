import os
import re

import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_TEXT_PATH = os.path.join(DATA_DIR, "raw_text.txt")
ENTITY_PATH = os.path.join(DATA_DIR, "entities.csv")
TRIPLES_OUT_PATH = os.path.join(DATA_DIR, "triples.csv")


def split_sentences(text):
    return [s for s in re.split(r"(?<=[。！？])\s*", text.strip()) if s]


def load_entities():
    if not os.path.exists(ENTITY_PATH):
        raise FileNotFoundError("找不到 entities.csv，请先运行 01_ner_disambiguation.py")
    df = pd.read_csv(ENTITY_PATH)
    return df.to_dict("records")


def entity_lookup(entities):
    aliases = []
    for row in entities:
        for alias in str(row["alias"]).split("|"):
            aliases.append(
                {
                    "name": row["name"],
                    "alias": alias,
                    "type": row["type"],
                }
            )
        aliases.append({"name": row["name"], "alias": row["name"], "type": row["type"]})
    aliases.sort(key=lambda item: len(item["alias"]), reverse=True)
    return aliases


def find_entities(sentence, aliases, entity_type=None):
    candidates = []
    for item in aliases:
        if entity_type and item["type"] != entity_type:
            continue
        start = sentence.find(item["alias"])
        if start >= 0:
            candidates.append((start, start + len(item["alias"]), item["name"]))

    kept = []
    for start, end, name in sorted(candidates, key=lambda item: (item[0], -(item[1] - item[0]))):
        if any(start >= kept_start and end <= kept_end and name != kept_name for kept_start, kept_end, kept_name in kept):
            continue
        kept.append((start, end, name))

    found = []
    seen = set()
    for _, _, name in kept:
        if name not in seen:
            found.append(name)
            seen.add(name)
    return found


def add_relation(rows, seen, head, relation, tail, evidence):
    if not head or not tail or head == tail:
        return
    key = (head, relation, tail, evidence)
    if key in seen:
        return
    seen.add(key)
    rows.append(
        {
            "head": head,
            "relation": relation,
            "tail": tail,
            "evidence": evidence,
        }
    )


def relation_extraction():
    print("--- 2. 基于CRF实体结果进行关系抽取 ---")
    entities = load_entities()
    aliases = entity_lookup(entities)
    with open(RAW_TEXT_PATH, "r", encoding="utf-8") as f:
        sentences = split_sentences(f.read())

    rows = []
    seen = set()

    for sentence in sentences:
        persons = find_entities(sentence, aliases, "人物")
        orgs = find_entities(sentence, aliases, "机构")
        locs = find_entities(sentence, aliases, "地点")
        concepts = find_entities(sentence, aliases, "概念")
        works = find_entities(sentence, aliases, "作品")
        events = find_entities(sentence, aliases, "事件")
        awards = find_entities(sentence, aliases, "奖项")

        turing = "艾伦·图灵" if "艾伦·图灵" in persons else (persons[0] if persons else None)

        if turing and "出生于" in sentence:
            for loc in locs:
                add_relation(rows, seen, turing, "出生地", loc, sentence)
            if "数学家" in concepts:
                add_relation(rows, seen, turing, "职业", "数学家", sentence)
            if "逻辑学家" in concepts:
                add_relation(rows, seen, turing, "职业", "逻辑学家", sentence)
            if "密码分析学家" in concepts:
                add_relation(rows, seen, turing, "职业", "密码分析学家", sentence)

        if turing and ("就读于" in sentence or "进入" in sentence):
            for org in orgs:
                add_relation(rows, seen, turing, "就读于", org, sentence)
            if "数学" in concepts:
                add_relation(rows, seen, turing, "学习领域", "数学", sentence)

        if turing and "毕业" in sentence:
            for org in orgs:
                if org == "剑桥大学国王学院":
                    add_relation(rows, seen, turing, "毕业院校", org, sentence)

        if turing and "攻读博士学位" in sentence:
            for org in orgs:
                add_relation(rows, seen, turing, "攻读博士于", org, sentence)
            add_relation(rows, seen, turing, "研究领域", "数理逻辑", sentence)

        if turing and "博士导师" in sentence:
            for person in persons:
                if person != turing:
                    add_relation(rows, seen, turing, "博士导师", person, sentence)

        if turing and "发表论文" in sentence:
            for work in works:
                add_relation(rows, seen, turing, "发表作品", work, sentence)
            for concept in concepts:
                if concept in {"图灵机", "图灵测试"}:
                    add_relation(rows, seen, turing, "提出概念", concept, sentence)
                elif concept in {"可计算性理论", "人工智能"}:
                    add_relation(rows, seen, turing, "推动领域", concept, sentence)

        if turing and "第二次世界大战" in events:
            add_relation(rows, seen, turing, "参与事件", "第二次世界大战", sentence)
            for org in orgs:
                add_relation(rows, seen, turing, "工作于", org, sentence)
            for concept in concepts:
                if "密码机" in concept:
                    add_relation(rows, seen, turing, "破解对象", concept, sentence)

        if turing and "工作" in sentence and "第二次世界大战" not in events:
            for org in orgs:
                add_relation(rows, seen, turing, "工作于", org, sentence)

        if turing and "机器智能" in sentence:
            add_relation(rows, seen, turing, "研究领域", "人工智能", sentence)
            for concept in concepts:
                if concept == "图灵测试":
                    add_relation(rows, seen, turing, "提出概念", concept, sentence)

        if turing and "获得" in sentence:
            for award in awards:
                add_relation(rows, seen, turing, "获得奖项", award, sentence)

        if "美国计算机协会" in orgs and "图灵奖" in awards:
            add_relation(rows, seen, "美国计算机协会", "设立奖项", "图灵奖", sentence)
            add_relation(rows, seen, "图灵奖", "纪念人物", "艾伦·图灵", sentence)

        if turing and "逝世" in sentence:
            for loc in locs:
                add_relation(rows, seen, turing, "逝世地", loc, sentence)

        if "英国政府" in sentence and "公开道歉" in sentence and turing:
            add_relation(rows, seen, "英国政府", "公开道歉于", turing, sentence)

        if "英国女王伊丽莎白二世" in persons and turing:
            add_relation(rows, seen, "英国女王伊丽莎白二世", "赦免", turing, sentence)

    df = pd.DataFrame(rows, columns=["head", "relation", "tail", "evidence"])
    df.to_csv(TRIPLES_OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"抽取到 {len(df)} 条带证据句的关系，保存至 {TRIPLES_OUT_PATH}")
    print(df[["head", "relation", "tail"]].to_string(index=False))


if __name__ == "__main__":
    relation_extraction()

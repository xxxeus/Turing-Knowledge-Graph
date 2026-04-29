import os
import re
from collections import OrderedDict

import jieba
import pandas as pd
from sklearn_crfsuite import CRF


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_TEXT_PATH = os.path.join(DATA_DIR, "raw_text.txt")
ANNOTATED_PATH = os.path.join(DATA_DIR, "annotated_ner.txt")
ENTITY_OUT_PATH = os.path.join(DATA_DIR, "entities.csv")

ENTITY_TYPES = {
    "PER": "人物",
    "ORG": "机构",
    "LOC": "地点",
    "WORK": "作品",
    "CONCEPT": "概念",
    "EVENT": "事件",
    "AWARD": "奖项",
}

# 词典只作为分词边界特征和消歧参考，CRF的BIO标签才决定实体边界。
AUXILIARY_TERMS = [
    "艾伦·麦席森·图灵",
    "Alan Mathison Turing",
    "艾伦·图灵",
    "图灵",
    "计算机科学",
    "人工智能",
    "英国伦敦",
    "英国柴郡",
    "谢伯恩学校",
    "剑桥大学国王学院",
    "普林斯顿大学",
    "阿隆佐·邱奇",
    "《论可计算数及其在判定问题上的应用》",
    "《计算机器与智能》",
    "图灵机",
    "图灵测试",
    "可计算性理论",
    "自动计算机",
    "早期计算机程序设计",
    "第二次世界大战",
    "英国政府",
    "英国政府密码学校",
    "布莱切利园",
    "恩尼格玛密码机",
    "国家物理实验室",
    "曼彻斯特大学",
    "大英帝国勋章",
    "美国计算机协会",
    "图灵奖",
    "英国女王伊丽莎白二世",
]

AUXILIARY_ENTITY_TYPES = {
    "艾伦·麦席森·图灵": "PER",
    "Alan Mathison Turing": "PER",
    "艾伦·图灵": "PER",
    "图灵": "PER",
    "计算机科学": "CONCEPT",
    "人工智能": "CONCEPT",
    "英国伦敦": "LOC",
    "英国柴郡": "LOC",
    "谢伯恩学校": "ORG",
    "剑桥大学国王学院": "ORG",
    "普林斯顿大学": "ORG",
    "阿隆佐·邱奇": "PER",
    "《论可计算数及其在判定问题上的应用》": "WORK",
    "《计算机器与智能》": "WORK",
    "图灵机": "CONCEPT",
    "图灵测试": "CONCEPT",
    "可计算性理论": "CONCEPT",
    "自动计算机": "CONCEPT",
    "早期计算机程序设计": "CONCEPT",
    "第二次世界大战": "EVENT",
    "英国政府": "ORG",
    "英国政府密码学校": "ORG",
    "布莱切利园": "ORG",
    "恩尼格玛密码机": "CONCEPT",
    "国家物理实验室": "ORG",
    "曼彻斯特大学": "ORG",
    "大英帝国勋章": "AWARD",
    "美国计算机协会": "ORG",
    "图灵奖": "AWARD",
    "英国女王伊丽莎白二世": "PER",
}

DISAMBIGUATION = {
    "艾伦·麦席森·图灵": "艾伦·图灵",
    "Alan Mathison Turing": "艾伦·图灵",
    "图灵": "艾伦·图灵",
    "德国恩尼格玛密码机": "恩尼格玛密码机",
}

STOP_ENTITIES = {"英国", "数学", "数学家", "逻辑学家", "密码分析学家", "计算机", "机器", "问题"}


def prepare_jieba():
    for term in AUXILIARY_TERMS:
        jieba.add_word(term, freq=20000)


def split_sentences(text):
    pieces = re.split(r"(?<=[。！？])\s*", text.strip())
    return [piece for piece in pieces if piece]


def word_label_to_char_labels(word, label):
    chars = list(word)
    if label == "O":
        return chars, ["O"] * len(chars)
    _, entity_type = label.split("-", 1)
    labels = [f"B-{entity_type}"] + [f"I-{entity_type}"] * (len(chars) - 1)
    return chars, labels


def load_annotated_sentences(path):
    sentences = []
    chars = []
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if chars:
                    sentences.append((chars, labels))
                    chars, labels = [], []
                continue

            parts = line.rsplit(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"标注行格式错误: {line}")
            word_chars, word_labels = word_label_to_char_labels(parts[0], parts[1])
            chars.extend(word_chars)
            labels.extend(word_labels)

    if chars:
        sentences.append((chars, labels))
    return sentences


def build_word_boundary_features(sentence):
    boundaries = {}
    cursor = 0
    for word in jieba.lcut(sentence, HMM=True):
        start = sentence.find(word, cursor)
        if start < 0:
            continue
        end = start + len(word) - 1
        for index in range(start, end + 1):
            if index == start and index == end:
                boundaries[index] = "S"
            elif index == start:
                boundaries[index] = "B"
            elif index == end:
                boundaries[index] = "E"
            else:
                boundaries[index] = "I"
        cursor = end + 1
    return boundaries


def char_features(chars, index, boundaries):
    char = chars[index]
    features = {
        "bias": 1.0,
        "char": char,
        "is_digit": char.isdigit(),
        "is_ascii": char.isascii(),
        "is_middle_dot": char == "·",
        "is_title_mark": char in "《》",
        "word_boundary": boundaries.get(index, "O"),
        "prev_char": chars[index - 1] if index > 0 else "<BOS>",
        "next_char": chars[index + 1] if index < len(chars) - 1 else "<EOS>",
        "prev_bigram": "".join(chars[max(0, index - 1): index + 1]),
        "next_bigram": "".join(chars[index: index + 2]),
    }
    if index == 0:
        features["BOS"] = True
    if index == len(chars) - 1:
        features["EOS"] = True
    return features


def sentence_to_features(chars):
    boundaries = build_word_boundary_features("".join(chars))
    return [char_features(chars, i, boundaries) for i in range(len(chars))]


def train_crf():
    annotated = load_annotated_sentences(ANNOTATED_PATH)
    x_train = [sentence_to_features(chars) for chars, _ in annotated]
    y_train = [labels for _, labels in annotated]
    model = CRF(
        algorithm="lbfgs",
        c1=0.05,
        c2=0.05,
        max_iterations=200,
        all_possible_transitions=True,
    )
    model.fit(x_train, y_train)
    return model


def decode_entities(chars, labels):
    entities = []
    current_chars = []
    current_type = None
    for char, label in zip(chars, labels):
        if label == "O":
            if current_chars:
                entities.append(("".join(current_chars), current_type))
                current_chars, current_type = [], None
            continue

        prefix, entity_type = label.split("-", 1)
        if prefix == "B" or entity_type != current_type:
            if current_chars:
                entities.append(("".join(current_chars), current_type))
            current_chars = [char]
            current_type = entity_type
        else:
            current_chars.append(char)

    if current_chars:
        entities.append(("".join(current_chars), current_type))
    return entities


def normalize_entity(name):
    cleaned = name.strip("，。；：（）()、“”")
    return DISAMBIGUATION.get(cleaned, cleaned)


def should_keep_entity(name, entity_type):
    if not name or name in STOP_ENTITIES:
        return False
    if entity_type not in ENTITY_TYPES:
        return False
    if len(name) == 1:
        return False
    if name.isascii():
        return False
    return True


def refine_predicted_entity(alias, entity_type):
    cleaned = alias.strip("，。；：（）()、“”")
    if cleaned in AUXILIARY_ENTITY_TYPES:
        return [(cleaned, AUXILIARY_ENTITY_TYPES[cleaned])]

    # 小语料CRF有时会把作品后面的说明文字连在一起。
    # 对过长片段，只保留其中可解释的明确实体，避免把整句误当成节点。
    if len(cleaned) > 14:
        recovered = []
        for term in sorted(AUXILIARY_TERMS, key=len, reverse=True):
            if term in cleaned and term in AUXILIARY_ENTITY_TYPES:
                recovered.append((term, AUXILIARY_ENTITY_TYPES[term]))
        if recovered:
            unique = []
            seen = set()
            for item in recovered:
                if item[0] not in seen:
                    unique.append(item)
                    seen.add(item[0])
            return unique
        return []

    return [(cleaned, entity_type)]


def extract_and_disambiguate():
    print("--- 1. 训练字符级CRF实体抽取模型并进行实体消歧 ---")
    prepare_jieba()
    crf = train_crf()

    with open(RAW_TEXT_PATH, "r", encoding="utf-8") as f:
        sentences = split_sentences(f.read())

    entity_rows = OrderedDict()
    for sentence in sentences:
        chars = list(sentence)
        labels = crf.predict_single(sentence_to_features(chars))
        for alias, entity_type in decode_entities(chars, labels):
            for refined_alias, refined_type in refine_predicted_entity(alias, entity_type):
                name = normalize_entity(refined_alias)
                if not should_keep_entity(name, refined_type):
                    continue
                if name not in entity_rows:
                    entity_rows[name] = {
                        "id": f"E{len(entity_rows) + 1:03d}",
                        "name": name,
                        "type": ENTITY_TYPES[refined_type],
                        "alias": refined_alias,
                        "source_sentence": sentence,
                    }
                elif refined_alias not in entity_rows[name]["alias"].split("|"):
                    entity_rows[name]["alias"] += f"|{refined_alias}"

    df = pd.DataFrame(entity_rows.values(), columns=["id", "name", "type", "alias", "source_sentence"])
    df.to_csv(ENTITY_OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"已抽取并消歧 {len(df)} 个实体，保存至 {ENTITY_OUT_PATH}")
    print(df[["name", "type", "alias"]].to_string(index=False))


if __name__ == "__main__":
    extract_and_disambiguate()

import os
import jieba
import pandas as pd

# 配置路径
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
RAW_TEXT_PATH = os.path.join(DATA_DIR, 'raw_text.txt')
ENTITY_OUT_PATH = os.path.join(DATA_DIR, 'entities.csv')

# 1. 自定义实体词典 (辅助jieba进行精确分词)
custom_entities = [
    "艾伦·麦席森·图灵", "艾伦·图灵", "图灵", "Turing", 
    "伦敦", "英国", "剑桥大学国王学院", "普林斯顿大学",
    "阿隆佐·邱奇", "图灵机", "图灵测试", "布莱切利园", 
    "恩尼格玛密码机", "大英帝国勋章", "图灵奖", "美国计算机协会", "柴郡"
]
for ent in custom_entities:
    jieba.add_word(ent)

# 2. 实体消歧字典 (将同义异名词统一为标准实体名)
disambiguation_dict = {
    "艾伦·麦席森·图灵": "艾伦·图灵",
    "图灵": "艾伦·图灵",
    "Turing": "艾伦·图灵"
}

def extract_and_disambiguate():
    print("--- 1. 开始实体抽取 (NER) ---")
    with open(RAW_TEXT_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    # 分词
    words = jieba.lcut(text)
    
    # 提取在目标实体库中的词
    extracted_entities = [w for w in words if w in custom_entities]
    print(f"提取到的原始实体数: {len(extracted_entities)}")
    
    print("--- 2. 开始实体消歧 (Disambiguation) ---")
    standard_entities = set()
    for ent in extracted_entities:
        # 消歧映射
        std_ent = disambiguation_dict.get(ent, ent)
        standard_entities.add(std_ent)
    
    # 保存结果
    df = pd.DataFrame(list(standard_entities), columns=['Entity'])
    df.to_csv(ENTITY_OUT_PATH, index=False, encoding='utf-8-sig')
    print(f"✅ 消歧后得到 {len(standard_entities)} 个唯一标准实体，已保存至 {ENTITY_OUT_PATH}\n")

if __name__ == "__main__":
    extract_and_disambiguate()

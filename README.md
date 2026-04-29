# 艾伦·图灵知识图谱

本项目是《认知计算与知识工程》课程个人作业，目标是围绕艾伦·图灵构建一个小型但完整、可复现的知识图谱。流程从非结构化中文语料开始，经过人工 BIO 标注、CRF 实体抽取、实体消歧、关系抽取，最后生成知识图谱图片。

## 项目结构

```text
data/
  raw_text.txt          # 非结构化原始语料
  annotated_ner.txt     # BIO格式人工标注样本
  entities.csv          # CRF抽取并消歧后的实体表
  triples.csv           # 带证据句的关系三元组
  kg_result.png         # 最终知识图谱图片
code/
  01_ner_disambiguation.py
  02_relation_extraction.py
  03_kg_visualization.py
requirements.txt
```

## 构建方法

### 1. 数据收集

`data/raw_text.txt` 保存关于图灵生平、教育经历、学术贡献、二战密码破译、奖项纪念和身后平反的非结构化中文文本。项目没有直接把结构化知识导入图谱，而是从文本中抽取实体和关系。

### 2. 实体抽取

`data/annotated_ner.txt` 使用 BIO 格式标注训练样本，实体类型包括：

- `PER`：人物
- `ORG`：机构
- `LOC`：地点
- `WORK`：作品
- `CONCEPT`：概念
- `EVENT`：事件
- `AWARD`：奖项

`code/01_ner_disambiguation.py` 使用 `sklearn-crfsuite` 训练 CRF 模型。特征包括当前词、前后词、词性、词长、数字特征、专名符号、书名号、前后缀等。项目中保留少量辅助词典，但它只用于帮助分词和提供 CRF 特征，不作为主要实体匹配方法。

脚本会输出 `data/entities.csv`，字段为：

```text
id,name,type,alias,source_sentence
```

### 3. 关系抽取

`code/02_relation_extraction.py` 读取 CRF 输出的实体表，再结合原始证据句进行规则模板抽取。当前覆盖的关系包括出生地、就读于、毕业院校、博士导师、发表作品、提出概念、工作于、参与事件、破解对象、获得奖项、设立奖项、纪念人物、逝世地、赦免等。

脚本会输出 `data/triples.csv`，字段为：

```text
head,relation,tail,evidence
```

每条关系都保留 `evidence` 证据句，方便检查图谱事实来源。

### 4. 图谱可视化

`code/03_kg_visualization.py` 使用 `networkx` 构建有向图，并用 `matplotlib` 输出高清图片。节点颜色按实体类型区分，边标签显示关系名称，最终图片保存为 `data/kg_result.png`。

## 运行方式

```bash
pip install -r requirements.txt
python code/01_ner_disambiguation.py
python code/02_relation_extraction.py
python code/03_kg_visualization.py
```

运行完成后，提交或拍照使用 `data/kg_result.png`。

## 对中期建议的改进

1. **避免直接导入结构化数据。** 当前项目从 `raw_text.txt` 的非结构化文本开始处理，实体和关系结果都能追溯到原始句子。
2. **避免只用预设词匹配。** 实体抽取主流程使用传统机器学习方法 CRF，词典只作为分词和特征辅助。
3. **避免全部依赖大语言模型。** 项目主流程不使用大语言模型，而是采用人工标注样本、CRF、规则模板和图结构可视化完成知识图谱构建。

## 当前图谱内容

图谱围绕“艾伦·图灵”展开，包含人物、机构、地点、作品、概念、事件和奖项等实体，覆盖图灵的出生地、教育经历、博士导师、代表论文、图灵机、图灵测试、布莱切利园、恩尼格玛密码机、大英帝国勋章、图灵奖等核心知识。

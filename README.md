# Turing-Knowledge-Graph
# 艾伦·图灵 知识图谱 (Alan Turing Knowledge Graph)

## 📖 项目简介
本项目为《认知计算与知识工程》课程作业。旨在围绕计算机科学之父、人工智能之父——**艾伦·图灵（Alan Turing）**，构建一个多维度的知识图谱。项目涵盖了从非结构化文本中抽取实体、实体消歧、关系抽取，到最终图谱可视化的全过程。

## 📂 仓库目录结构
* `data/` : 存放原始语料文本及结构化后的三元组数据 (CSV)。
* `code/` : 存放实体抽取、实体消歧、关系抽取及图谱可视化的 Python 脚本。
* `img/` : 存放最终生成的知识图谱可视化截图/照片。

## 📅 开发计划 (对应每周课程进度)
- [x] **Week 5**: 确立本体模式（Schema），完成语料收集，建立代码仓库。
- [ ] **Week 6**: 基于自然语言处理工具完成**实体抽取（NER）**，并进行同义词映射实现**实体消歧**。
- [ ] **Week 7**: 通过规则模板与依存句法分析完成**关系抽取**，生成 `(实体, 关系, 实体)` 三元组。
- [ ] **Week 8**: 使用图数据库 (Neo4j) 或 Python 可视化库 (NetworkX/Pyecharts) 完成图谱构建，并拍照提交。

## 🏷️ 本体设计 (Ontology Schema)
* **实体类 (Entities):** 人物 (Person)、机构 (Organization)、概念 (Concept)、事件 (Event)、奖项 (Award)
* **关系类 (Relations):** 毕业于 (graduated_from)、提出 (proposed)、参与 (participated_in)、获得 (awarded)、工作于 (worked_at)

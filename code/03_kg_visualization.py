import os
import platform
import math

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ENTITY_PATH = os.path.join(DATA_DIR, "entities.csv")
TRIPLES_PATH = os.path.join(DATA_DIR, "triples.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "kg_result.png")

TYPE_COLORS = {
    "人物": "#f4a261",
    "机构": "#2a9d8f",
    "地点": "#457b9d",
    "作品": "#8d99ae",
    "概念": "#e9c46a",
    "事件": "#e76f51",
    "奖项": "#9b5de5",
    "未知": "#b8b8b8",
}


def set_chinese_font():
    system = platform.system()
    if system == "Windows":
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    elif system == "Darwin":
        plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
    else:
        plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False


def build_and_plot_graph():
    print("--- 3. 构建并可视化知识图谱 ---")
    if not os.path.exists(ENTITY_PATH):
        raise FileNotFoundError("找不到 entities.csv，请先运行 01_ner_disambiguation.py")
    if not os.path.exists(TRIPLES_PATH):
        raise FileNotFoundError("找不到 triples.csv，请先运行 02_relation_extraction.py")

    set_chinese_font()
    entities = pd.read_csv(ENTITY_PATH)
    triples = pd.read_csv(TRIPLES_PATH)

    type_map = dict(zip(entities["name"], entities["type"]))
    graph = nx.DiGraph()

    for _, row in entities.iterrows():
        graph.add_node(row["name"], entity_type=row["type"])

    for _, row in triples.iterrows():
        graph.add_node(row["head"], entity_type=type_map.get(row["head"], "未知"))
        graph.add_node(row["tail"], entity_type=type_map.get(row["tail"], "未知"))
        graph.add_edge(row["head"], row["tail"], label=row["relation"])

    plt.figure(figsize=(18, 12), dpi=180)
    pos = radial_layout(graph, center="艾伦·图灵")

    node_colors = [TYPE_COLORS.get(graph.nodes[node].get("entity_type", "未知"), TYPE_COLORS["未知"]) for node in graph.nodes]
    node_sizes = [3600 if node == "艾伦·图灵" else 2300 for node in graph.nodes]

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="#2f2f2f",
        linewidths=1.2,
        alpha=0.95,
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        width=1.4,
        edge_color="#6c757d",
        arrows=True,
        arrowsize=18,
        connectionstyle="arc3,rad=0.08",
        min_source_margin=18,
        min_target_margin=18,
    )
    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight="bold", font_color="#202020")

    edge_labels = nx.get_edge_attributes(graph, "label")
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=7,
        font_color="#9d0208",
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.75},
        label_pos=0.68,
    )

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=entity_type, markerfacecolor=color, markersize=12)
        for entity_type, color in TYPE_COLORS.items()
        if entity_type in {graph.nodes[node].get("entity_type", "未知") for node in graph.nodes}
    ]
    plt.legend(handles=legend_handles, loc="lower left", frameon=False, fontsize=10)
    plt.title("艾伦·图灵知识图谱", fontsize=22, fontweight="bold", pad=22)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, bbox_inches="tight", facecolor="white")
    print(f"知识图谱图片已保存至 {OUTPUT_PATH}")


def radial_layout(graph, center):
    if center not in graph:
        return nx.spring_layout(graph, k=1.6, iterations=200, seed=7)

    nodes = [node for node in graph.nodes if node != center]
    nodes.sort(key=lambda node: (graph.nodes[node].get("entity_type", ""), node))
    pos = {center: (0.0, 0.0)}

    radius_x = 6.4
    radius_y = 4.3
    for index, node in enumerate(nodes):
        angle = 2 * math.pi * index / max(len(nodes), 1)
        pos[node] = (radius_x * math.cos(angle), radius_y * math.sin(angle))
    return pos


if __name__ == "__main__":
    build_and_plot_graph()

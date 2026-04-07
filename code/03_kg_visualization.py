import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import platform

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
TRIPLES_PATH = os.path.join(DATA_DIR, 'triples.csv')

def set_chinese_font():
    """解决 matplotlib 中文乱码问题"""
    system = platform.system()
    if system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
    elif system == 'Darwin': # Mac OS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    else: # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False

def build_and_plot_graph():
    print("--- 4. 开始构建知识图谱并可视化 ---")
    set_chinese_font()
    
    if not os.path.exists(TRIPLES_PATH):
        print("错误: 找不到 triples.csv，请先运行 02_relation_extraction.py")
        return

    df = pd.read_csv(TRIPLES_PATH)
    
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['Head'], row['Tail'], label=row['Relation'])
    
    plt.figure(figsize=(12, 8), dpi=150)
    
    # 采用弹簧布局，通过 k 参数调节节点距离
    pos = nx.spring_layout(G, k=1.2, seed=42)
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=2800, node_color='#87CEFA', edgecolors='#4682B4', alpha=0.9)
    
    # 绘制边
    nx.draw_networkx_edges(G, pos, width=1.8, edge_color='#A9A9A9', arrows=True, arrowsize=20, connectionstyle='arc3,rad=0.1')
    
    # 绘制节点标签
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', font_color='#333333')
    
    # 绘制关系标签
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color='#DC143C')
    
    plt.title("艾伦·图灵 (Alan Turing) 知识图谱", fontsize=18, fontweight='bold', pad=20)
    plt.axis('off')
    
    print("✅ 知识图谱生成完毕，正在弹出可视化窗口...")
    
    # 保存图片到 data 目录 (可选)
    plt.savefig(os.path.join(DATA_DIR, 'kg_result.png'), bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    build_and_plot_graph()

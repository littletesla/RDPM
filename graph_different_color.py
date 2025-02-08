def graph(df, name):
    import networkx as nx
    import matplotlib.pyplot as plt
    G = nx.Graph()
    for node in df.columns:
        G.add_node(node)
    positive_threshold = 0.1
    negative_threshold = -0.1
    for i, source in enumerate(df.index):
        for j, target in enumerate(df.columns):
            if i != j:
                weight = df.at[source, target]
                if weight > positive_threshold or weight < negative_threshold:
                    G.add_edge(source, target, weight=weight)
    pos = nx.circular_layout(G)
    node_colors = []
    for i, node in enumerate(df.columns):
        if i < 10:
            node_colors.append('purple')
        elif i < 20:
            node_colors.append('yellow')
        else:
            node_colors.append('green')
    positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > positive_threshold]
    negative_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < negative_threshold]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color=node_colors, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=positive_edges, edge_color='blue', width=0.5, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=negative_edges, edge_color='red', width=0.5, ax=ax)
    label_pos = {node: (coords[0] * 1.1, coords[1] * 1.1) for node, coords in pos.items()}
    nx.draw_networkx_labels(G, label_pos, font_size=8, ax=ax)
    plt.savefig(f'{name}.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()

if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('name.csv', index_col=0)
    graph(df, 'name')
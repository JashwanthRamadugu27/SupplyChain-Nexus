import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("feature_engineered_supply_chain_data.csv")

print("Data loaded successfully.")
print("Shape:", df.shape)


# ============================================================
# 2. CREATE DIRECTED GRAPH
# ============================================================

G = nx.DiGraph()


# ============================================================
# 3. ADD SUPPLY CHAIN CONNECTIONS
# ============================================================

for _, row in df.iterrows():

    origin = row["origin_port"]
    hub = row["logistic_hub"]
    customer = row["customer"]

    # --------------------------------------------------------
    # Origin Port → Logistic Hub
    # --------------------------------------------------------

    if pd.notna(origin) and pd.notna(hub):

        G.add_edge(
            f"PORT:{origin}",
            f"HUB:{hub}"
        )

    # --------------------------------------------------------
    # Logistic Hub → Customer
    # --------------------------------------------------------

    if pd.notna(hub) and pd.notna(customer):

        G.add_edge(
            f"HUB:{hub}",
            f"CUSTOMER:{customer}"
        )

    # --------------------------------------------------------
    # Direct Shipment
    # Origin Port → Customer
    # --------------------------------------------------------

    elif pd.notna(origin) and pd.notna(customer):

        G.add_edge(
            f"PORT:{origin}",
            f"CUSTOMER:{customer}"
        )


# ============================================================
# 4. NETWORK SUMMARY
# ============================================================

print("\n================ NETWORK SUMMARY ================")

print("Number of nodes:", G.number_of_nodes())

print("Number of connections:", G.number_of_edges())


# ============================================================
# 5. NODE TYPES
# ============================================================

ports = [
    node for node in G.nodes
    if node.startswith("PORT:")
]

hubs = [
    node for node in G.nodes
    if node.startswith("HUB:")
]

customers = [
    node for node in G.nodes
    if node.startswith("CUSTOMER:")
]


print("\nNode types:")

print("Ports:", len(ports))

print("Logistic hubs:", len(hubs))

print("Customers:", len(customers))


# ============================================================
# 6. NODE DEGREE ANALYSIS
# ============================================================

degree_data = []

for node in G.nodes:

    degree_data.append({

        "node": node,

        "node_type": node.split(":")[0],

        "in_degree": G.in_degree(node),

        "out_degree": G.out_degree(node),

        "total_degree": G.degree(node)

    })


degree_df = pd.DataFrame(degree_data)


# ============================================================
# 7. TOP CONNECTED NODES
# ============================================================

top_nodes = degree_df.sort_values(
    "total_degree",
    ascending=False
).head(20)


print("\n================ TOP CONNECTED NODES ================")

print(top_nodes.to_string(index=False))


# ============================================================
# 8. BETWEenness CENTRALITY
# ============================================================

centrality = nx.betweenness_centrality(G)

centrality_df = pd.DataFrame({

    "node": list(centrality.keys()),

    "betweenness_centrality": list(
        centrality.values()
    )

})


centrality_df = centrality_df.sort_values(
    "betweenness_centrality",
    ascending=False
)


print("\n================ TOP CENTRAL NODES ================")

print(
    centrality_df.head(20).to_string(
        index=False
    )
)


# ============================================================
# 9. SAVE NETWORK INFORMATION
# ============================================================

degree_df.to_csv(
    "network_node_degree.csv",
    index=False
)

centrality_df.to_csv(
    "network_node_centrality.csv",
    index=False
)


# ============================================================
# 10. CREATE NETWORK VISUALIZATION
# ============================================================

plt.figure(figsize=(16, 10))

pos = nx.spring_layout(
    G,
    seed=42,
    k=1.2
)


# Draw nodes

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=300
)


# Draw connections

nx.draw_networkx_edges(
    G,
    pos,
    arrows=True,
    alpha=0.35,
    arrowsize=10
)


# Draw labels

nx.draw_networkx_labels(
    G,
    pos,
    font_size=7
)


plt.title(
    "SupplyChain Nexus - Supply Chain Network"
)

plt.axis("off")

plt.tight_layout()


plt.savefig(
    "SupplyChain_Network.png",
    dpi=300
)


plt.show()


# ============================================================
# 11. FINAL OUTPUT
# ============================================================

print("\n================ FILES CREATED ================")

print("1. network_node_degree.csv")

print("2. network_node_centrality.csv")

print("3. SupplyChain_Network.png")

print("\nNetwork construction completed successfully.")
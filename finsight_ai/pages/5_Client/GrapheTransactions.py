"""🕸️ Graphe des Transactions — Vue simple (cohérente schéma Supabase)

- Construit un graphe comptes→comptes depuis `transactions` (account_id -> counter_party)
- Calcule des centralités basiques (degree, betweenness, closeness, PageRank)
- Détecte des communautés (greedy modularity)
- Affiche une visualisation interactive + tableaux simples

Pas d'IA/ML ici (page volontairement simple et pédagogique).
"""

import os
import sys
from pathlib import Path
import importlib.util
from typing import Dict

import streamlit as st
import networkx as nx

_THIS_DIR = Path(__file__).parent


def _load_local(module_filename: str, module_name: str):
    path = _THIS_DIR / module_filename
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module

_q = _load_local("queries.py", "client_queries")
_viz = _load_local("visualization.py", "client_visualization")


def build_graph(edges_df):
    G = nx.DiGraph()
    for _, row in edges_df.iterrows():
        G.add_edge(
            row["source"],
            row["target"],
            weight=float(row["total_amount"] or 0.0),
            tx_count=int(row["tx_count"] or 0),
            avg_amount=float(row["avg_amount"] or 0.0),
            max_amount=float(row["max_amount"] or 0.0),
        )
    return G


def compute_graph_features(G: nx.DiGraph):
    if len(G) == 0:
        import pandas as pd
        return pd.DataFrame(columns=[
            "account_id", "degree_centrality", "in_degree", "out_degree",
            "betweenness_centrality", "closeness_centrality", "pagerank",
            "in_amount", "out_amount", "total_amount", "total_degree"
        ])
    import pandas as pd
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, weight="weight", normalized=True)
    closeness_centrality = nx.closeness_centrality(G)
    try:
        pagerank = nx.pagerank(G, max_iter=50)
    except Exception:
        pagerank = {node: 0 for node in G.nodes}
    rows = []
    for node in G.nodes:
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        in_amount = sum([G[u][node].get("weight", 0) for u in G.predecessors(node)])
        out_amount = sum([G[node][v].get("weight", 0) for v in G.successors(node)])
        rows.append({
            "account_id": node,
            "degree_centrality": degree_centrality.get(node, 0),
            "in_degree": in_degree,
            "out_degree": out_degree,
            "betweenness_centrality": betweenness_centrality.get(node, 0),
            "closeness_centrality": closeness_centrality.get(node, 0),
            "pagerank": pagerank.get(node, 0),
            "in_amount": in_amount,
            "out_amount": out_amount,
            "total_amount": in_amount + out_amount,
            "total_degree": in_degree + out_degree,
        })
    return pd.DataFrame(rows)


def detect_communities(G: nx.DiGraph) -> Dict[str, int]:
    if len(G) == 0:
        return {}
    G_undirected = G.to_undirected()
    try:
        import networkx.algorithms.community as nx_comm
        communities = list(nx_comm.greedy_modularity_communities(G_undirected))
        mapping = {}
        for i, comm in enumerate(communities):
            for node in comm:
                mapping[node] = i
        return mapping
    except Exception:
        return {node: 0 for node in G.nodes}


def main() -> None:
    st.subheader("🕸️ Transaction Graph (simple)")

    c1, c2, c3 = st.columns(3)
    with c1:
        period_options = [7, 14, 30, 60, 90, "Custom"]
        period_choice = st.selectbox("Period", period_options, index=2)
        if period_choice == "Custom":
            days = st.number_input("Number of days", min_value=1, value=30, step=1)
        else:
            days = period_choice
    with c2:
        min_amount = st.number_input("Minimum amount ($)", min_value=0, value=300, step=100)
    with c3:
        connections_options = [300, 500, 1000, 2000, "Custom"]
        connections_choice = st.selectbox("Max connections", connections_options, index=2)
        if connections_choice == "Custom":
            max_edges = st.number_input("Number of connections", min_value=100, value=1000, step=100)
        else:
            max_edges = connections_choice

    with st.spinner("Building network..."):
        edges_df = _q.get_transaction_network(days, min_amount, max_edges)
        if edges_df.empty:
            st.info("No connections found with these filters.")
            return
        G = build_graph(edges_df)
        features_df = compute_graph_features(G)
        communities = detect_communities(G)

    st.markdown(
        f"""
        <div style="background: linear-gradient(90deg,#4C1D95,#5B21B6); 
                    border: 1px solid #8B5CF6; 
                    padding: 10px 16px; 
                    border-radius: 10px; 
                    color: #EDE9FE; 
                    font-weight: 600; 
                    margin: 6px 0 12px 0;">
            Network: {len(G.nodes)} accounts, {len(G.edges)} connections | {len(set(communities.values()))} communities
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🕸️ Network visualization")
    fig = _viz.create_network_plot(G, features_df, communities)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Top influential accounts")
    top_accounts = features_df.nlargest(8, "pagerank")[
        ["account_id", "pagerank", "total_degree", "in_degree", "out_degree", "total_amount"]
    ].copy()
    top_accounts["pagerank"] = top_accounts["pagerank"].apply(lambda x: f"{x:.4f}")
    top_accounts["total_amount"] = top_accounts["total_amount"].apply(lambda x: f"${x:,.0f}")
    display_compact = top_accounts.rename(columns={
        "account_id": "Account",
        "pagerank": "PageRank",
        "total_degree": "Total degree",
        "in_degree": "In",
        "out_degree": "Out",
        "total_amount": "Total amount",
    })
    st.dataframe(display_compact, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main() 
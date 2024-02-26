import networkx as nx
import pandas as pd

# Load datasets
posts = pd.read_csv("bot-python/data/sampled_posts.csv")
follows = pd.read_csv("bot-python/data/sampled_follows.csv")
likes = pd.read_csv("bot-python/data/sampled_likes.csv")
profiles = pd.read_csv("bot-python/data/sampled_profiles.csv")
reposts = pd.read_csv("bot-python/data/sampled_reposts.csv")

G = nx.Graph()

# Add nodes for each user
for did in profiles['did']:
    G.add_node(did)

# Increment edge weights for likes, reposts, and follows
# Each interaction type can have different weights
for _, row in likes.iterrows():
    G.add_edge(row['did'], row['subject_uri'], weight=0.5)  # Adjust weight as needed

for _, row in reposts.iterrows():
    G.add_edge(row['did'], row['subject_uri'], weight=0.7)  # Adjust weight as needed

for _, row in follows.iterrows():
    G.add_edge(row['did'], row['subject'], weight=1.0)  # Adjust weight as needed
    
import community as community_louvain

import matplotlib.pyplot as plt

# Detect communities
partition = community_louvain.best_partition(G)

# Assign community to each user in the profiles DataFrame
profiles['community'] = profiles['did'].map(partition)

def visualize_communities(graph, partition):
    # Position nodes using the spring layout
    pos = nx.spring_layout(graph)
    
    # Convert community assignment to a list of community IDs
    community_ids = list(partition.values())
    
    # Draw the nodes, coloring them by their community
    cmap = plt.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(graph, pos, node_size=40, cmap=cmap, node_color=community_ids, alpha=0.8)
    
    # Draw the edges
    nx.draw_networkx_edges(graph, pos, alpha=0.1)
    
    plt.show()

# Assuming 'G' is your graph and 'partition' is the community assignment from community detection
visualize_communities(G, partition)




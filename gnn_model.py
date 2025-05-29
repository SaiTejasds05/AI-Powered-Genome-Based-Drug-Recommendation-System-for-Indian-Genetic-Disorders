import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data

class GenomicGNN(nn.Module):
    """
    Graph Neural Network for genomic data analysis.
    Uses gene-gene and gene-drug interactions to predict drug responses
    for specific genetic variants.
    """
    def __init__(self, num_features, hidden_dim=64, num_classes=1, dropout=0.3):
        super(GenomicGNN, self).__init__()
        
        # Graph convolutional layers
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Attention mechanism for gene-drug interactions
        self.attention = GATConv(hidden_dim, hidden_dim, heads=4, dropout=dropout)
        
        # Fully connected layers for prediction
        self.fc1 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph convolutional layers
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        
        # Apply attention mechanism
        x = self.attention(x, edge_index)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Fully connected layers for prediction
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class GenomicFeatureEmbedding(nn.Module):
    """
    Genomic feature embedding model to convert genomic variants
    into a dense vector representation.
    """
    def __init__(self, vocab_size, embedding_dim=64):
        super(GenomicFeatureEmbedding, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, x):
        # Get embeddings
        x = self.embedding(x)
        
        # Mean pooling over sequence
        x = torch.mean(x, dim=1)
        
        # Final transformation
        x = F.relu(self.fc(x))
        
        return x

class DrugResponsePredictor(nn.Module):
    """
    Combined model that uses genomic feature embeddings and GNN
    to predict drug responses.
    """
    def __init__(self, vocab_size, gnn_features, hidden_dim=64, num_classes=1):
        super(DrugResponsePredictor, self).__init__()
        
        self.embedding_model = GenomicFeatureEmbedding(vocab_size, hidden_dim)
        self.gnn_model = GenomicGNN(gnn_features, hidden_dim, hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, variant_ids, graph_data):
        # Get genomic feature embeddings
        variant_embeddings = self.embedding_model(variant_ids)
        
        # Get GNN embeddings
        graph_embeddings = self.gnn_model(graph_data)
        
        # Concatenate embeddings
        combined = torch.cat([variant_embeddings, graph_embeddings], dim=1)
        
        # Final prediction
        x = F.relu(self.fc1(combined))
        x = self.fc2(x)
        
        return x

def build_gene_interaction_graph(gene_list, relationships_df):
    """
    Build a gene interaction graph from the relationships dataframe.
    
    Args:
        gene_list: List of genes to include in the graph
        relationships_df: DataFrame containing gene-gene and gene-drug relationships
    
    Returns:
        nx_graph: NetworkX graph object
        gene_to_idx: Mapping of gene names to node indices
    """
    # Create empty graph
    G = nx.Graph()
    
    # Add genes as nodes
    gene_to_idx = {gene: i for i, gene in enumerate(gene_list)}
    for gene in gene_list:
        G.add_node(gene_to_idx[gene], name=gene, type='gene')
    
    # Filter relationships for genes in our list
    gene_relationships = relationships_df[
        (relationships_df['Entity1_type'] == 'Gene') & 
        (relationships_df['Entity2_type'] == 'Gene')
    ]
    
    # Add edges between genes
    for _, row in gene_relationships.iterrows():
        gene1 = row['Entity1_name']
        gene2 = row['Entity2_name']
        
        if gene1 in gene_to_idx and gene2 in gene_to_idx:
            G.add_edge(gene_to_idx[gene1], gene_to_idx[gene2], 
                       weight=1.0, relation=row['Evidence'])
    
    # Add drug-gene interactions
    drug_gene = relationships_df[
        ((relationships_df['Entity1_type'] == 'Gene') & 
         (relationships_df['Entity2_type'] == 'Chemical')) |
        ((relationships_df['Entity1_type'] == 'Chemical') & 
         (relationships_df['Entity2_type'] == 'Gene'))
    ]
    
    # Add drugs as nodes
    drug_list = set()
    for _, row in drug_gene.iterrows():
        if row['Entity1_type'] == 'Chemical':
            drug_list.add(row['Entity1_name'])
        else:
            drug_list.add(row['Entity2_name'])
    
    drug_to_idx = {drug: i + len(gene_list) for i, drug in enumerate(drug_list)}
    
    for drug in drug_list:
        G.add_node(drug_to_idx[drug], name=drug, type='drug')
    
    # Add gene-drug edges
    for _, row in drug_gene.iterrows():
        gene = row['Entity1_name'] if row['Entity1_type'] == 'Gene' else row['Entity2_name']
        drug = row['Entity2_name'] if row['Entity2_type'] == 'Chemical' else row['Entity1_name']
        
        if gene in gene_to_idx and drug in drug_to_idx:
            weight = 0.5  # Default weight
            
            # Set weight based on association
            if row['Association'] == 'associated':
                weight = 1.0
            elif row['Association'] == 'ambiguous':
                weight = 0.5
            elif row['Association'] == 'not associated':
                weight = 0.1
                
            G.add_edge(gene_to_idx[gene], drug_to_idx[drug], 
                       weight=weight, relation='gene-drug')
    
    return G, gene_to_idx, drug_to_idx

def convert_to_torch_geometric(nx_graph, node_features=None):
    """
    Convert NetworkX graph to PyTorch Geometric data format.
    
    Args:
        nx_graph: NetworkX graph
        node_features: Optional pre-computed node features
        
    Returns:
        PyTorch Geometric Data object
    """
    # Get edge index
    edge_index = []
    edge_attr = []
    
    for u, v, data in nx_graph.edges(data=True):
        edge_index.append([u, v])
        edge_index.append([v, u])  # Add reverse edge for undirected graph
        
        # Add edge attributes/weights
        weight = data.get('weight', 1.0)
        edge_attr.append([weight])
        edge_attr.append([weight])  # Same weight for reverse edge
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # If no node features provided, use one-hot encoding of node type
    if node_features is None:
        num_nodes = nx_graph.number_of_nodes()
        x = torch.zeros((num_nodes, 2), dtype=torch.float)
        
        for node, attrs in nx_graph.nodes(data=True):
            node_type = attrs.get('type', 'gene')
            if node_type == 'gene':
                x[node, 0] = 1.0
            else:  # drug
                x[node, 1] = 1.0
    else:
        x = node_features
    
    # Create PyTorch Geometric data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data
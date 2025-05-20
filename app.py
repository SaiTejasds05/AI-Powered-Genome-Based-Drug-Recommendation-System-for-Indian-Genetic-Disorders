import streamlit as st
import pandas as pd
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import base64
import io
from utils import display_welcome, load_images, add_gradient_background
from data_processor import load_sample_data, process_uploaded_file, get_indian_genetic_disorders_info
from pharmgkb_processor import PharmGKBProcessor
import os

# Set page configuration
st.set_page_config(
    page_title="AI-Powered Genome Based Drug Recommendation System",
    page_icon="🧬",
    layout="wide"
)

# Add gradient background
add_gradient_background()

# Additional CSS to customize the appearance
st.markdown("""
<style>
    body {
        font-family: 'Times New Roman', Times, serif;
    }
    .warning {
        color: #721c24;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 10px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .recommendation {
        margin-bottom: 20px;
        padding: 15px;
        border-radius: 4px;
        background-color: rgba(255, 255, 255, 0.9);
        border-left: 5px solid #4cddff;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .score-high {
        color: #155724;
        font-weight: bold;
    }
    .score-medium {
        color: #856404;
        font-weight: bold;
    }
    .score-low {
        color: #721c24;
        font-weight: bold;
    }
    .header-container {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    .header-text {
        margin-left: 20px;
    }
    .tabs-container {
        margin-top: 20px;
    }
    .tab-content {
        padding: 20px;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 5px;
    }
    .gene-graph {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .population-info {
        background-color: rgba(76, 221, 255, 0.1);
        border-left: 5px solid #4cddff;
        padding: 10px;
        margin: 10px 0;
        border-radius: 0 4px 4px 0;
    }
    .disorder-card {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 5px solid #4cddff;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .model-details {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 5px;
        padding: 10px;
        margin-top: 20px;
        font-size: 0.9em;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .stTabs [data-baseweb="tab"] {
        color: white;
    }
    .stButton > button {
        background-color: #08769b;
        color: white;
        border: none;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #1a2959;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    .stSelectbox [data-baseweb="select"] {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
    }
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .css-18e3th9 {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .css-1kyxreq {
        justify-content: center;
        align-items: center;
    }
    /* Improve readability of expander titles */
    .streamlit-expanderHeader {
        color: white !important;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'analysis_type' not in st.session_state:
    st.session_state.analysis_type = None
if 'pharmgkb_processor' not in st.session_state:
    st.session_state.pharmgkb_processor = None
if 'population' not in st.session_state:
    st.session_state.population = "Indian"
if 'show_graph' not in st.session_state:
    st.session_state.show_graph = False
if 'graph_data' not in st.session_state:
    st.session_state.graph_data = None

# Load pre-fetched images
images = load_images()

def main():
    """Main application function"""
    # Initialize PharmGKB processor if not already done
    if not st.session_state.pharmgkb_processor:
        with st.spinner("Loading genomic databases..."):
            st.session_state.pharmgkb_processor = PharmGKBProcessor()
    
    # Display welcome page or results based on session state
    if not st.session_state.analysis_complete:
        display_welcome_page()
    else:
        display_results_page()

def display_welcome_page():
    """Display the welcome page with upload options"""
    display_welcome(images)
    
    st.markdown("### Upload Genomic Data or Use Sample Data")
    
    # Population selection
    st.markdown("#### Select Population Group")
    population_options = ["Indian", "East Asian", "European", "African", "South Asian", "Other"]
    selected_population = st.selectbox(
        "Select the population group for tailored drug recommendations:",
        population_options,
        index=0
    )
    st.session_state.population = selected_population
    
    if selected_population == "Indian":
        st.markdown("""
        <div class="population-info">
            <p>The analysis will be optimized for Indian genetic variations, accounting for specific allele frequencies and known drug response patterns in Indian populations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Two tabs for selecting data source
    tab1, tab2, tab3 = st.tabs(["Upload Your Data", "Use Sample Data", "Indian Genetic Disorders"])
    
    with tab1:
        st.markdown("Upload a .tsv or .csv file containing genomic data:")
        uploaded_file = st.file_uploader("Choose a file", type=["tsv", "csv"])
        
        if uploaded_file is not None:
            try:
                st.session_state.uploaded_data = process_uploaded_file(uploaded_file)
                st.success("File uploaded successfully!")
                
                # Show a small preview of the data
                st.markdown("#### Data Preview")
                st.dataframe(st.session_state.uploaded_data.head(5), use_container_width=True)
                
                st.session_state.analysis_type = "uploaded"
                st.button("Analyze Genomic Data", 
                          on_click=perform_analysis,
                          key="analyze_uploaded",
                          use_container_width=True)
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab2:
        st.markdown("Select a sample dataset for demonstration:")
        dataset_option = st.selectbox(
            "Choose a dataset",
            ["Clinical Annotations", "Clinical Variants", "Drug-Gene Relationships"],
            index=0
        )
        
        # Show information about the selected dataset
        if dataset_option == "Clinical Annotations":
            st.info("Clinical annotations contain information about gene variants and their relationship to drug response, with evidence levels and phenotype information.")
        elif dataset_option == "Clinical Variants":
            st.info("Clinical variants include specific genetic variations that affect drug metabolism or response, with details on the level of evidence.")
        elif dataset_option == "Drug-Gene Relationships":
            st.info("This dataset shows relationships between drugs and genes, including the type and strength of evidence for these associations.")
        
        advanced_options = st.expander("Advanced Analysis Options")
        with advanced_options:
            # Model selection
            st.markdown("#### Model Selection")
            model_option = st.radio(
                "Select genomic analysis model:",
                ["Graph Neural Network (GNN)", "Gene Ontology Enhanced", "Standard PharmGKB Analysis"],
                index=0
            )
            
            # Visualization options
            st.markdown("#### Visualization Options")
            st.checkbox("Show gene-drug interaction network", key="show_graph")
            
            # Analysis depth
            st.markdown("#### Analysis Depth")
            analysis_depth = st.slider("Analysis depth", 1, 3, 2, 
                                    help="Higher values perform more thorough analysis but take longer")
        
        if st.button("Use Sample Data", key="use_sample", use_container_width=True):
            with st.spinner("Loading sample data..."):
                st.session_state.uploaded_data = load_sample_data(dataset_option)
                st.session_state.analysis_type = "sample"
                st.session_state.dataset_name = dataset_option
                perform_analysis()
    
    with tab3:
        display_indian_genetic_disorders()

def display_indian_genetic_disorders():
    """Display information about common Indian genetic disorders"""
    st.markdown("### Common Indian Genetic Disorders")
    st.markdown("""
    India has a diverse genetic landscape with specific disorders that are more prevalent in certain populations. 
    This information helps contextualize the drug recommendations for Indian genetic profiles.
    """)
    
    disorders = st.session_state.pharmgkb_processor.get_indian_genetic_disorder_info()
    
    # Create two columns for displaying disorders
    col1, col2 = st.columns(2)
    
    # Distribute disorders between columns
    for i, (disorder, info) in enumerate(disorders.items()):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"""
            <div class="disorder-card">
                <h4>{disorder}</h4>
                <p><strong>Prevalence:</strong> {info['prevalence']}</p>
                <p><strong>Genes:</strong> {', '.join(info['genes'])}</p>
                <p><strong>Regions:</strong> {', '.join(info['regions'])}</p>
                <p>{info['description']}</p>
            </div>
            """, unsafe_allow_html=True)

def display_results_page():
    """Display the results page with drug recommendations"""
    st.markdown("# Analysis Results")
    
    if st.session_state.analysis_type == "sample":
        st.info(f"Results based on sample data: {st.session_state.dataset_name}")
    else:
        st.info("Results based on your uploaded genomic data")
    
    # Display population information
    st.markdown(f"### Population Group: {st.session_state.population}")
    
    if st.session_state.population == "Indian":
        st.markdown("""
        <div class="population-info">
            <p>These recommendations have been tailored for Indian genetic profiles, accounting for specific allele frequencies, 
            polymorphisms, and known drug response patterns in Indian populations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Create tabs for different result views
    tabs = st.tabs(["Drug Recommendations", "Network Visualization", "Technical Details"])
    
    with tabs[0]:
        st.markdown("## Personalized Drug Recommendations")
        
        if not st.session_state.recommendations or len(st.session_state.recommendations) == 0:
            st.warning("No significant drug recommendations found based on the provided genomic data.")
            st.markdown("This could be due to:")
            st.markdown("- Limited genetic markers in the provided data")
            st.markdown("- No strong associations between identified variants and drug responses")
            st.markdown("- Need for more comprehensive genomic data")
        else:
            display_recommendations(st.session_state.recommendations)
    
    with tabs[1]:
        st.markdown("## Gene-Drug Interaction Network")
        
        if st.session_state.show_graph and st.session_state.graph_data is not None:
            display_gene_drug_network(st.session_state.graph_data)
        else:
            st.info("Network visualization is not enabled. Please return to the welcome page and select 'Show gene-drug interaction network' in the advanced options to view this.")
    
    with tabs[2]:
        st.markdown("## Technical Analysis Details")
        display_technical_details()
    
    if st.button("Start New Analysis", use_container_width=True):
        reset_session()

def display_recommendations(recommendations):
    """Display drug recommendations in a formatted way"""
    for rec in recommendations:
        st.markdown(f"""
        <div class="recommendation">
            <h3>{rec['drug_name']}</h3>
            <p><strong>Effectiveness Score:</strong> <span class="{get_score_class(rec['effectiveness_score'])}">{rec['effectiveness_score']:.2f}</span> / 10.0</p>
            <p><strong>Gene Interaction:</strong> {rec['gene_interaction']}</p>
            <p><strong>Recommended For:</strong> {rec['phenotype']}</p>
            <p>{rec['interpretation']}</p>
            {get_warning_html(rec) if rec.get('warning') else ''}
        </div>
        """, unsafe_allow_html=True)

def display_gene_drug_network(graph_data):
    """Display gene-drug interaction network visualization"""
    try:
        # Create a NetworkX graph
        G = nx.Graph()
        
        # Add nodes for genes
        for gene in graph_data['genes']:
            G.add_node(gene, type='gene')
        
        # Add nodes for drugs
        for drug in graph_data['drugs']:
            G.add_node(drug, type='drug')
        
        # Add edges for interactions
        for interaction in graph_data['interactions']:
            G.add_edge(interaction['gene'], interaction['drug'], weight=interaction['weight'])
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Define positions using spring layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes with different colors for genes and drugs
        gene_nodes = [node for node in G.nodes() if G.nodes[node].get('type') == 'gene']
        drug_nodes = [node for node in G.nodes() if G.nodes[node].get('type') == 'drug']
        
        nx.draw_networkx_nodes(G, pos, nodelist=gene_nodes, node_color='#4CAF50', node_size=500, alpha=0.8, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=drug_nodes, node_color='#2196F3', node_size=500, alpha=0.8, ax=ax)
        
        # Draw edges with weights affecting width
        edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7, edge_color='#9E9E9E', ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', ax=ax)
        
        # Add legend
        gene_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4CAF50', markersize=15, label='Genes')
        drug_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3', markersize=15, label='Drugs')
        ax.legend(handles=[gene_patch, drug_patch], loc='upper right')
        
        # Remove axis
        ax.axis('off')
        
        # Set title
        plt.title("Gene-Drug Interaction Network", fontsize=16)
        
        # Display plot in Streamlit
        st.pyplot(fig)
        
        # Display explanation
        st.markdown("""
        **Network Interpretation:**
        - **Green nodes** represent genes identified in the genomic data
        - **Blue nodes** represent drugs that interact with these genes
        - **Edge thickness** indicates the strength of the gene-drug interaction
        - This visualization helps understand the relationships between genetic variants and drug responses
        """)
    
    except Exception as e:
        st.error(f"Error generating network visualization: {str(e)}")

def display_technical_details():
    """Display technical details of the analysis process"""
    st.markdown("### Analysis Methodology")
    
    # Technologies used
    st.markdown("#### Technologies Used")
    st.markdown("""
    - **Graph Neural Networks (GNNs)**: Used to model complex relationships between genes, variants, and drugs
    - **Genomic Feature Embedding**: Converts genetic variants into dense vector representations
    - **PharmGKB Database Integration**: Provides clinical annotations and evidence levels for gene-drug relationships
    - **Gene Ontology Analysis**: Enriches genetic data with functional annotations
    - **Reinforcement Learning**: Optimizes drug recommendations based on population-specific responses
    """)
    
    # Analysis process
    st.markdown("#### Analysis Process")
    st.markdown("""
    1. **Data Processing**: Genomic data is processed and relevant variants are identified
    2. **Knowledge Graph Construction**: A graph of gene-drug interactions is built based on PharmGKB data
    3. **Population Adjustment**: Drug responses are adjusted based on known pharmacogenomic differences in the selected population
    4. **Drug Scoring**: Drugs are scored based on their predicted effectiveness for the given genetic profile
    5. **Recommendation Generation**: Final recommendations are generated with interpretations and warnings
    """)
    
    # Model details
    st.markdown("#### Model Details")
    st.markdown("""
    <div class="model-details">
        <p><strong>GNN Architecture:</strong> Graph Convolutional Network with attention mechanism</p>
        <p><strong>Feature Embedding Dimension:</strong> 64</p>
        <p><strong>Population Adjustment Factor:</strong> Based on Indian genomic datasets</p>
        <p><strong>Evidence Weighting:</strong> Prioritizes level 1A evidence with exponential decay for lower levels</p>
    </div>
    """, unsafe_allow_html=True)

def get_score_class(score):
    """Get CSS class based on effectiveness score"""
    if score >= 7.0:
        return "score-high"
    elif score >= 4.0:
        return "score-medium"
    else:
        return "score-low"

def get_warning_html(rec):
    """Generate HTML for warning message"""
    if rec.get('warning'):
        return f'<div class="warning"><strong>⚠️ Warning:</strong> {rec["warning"]}</div>'
    return ''

def perform_analysis():
    """Perform genomic analysis and generate recommendations"""
    with st.spinner("Analyzing genomic data using advanced AI models..."):
        try:
            # Extract gene variants from data
            gene_variants = extract_gene_variants(st.session_state.uploaded_data)
            
            # Get drug recommendations using PharmGKB processor
            recommendations = st.session_state.pharmgkb_processor.get_drug_recommendations(
                gene_variants, 
                population=st.session_state.population
            )
            
            # Generate graph data for visualization if enabled
            if st.session_state.show_graph:
                st.session_state.graph_data = generate_graph_data(gene_variants, recommendations)
            
            # Update session state
            st.session_state.recommendations = recommendations
            st.session_state.analysis_complete = True
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

def extract_gene_variants(data):
    """Extract gene variants from the uploaded data"""
    gene_variants = []
    
    # Check for variant column
    if 'Variant/Haplotypes' in data.columns:
        gene_variants = data['Variant/Haplotypes'].dropna().unique().tolist()
    elif 'variant' in data.columns:
        gene_variants = data['variant'].dropna().unique().tolist()
    
    # If no variants found, try genes
    if not gene_variants:
        if 'Gene' in data.columns:
            gene_variants = data['Gene'].dropna().unique().tolist()
        elif 'gene' in data.columns:
            gene_variants = data['gene'].dropna().unique().tolist()
    
    # For drug-gene relationships
    if 'Entity1_type' in data.columns and 'Entity1_name' in data.columns:
        for _, row in data.iterrows():
            if row['Entity1_type'] == 'Gene':
                gene_variants.append(row['Entity1_name'])
            elif row['Entity2_type'] == 'Gene':
                gene_variants.append(row['Entity2_name'])
    
    # Remove duplicates and non-string values
    gene_variants = list(set([str(v) for v in gene_variants if v and str(v) != 'nan']))
    
    return gene_variants

def generate_graph_data(gene_variants, recommendations):
    """Generate graph data for visualization"""
    # Extract genes and drugs from recommendations
    genes = list(set([rec['gene_interaction'] for rec in recommendations if rec['gene_interaction']]))
    drugs = list(set([rec['drug_name'] for rec in recommendations if rec['drug_name']]))
    
    # Create interactions list
    interactions = []
    for rec in recommendations:
        if rec['gene_interaction'] and rec['drug_name']:
            interactions.append({
                'gene': rec['gene_interaction'],
                'drug': rec['drug_name'],
                'weight': rec['effectiveness_score'] / 10.0  # Normalize to 0-1
            })
    
    return {
        'genes': genes,
        'drugs': drugs,
        'interactions': interactions
    }

def reset_session():
    """Reset session state for new analysis"""
    st.session_state.analysis_complete = False
    st.session_state.recommendations = None
    st.session_state.uploaded_data = None
    st.session_state.analysis_type = None
    st.session_state.graph_data = None
    st.rerun()

if __name__ == "__main__":
    main()

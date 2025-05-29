import streamlit as st
import pandas as pd
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import base64
import io
from utils import display_welcome, load_images, add_gradient_background
from data_processor import load_sample_data, process_uploaded_file, get_indian_genetic_disorders_info, process_pdf_file
from pharmgkb_processor import PharmGKBProcessor
from data_manager import DataManager
from clinical_validator import ClinicalValidator
import os
import streamlit.components.v1 as components
import sqlite3

# Initialize components
data_manager = DataManager()
clinical_validator = ClinicalValidator()

# Set page configuration
st.set_page_config(
    page_title="AI-Powered Genome Based Drug Recommendation System",
    page_icon="🧬",
    layout="wide"
)

def init_admin_db():
    """Initialize the admin database"""
    conn = sqlite3.connect('hospital_data.db')
    c = conn.cursor()
    
    # Create admin credentials table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS admin_credentials
                 (username TEXT PRIMARY KEY,
                  password TEXT)''')
    
    # Insert default admin credentials if they don't exist
    c.execute('SELECT * FROM admin_credentials WHERE username = ?', ('GDAP5',))
    if not c.fetchone():
        c.execute('INSERT INTO admin_credentials (username, password) VALUES (?, ?)',
                 ('GDAP5', 'GDAP512740'))
    
    conn.commit()
    conn.close()

def authenticate_admin(username, password):
    """Authenticate admin user"""
    conn = sqlite3.connect('hospital_data.db')
    c = conn.cursor()
    c.execute('SELECT * FROM admin_credentials WHERE username = ? AND password = ?',
             (username, password))
    result = c.fetchone()
    conn.close()
    return result is not None

def display_admin_login():
    """Display admin login form"""
    with st.sidebar:
        st.markdown("### Admin Access")
        username = st.text_input("Admin Username")
        password = st.text_input("Admin Password", type="password")
        
        if st.button("Access Admin Panel"):
            if authenticate_admin(username, password):
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("Invalid admin credentials")

def display_admin_panel():
    """Display admin panel content"""
    st.title("Admin Dashboard")
    
    # Get data
    conn = sqlite3.connect('hospital_data.db')
    
    # Get patient information
    patients_df = pd.read_sql_query('''
        SELECT p.*, 
               COUNT(g.id) as genetic_data_count,
               COUNT(a.id) as analysis_count
        FROM patients p
        LEFT JOIN genetic_data g ON p.patient_id = g.patient_id
        LEFT JOIN analysis_results a ON p.patient_id = a.patient_id
        GROUP BY p.patient_id
    ''', conn)
    
    # Get genetic data information
    genetic_data_df = pd.read_sql_query('''
        SELECT g.*, p.name as patient_name
        FROM genetic_data g
        JOIN patients p ON g.patient_id = p.patient_id
        ORDER BY g.upload_date DESC
    ''', conn)
    
    # Get analysis results
    analysis_df = pd.read_sql_query('''
        SELECT a.*, p.name as patient_name
        FROM analysis_results a
        JOIN patients p ON a.patient_id = p.patient_id
        ORDER BY a.analysis_date DESC
    ''', conn)
    
    conn.close()
    
    # Display statistics
    st.markdown("### Dashboard Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Patients", len(patients_df))
    with col2:
        st.metric("Total Genetic Data Records", len(genetic_data_df))
    with col3:
        st.metric("Total Analysis Results", len(analysis_df))
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Patient Database", "Genetic Data", "Analysis Results"])
    
    with tab1:
        st.markdown("### Patient Database")
        if not patients_df.empty:
            st.dataframe(patients_df, use_container_width=True)
        else:
            st.info("No patient records found")
    
    with tab2:
        st.markdown("### Genetic Data Records")
        if not genetic_data_df.empty:
            st.dataframe(genetic_data_df, use_container_width=True)
        else:
            st.info("No genetic data records found")
    
    with tab3:
        st.markdown("### Analysis Results")
        if not analysis_df.empty:
            st.dataframe(analysis_df, use_container_width=True)
        else:
            st.info("No analysis results found")
    
    # Add export functionality
    st.markdown("### Export Data")
    export_format = st.selectbox("Select export format", ["CSV", "Excel"])
    
    if st.button("Export All Data"):
        try:
            if export_format == "CSV":
                patients_df.to_csv("patient_database.csv", index=False)
                genetic_data_df.to_csv("genetic_data.csv", index=False)
                analysis_df.to_csv("analysis_results.csv", index=False)
            else:
                with pd.ExcelWriter("admin_database.xlsx") as writer:
                    patients_df.to_excel(writer, sheet_name="Patients", index=False)
                    genetic_data_df.to_excel(writer, sheet_name="Genetic Data", index=False)
                    analysis_df.to_excel(writer, sheet_name="Analysis Results", index=False)
            st.success("Data exported successfully!")
        except Exception as e:
            st.error(f"Error exporting data: {str(e)}")

# Add background image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/jpg;base64,%s");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set the background image
set_background('background.jpg')

# Additional CSS to customize the appearance
st.markdown("""
<style>
    body {
        font-family: 'Times New Roman', Times, serif;
    }
    .warning {
        color: #721c24;
        background-color: rgba(248, 215, 218, 0.9);
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
        color: black !important;
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
    /* Make content containers more visible over the background */
    div[data-testid="stVerticalBlock"] {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 1rem;
        border-radius: 5px;
    }
    div[data-testid="stHorizontalBlock"] {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 1rem;
        border-radius: 5px;
    }
    /* Add black outline to metric boxes */
    .stMetric {
        border: 2px solid #000000;
        border-radius: 5px;
        padding: 10px;
        background-color: rgba(255, 255, 255, 0.9);
    }
    /* Add black outline to tabs */
    .stTabs [data-baseweb="tab-list"] {
        border: 2px solid #000000;
        border-radius: 5px;
        background-color: rgba(255, 255, 255, 0.9);
    }
    /* Add black outline to export section */
    .export-section {
        border: 2px solid #000000;
        border-radius: 5px;
        padding: 15px;
        background-color: rgba(255, 255, 255, 0.9);
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
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
if 'patient_info' not in st.session_state:
    st.session_state.patient_info = None
if 'admin_authenticated' not in st.session_state:
    st.session_state.admin_authenticated = False

# Load pre-fetched images
images = load_images()

def main():
    """Main application function"""
    # Initialize admin database
    init_admin_db()
    
    # Initialize PharmGKB processor if not already done
    if not st.session_state.pharmgkb_processor:
        with st.spinner("Loading genomic databases..."):
            st.session_state.pharmgkb_processor = PharmGKBProcessor()
    
    # Display admin login in sidebar
    display_admin_login()
    
    # If admin is authenticated, show admin panel
    if st.session_state.admin_authenticated:
        display_admin_panel()
        if st.sidebar.button("Exit Admin Panel"):
            st.session_state.admin_authenticated = False
            st.rerun()
    else:
        # Display welcome page or results based on session state
        if not st.session_state.analysis_complete:
            display_welcome_page()
        else:
            display_results_page()

def display_welcome_page():
    """Display the welcome page with upload options"""
    display_welcome(images)
    
    st.markdown("### Upload Genomic Data or Use Sample Data")
    
    # Population selection - fixed to "Indian"
    st.markdown("#### Population Group")
    selected_population = "Indian"
    st.session_state.population = selected_population

    st.markdown("""
    <div class="population-info">
        <p>The analysis is optimized for <strong>Indian genetic variations</strong>, accounting for specific allele frequencies and known drug response patterns in Indian populations.</p>
    </div>
    """, unsafe_allow_html=True)

    # Two tabs for selecting data source
    tab1, tab2, tab3 = st.tabs(["Upload Your Data", "Use Sample Data", "Indian Genetic Disorders"])

    with tab1:
        st.markdown("""
        ### Upload Genomic Data
        Please provide the following information to get personalized drug recommendations:
        """)
        
        # Patient Information
        st.markdown("#### Patient Information")
        patient_id = st.text_input("Patient ID")
        patient_name = st.text_input("Patient Name")
        dob = st.date_input("Date of Birth")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        # Store patient info
        if patient_id and patient_name:
            st.session_state.patient_info = {
                'patient_id': patient_id,
                'name': patient_name,
                'dob': str(dob),
                'gender': gender
            }
        
        # Add genetic disease input field
        genetic_disease = st.text_input("Enter the genetic disease name:", 
                                      help="Please enter the specific genetic disease the patient has been diagnosed with")
        
        st.markdown("""
        #### Upload Genomic Data File
        You can upload your genomic data in any of these formats:
        - **TSV/CSV**: Structured genomic data files
        """)
        
        uploaded_file = st.file_uploader("Choose a file", type=["tsv", "csv"])

        # Add advanced options for uploaded files
        advanced_options = st.expander("Advanced Analysis Options")
        with advanced_options:
            # Model selection
            st.markdown("#### Model Selection")
            model_option = st.radio(
                "Select genomic analysis model:",
                ["Graph Neural Network (GNN)", "Gene Ontology Enhanced", "Standard PharmGKB Analysis"],
                index=0,
                key="upload_model_selection"
            )

            # Visualization options
            st.markdown("#### Visualization Options")
            st.checkbox("Show gene-drug interaction network", key="upload_show_graph")

            # Analysis depth
            st.markdown("#### Analysis Depth")
            analysis_depth = st.slider("Analysis depth", 1, 3, 2, 
                                    help="Higher values perform more thorough analysis but take longer",
                                    key="upload_analysis_depth")
        
        if uploaded_file is not None:
            try:
                # Store the genetic disease in session state
                st.session_state.genetic_disease = genetic_disease
                
                # Process the uploaded file
                st.session_state.uploaded_data = process_uploaded_file(uploaded_file)
                
                # Validate the data
                is_valid, message = data_manager.validate_genetic_data(st.session_state.uploaded_data)
                if not is_valid:
                    st.error(message)
                    return
                
                st.success("File uploaded successfully!")
                
                # Show a small preview of the data
                st.markdown("#### Data Preview")
                st.dataframe(st.session_state.uploaded_data.head(5), use_container_width=True)
                
                st.session_state.analysis_type = "uploaded"
                if st.button("Analyze Genomic Data", 
                          on_click=perform_analysis,
                          key="analyze_uploaded",
                          use_container_width=True):
                    # Store patient data
                    if st.session_state.patient_info:
                        data_manager.add_patient(**st.session_state.patient_info)
                        data_manager.store_genetic_data(
                            st.session_state.patient_info['patient_id'],
                            st.session_state.uploaded_data,
                            'genomic'
                        )
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
                index=0,
                key="sample_model_selection"
            )

            # Visualization options
            st.markdown("#### Visualization Options")
            st.checkbox("Show gene-drug interaction network", key="sample_show_graph")

            # Analysis depth
            st.markdown("#### Analysis Depth")
            analysis_depth = st.slider("Analysis depth", 1, 3, 2, 
                                    help="Higher values perform more thorough analysis but take longer",
                                    key="sample_analysis_depth")

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
            
            # Create genetic profile based on available columns
            genetic_profile = {}
            for _, row in st.session_state.uploaded_data.iterrows():
                if 'Gene' in row and 'Effect' in row:
                    genetic_profile[row['Gene']] = row['Effect']
                elif 'Gene' in row and 'type' in row:
                    genetic_profile[row['Gene']] = row['type']
                elif 'Gene' in row and 'Level of Evidence' in row:
                    genetic_profile[row['Gene']] = row['Level of Evidence']
                elif 'Gene' in row:
                    genetic_profile[row['Gene']] = "Unknown"
            
            # Validate recommendations against clinical guidelines
            validated_recommendations = clinical_validator.validate_recommendations(
                recommendations, 
                genetic_profile
            )
            
            # Generate graph data for visualization if enabled
            show_graph = (st.session_state.analysis_type == "uploaded" and 
                        st.session_state.get("upload_show_graph")) or \
                        (st.session_state.analysis_type == "sample" and 
                        st.session_state.get("sample_show_graph"))
            
            if show_graph:
                st.session_state.graph_data = generate_graph_data(gene_variants, validated_recommendations)
            
            # Store results if patient info is available
            if st.session_state.patient_info:
                data_manager.store_analysis_results(
                    st.session_state.patient_info['patient_id'],
                    validated_recommendations,
                    "system"  # Replace user_id with "system"
                )
            
            # Update session state
            st.session_state.recommendations = validated_recommendations
            st.session_state.analysis_complete = True
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

def display_results_page():
    """Display the results page with drug recommendations"""
    st.markdown("# Analysis Results")
    
    # Display clinical disclaimer
    clinical_validator.display_clinical_disclaimer()
    
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
    tabs = st.tabs(["Drug Recommendations", "Network Visualization", "Technical Details", "Clinical References"])
    
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
        
        show_graph = (st.session_state.analysis_type == "uploaded" and 
                     st.session_state.get("upload_show_graph")) or \
                     (st.session_state.analysis_type == "sample" and 
                     st.session_state.get("sample_show_graph"))
        
        if show_graph and st.session_state.graph_data is not None:
            display_gene_drug_network(st.session_state.graph_data)
        else:
            st.info("Network visualization is not enabled. Please return to the welcome page and select 'Show gene-drug interaction network' in the advanced options to view this.")
    
    with tabs[2]:
        st.markdown("## Technical Analysis Details")
        display_technical_details()
    
    with tabs[3]:
        st.markdown("## Clinical References")
        clinical_validator.display_clinical_references()
    
    # Add export to EMR button if patient info is available
    if st.session_state.patient_info:
        if st.button("Export to EMR", use_container_width=True):
            if data_manager.export_to_emr(
                st.session_state.patient_info['patient_id'],
                st.session_state.recommendations
            ):
                st.success("Results exported to EMR system")
    
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
            {get_clinical_warning_html(rec) if rec.get('clinical_warning') else ''}
        </div>
        """, unsafe_allow_html=True)
        
        # Display dosage recommendations if available
        if st.session_state.patient_info:
            dosage_info = clinical_validator.get_dosage_recommendations(
                rec['drug_name'],
                {row['Gene']: row['Effect'] for _, row in st.session_state.uploaded_data.iterrows()}
            )
            if dosage_info:
                st.markdown(f"""
                <div class="dosage-info">
                    <h4>Dosage Recommendations</h4>
                    <p><strong>Standard Dosage:</strong> {dosage_info['standard_dosage']}</p>
                    <p><strong>Recommended Dosage:</strong> {dosage_info['recommended_dosage']}</p>
                    <p><strong>Monitoring Parameters:</strong></p>
                    <ul>
                        {''.join(f'<li>{param}</li>' for param in dosage_info['monitoring_parameters'])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)

def get_clinical_warning_html(rec):
    """Generate HTML for clinical warning message"""
    if rec.get('clinical_warning'):
        return f'<div class="warning"><strong>⚠️ Clinical Warning:</strong> {rec["clinical_warning"]}</div>'
    return ''

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
    # Reset the show_graph states
    if "upload_show_graph" in st.session_state:
        st.session_state.upload_show_graph = False
    if "sample_show_graph" in st.session_state:
        st.session_state.sample_show_graph = False
    st.rerun()

if __name__ == "__main__":
    main()

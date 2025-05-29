import streamlit as st
import base64
import io
import os
from PIL import Image
from assets.serve_images import load_local_images

def display_welcome(images):
    """Display the welcome header and introduction."""
    # Header with logo and title
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Show the CRISPR image beside the title, enlarged to 2.5x
        if "crispr_technology" in images and images["crispr_technology"] is not None:
            st.image(images["crispr_technology"], width=300)
    
    with col2:
        st.title("AI-Powered Genome Based Drug Recommendation System for Indian Genetic Disorders")
        st.markdown(
            "Upload genomic data to get personalized drug suggestions using AI & genomic analysis."
        )
        
    # Display genetic diseases image prominently near the title (optional, can be removed for even smaller length)
    # if "genetic_diseases" in images and images["genetic_diseases"] is not None:
    #     st.image(images["genetic_diseases"], caption="New technology could safely treat hundreds of genetic diseases")
    
    st.markdown("---")
    
    # Main content with image grid
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### How It Works
        
        1. Upload your genomic data file (.tsv or .csv)
        2. Our AI analyzes genetic variants and mutations
        3. Get personalized drug recommendations based on your genetic profile
        4. Review potential gene-drug interactions and warnings
        
        This system specializes in analyzing genetic markers relevant to Indian genetic disorders.
        """)
    
    with col2:
        st.image("how_it_works.jpg", caption="AI-Powered Genomic Analysis System")
    
    # Remove the CRISPR technology image from below 'How it works'
    # if "crispr_technology" in images and images["crispr_technology"] is not None:
    #     st.image(images["crispr_technology"], caption="CRISPR Technology in Genomic Analysis")
    
    st.markdown("---")

def load_images():
    """Load and return images for the application."""
    # Try to load local images first
    local_images = load_local_images()
    
    # If local images not available or load fails, use fallback online images
    if not local_images or all(img is None for img in local_images.values()):
        fallback_images = {
            "dna_sequence": "https://img.freepik.com/free-vector/realistic-dna-spiral-genetic-code-structure_1284-30158.jpg",
            "gene_network": "https://img.freepik.com/free-vector/scientific-medical-dna-structure-system-network-mesh_1017-26432.jpg",
            "dna_logo": "https://img.freepik.com/free-vector/molecular-medicine-abstract-concept-illustration_335657-3891.jpg",
            "genetics_lab": "https://img.freepik.com/free-photo/scientist-working-laboratory_23-2148925977.jpg",
            "indian_population": "https://img.freepik.com/free-photo/group-indian-medical-students-university_23-2149013574.jpg",
            "system_diagram": "https://img.freepik.com/free-vector/scientific-medical-dna-structure-system-network-mesh_1017-26432.jpg",
            "crispr_technology": "https://img.freepik.com/free-vector/dna-structure-blue-background-genetic-engineering-science-concept_1017-33167.jpg",
            "genetic_diseases": "https://img.freepik.com/free-vector/dna-research-abstract-concept-vector-illustration-genetic-engineering-dna-research-laboratory-genomic-investigation-biotechnology-development-human-genome-project-scientists-abstract-metaphor_335657-4290.jpg"
        }
        return fallback_images
    
    return local_images

def create_svg_logo():
    """Create an SVG logo for the application."""
    svg_code = '''
    <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="bg-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#08769b;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#1a2959;stop-opacity:1" />
            </linearGradient>
        </defs>
        <rect width="100" height="100" fill="url(#bg-gradient)"/>
        <path d="M25,40 L75,40" stroke="#ffffff" stroke-width="3"/>
        <path d="M25,50 L75,50" stroke="#ffffff" stroke-width="3"/>
        <path d="M25,60 L75,60" stroke="#ffffff" stroke-width="3"/>
        
        <!-- DNA double helix representation -->
        <path d="M30,30 C40,35 60,45 70,30" stroke="#4cddff" stroke-width="2" fill="none"/>
        <path d="M30,70 C40,65 60,55 70,70" stroke="#4cddff" stroke-width="2" fill="none"/>
        
        <!-- Connecting lines -->
        <line x1="30" y1="30" x2="30" y2="70" stroke="#ffffff" stroke-width="1.5" stroke-dasharray="2,2"/>
        <line x1="70" y1="30" x2="70" y2="70" stroke="#ffffff" stroke-width="1.5" stroke-dasharray="2,2"/>
        
        <!-- Base pairs -->
        <circle cx="30" cy="30" r="2" fill="#4cddff"/>
        <circle cx="70" cy="30" r="2" fill="#4cddff"/>
        <circle cx="30" cy="70" r="2" fill="#4cddff"/>
        <circle cx="70" cy="70" r="2" fill="#4cddff"/>
    </svg>
    '''
    return svg_code

def add_gradient_background():
    """Add a gradient background to the app."""
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #08769b 0%, #1a2959 100%);
            color: white;
        }
        .stTabs [data-baseweb="tab-list"] {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 5px;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(255, 255, 255, 0.2);
        }
        .stMarkdown, .stText {
            color: white;
        }
        h1, h2, h3, h4, h5, h6 {
            color: white !important;
        }
        .recommendation, .disorder-card, .model-details {
            background-color: rgba(255, 255, 255, 0.9);
            color: #333333;
        }
        .recommendation p, .disorder-card p, .model-details p {
            color: #333333;
        }
        .population-info {
            background-color: rgba(255, 255, 255, 0.2);
            border-left: 5px solid #4cddff;
        }
        .stAlert {
            background-color: rgba(255, 255, 255, 0.9);
            color: #333333;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

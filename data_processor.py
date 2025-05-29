import pandas as pd
import numpy as np
import io
import streamlit as st
import os
import networkx as nx
#import PyPDF2  # Temporarily disabled for rollback
import re

def process_uploaded_file(uploaded_file):
    """Process the uploaded genomic data file."""
    try:
        # Check file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension == 'tsv':
            df = pd.read_csv(uploaded_file, sep='\t')
        else:
            raise ValueError("Unsupported file format. Please upload a .csv or .tsv file.")
        
        # Minimal validation of the data
        if df.empty:
            raise ValueError("The uploaded file contains no data.")
        
        # Perform basic data preprocessing
        df = preprocess_genomic_data(df)
        
        return df
    except Exception as e:
        raise Exception(f"Error processing the file: {str(e)}")

def preprocess_genomic_data(df):
    """Perform basic preprocessing on genomic data."""
    # Standardize column names for easier access
    df.columns = [col.strip() for col in df.columns]
    
    # Handle missing values in important columns
    for col in df.columns:
        # For gene, variant columns, fill with "Unknown"
        if any(term in col.lower() for term in ['gene', 'variant', 'haplotype']):
            df[col] = df[col].fillna('Unknown')
        
        # For score columns, fill with 0
        if any(term in col.lower() for term in ['score', 'weight']):
            df[col] = df[col].fillna(0)
    
    return df

def load_sample_data(dataset_option):
    """Load sample data for demonstration."""
    try:
        if dataset_option == "Clinical Annotations":
            df = pd.read_csv("attached_assets/clinical_annotations.tsv", sep='\t')
            # Extract a diverse subset for faster demonstration
            if len(df) > 100:
                # First, add key drugs for Indian populations
                key_genes = ['CYP2C9', 'CYP2D6', 'G6PD', 'TPMT', 'VKORC1', 'SLCO1B1']
                subset_indices = []
                for gene in key_genes:
                    gene_indices = df[df['Gene'] == gene].index.tolist()
                    if gene_indices:
                        subset_indices.extend(gene_indices[:5])  # Take up to 5 examples of each gene
                
                # If we have less than 50 samples, add random samples to reach at least 50
                if len(subset_indices) < 50:
                    additional_indices = np.random.choice(
                        df.index.difference(subset_indices), 
                        min(50 - len(subset_indices), len(df) - len(subset_indices)),
                        replace=False
                    )
                    subset_indices.extend(additional_indices)
                
                df = df.loc[subset_indices]
            
            return df
            
        elif dataset_option == "Clinical Variants":
            df = pd.read_csv("attached_assets/clinicalVariants.tsv", sep='\t')
            
            # Extract a diverse subset for faster demonstration
            if len(df) > 100:
                # Focus on high evidence levels and diverse gene set
                high_evidence = df[df['level of evidence'] == '1A']
                other_evidence = df[df['level of evidence'] != '1A']
                
                # Take all high evidence records (up to 30) and a sample of others
                high_sample = high_evidence.sample(min(30, len(high_evidence)))
                other_sample = other_evidence.sample(min(70, len(other_evidence)))
                
                df = pd.concat([high_sample, other_sample])
                
            return df
            
        elif dataset_option == "Drug-Gene Relationships":
            df = pd.read_csv("attached_assets/relationships.tsv", sep='\t')
            
            # Filter for gene-drug relationships only
            gene_drug = df[
                ((df['Entity1_type'] == 'Gene') & (df['Entity2_type'] == 'Chemical')) |
                ((df['Entity1_type'] == 'Chemical') & (df['Entity2_type'] == 'Gene'))
            ]
            
            # Sample a subset for demonstration
            if len(gene_drug) > 150:
                df = gene_drug.sample(150)
            else:
                df = gene_drug
                
            return df
        else:
            raise ValueError(f"Unknown dataset option: {dataset_option}")
            
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        # Provide fallback data if the file can't be loaded
        return create_fallback_sample_data(dataset_option)

def create_fallback_sample_data(dataset_option):
    """Create sample data if loading from file fails."""
    if dataset_option == "Clinical Annotations":
        return pd.DataFrame({
            "Clinical Annotation ID": ["655385012", "655385033", "655385102", "655385307", "655386244"],
            "Variant/Haplotypes": ["rs9923231", "rs1136201", "rs1042713", "rs1801133", "rs1045642"],
            "Gene": ["VKORC1", "ERBB2", "ADRB2", "MTHFR", "ABCB1"],
            "Level of Evidence": ["1A", "3", "2A", "2A", "3"],
            "Score": [484.375, 2.25, 11.75, 12.75, 6.25],
            "Phenotype Category": ["Dosage", "Toxicity", "Efficacy", "Toxicity", "Toxicity"],
            "Drug(s)": ["warfarin", "trastuzumab", "salmeterol", "methotrexate", "nevirapine"],
            "Phenotype(s)": ["", "Breast Neoplasms;Drug Toxicity", "Asthma", "Arthritis;Toxicity", "HIV;Toxic liver disease"]
        })
    elif dataset_option == "Clinical Variants":
        return pd.DataFrame({
            "variant": ["rs9923231", "rs1136201", "rs1042713", "rs1801133", "rs1045642", "rs116855232", "rs1800497"],
            "gene": ["VKORC1", "ERBB2", "ADRB2", "MTHFR", "ABCB1", "NUDT15", "ANKK1"],
            "type": ["Dosage", "Toxicity", "Efficacy", "Toxicity", "Toxicity", "Dosage", "Efficacy"],
            "level of evidence": ["1A", "3", "2A", "2A", "3", "1A", "4"],
            "chemicals": ["warfarin", "trastuzumab", "salmeterol", "methotrexate", "nevirapine", "mercaptopurine", "risperidone"],
            "phenotypes": ["", "Breast Neoplasms", "Asthma", "Toxicity", "HIV", "", "Autism;Schizophrenia"]
        })
    else:
        # Expanded set of relationships with Indian-relevant genes and drugs
        return pd.DataFrame({
            "Entity1_id": ["PA134865839", "PA166155579", "PA131", "PA134", "PA227", "PA31744", "PA27", "PA267", "PA108"],
            "Entity1_name": ["SLCO1B1", "rs4149056", "CYP3A5", "CYP2C9", "LDLR", "NQO1", "ABCB1", "ABCB1", "CETP"],
            "Entity1_type": ["Gene", "Variant", "Gene", "Gene", "Gene", "Gene", "Gene", "Gene", "Gene"],
            "Entity2_id": ["PA451089", "PA451089", "PA451089", "PA449957", "PA134850", "PA128406956", "PA451089", "PA450947", "PA450821"],
            "Entity2_name": ["pravastatin", "pravastatin", "pravastatin", "ibuprofen", "simvastatin", "fluorouracil", "pravastatin", "phenytoin", "rosuvastatin"],
            "Entity2_type": ["Chemical", "Chemical", "Chemical", "Chemical", "Chemical", "Chemical", "Chemical", "Chemical", "Chemical"],
            "Evidence": ["ClinicalAnnotation", "VariantAnnotation", "GuidelineAnnotation", "Pathway", "Pathway", "Literature", "MultilinkAnnotation", "ClinicalAnnotation", "ClinicalAnnotation"],
            "Association": ["ambiguous", "ambiguous", "associated", "associated", "associated", "associated", "associated", "associated", "ambiguous"]
        })

def get_indian_genetic_disorders_info():
    """Get information about common Indian genetic disorders."""
    return {
        "Beta-thalassemia": {
            "prevalence": "High in certain communities",
            "genes": ["HBB"],
            "regions": ["Gujarat", "Maharashtra", "West Bengal"],
            "description": "Beta-thalassemia is a blood disorder that reduces the production of hemoglobin.",
            "common_variants": ["c.92+5G>C", "c.92+1G>T", "c.124_127delTTCT"]
        },
        "Sickle Cell Disease": {
            "prevalence": "Common in tribal populations",
            "genes": ["HBB"],
            "regions": ["Central India", "Maharashtra", "Gujarat"],
            "description": "Sickle cell disease is a group of blood disorders typically inherited which results in abnormal hemoglobin molecules.",
            "common_variants": ["rs334"]
        },
        "Glucose-6-phosphate dehydrogenase deficiency": {
            "prevalence": "3-15% in various populations",
            "genes": ["G6PD"],
            "regions": ["Punjab", "Tamil Nadu", "Andhra Pradesh"],
            "description": "G6PD deficiency is a genetic disorder that causes red blood cells to break down in response to certain medications, infections, or foods.",
            "common_variants": ["rs1050828", "rs1050829"]
        },
        "Cystic Fibrosis": {
            "prevalence": "1 in 40,000-100,000",
            "genes": ["CFTR"],
            "regions": ["Pan-Indian"],
            "description": "Cystic fibrosis is a genetic disorder that affects mostly the lungs, but also the pancreas, liver, kidneys, and intestine.",
            "common_variants": ["c.1521_1523delCTT", "p.Arg1162Ter"]
        },
        "Duchenne Muscular Dystrophy": {
            "prevalence": "1 in 3,500 male births",
            "genes": ["DMD"],
            "regions": ["Pan-Indian"],
            "description": "Duchenne muscular dystrophy is a genetic disorder characterized by progressive muscle degeneration and weakness.",
            "common_variants": ["exon deletions/duplications"]
        },
        "Spinal Muscular Atrophy": {
            "prevalence": "1 in 10,000",
            "genes": ["SMN1"],
            "regions": ["Pan-Indian"],
            "description": "Spinal muscular atrophy is a genetic disorder characterized by weakness and wasting (atrophy) in muscles.",
            "common_variants": ["SMN1 deletion"]
        },
        "Phenylketonuria": {
            "prevalence": "Rare in India, 1 in 100,000",
            "genes": ["PAH"],
            "regions": ["More common in North India"],
            "description": "Phenylketonuria is an inborn error of metabolism that results in decreased metabolism of the amino acid phenylalanine.",
            "common_variants": ["R408W", "R261Q"]
        },
        "Wilson's Disease": {
            "prevalence": "Higher than global average, 1 in 30,000",
            "genes": ["ATP7B"],
            "regions": ["South India"],
            "description": "Wilson's disease is a genetic disorder in which copper builds up in the body, particularly in the liver and brain.",
            "common_variants": ["c.813C>A", "c.1708-1G>A"]
        }
    }
    
def get_indian_pharmacogenomic_info():
    """Get information about pharmacogenomic variations specific to Indian populations."""
    return {
        "CYP2C9": {
            "frequency": "CYP2C9*2 (8-10%), CYP2C9*3 (10-15%)",
            "relevance": "Affects metabolism of warfarin, phenytoin, NSAIDs",
            "description": "CYP2C9 variants are more common in Indian populations compared to East Asians, affecting dosing of several medications."
        },
        "CYP2D6": {
            "frequency": "CYP2D6*4 (7-10%), CYP2D6*10 (10-20%)",
            "relevance": "Affects metabolism of many antidepressants, antipsychotics, and codeine",
            "description": "Intermediate metabolizer phenotypes are common in Indian populations."
        },
        "SLCO1B1": {
            "frequency": "SLCO1B1*5 (15-20%)",
            "relevance": "Affects statin transport and risk of myopathy",
            "description": "Higher frequency in Indian populations leads to increased risk of statin-induced myopathy."
        },
        "VKORC1": {
            "frequency": "VKORC1*2 (15-20%)",
            "relevance": "Affects warfarin sensitivity",
            "description": "Variants affect vitamin K recycling and warfarin dosing requirements."
        },
        "G6PD": {
            "frequency": "Mediterranean variant (5-15%)",
            "relevance": "Affects response to antimalarials, sulfonamides, and certain antibiotics",
            "description": "Higher prevalence in certain regions leads to risk of hemolytic anemia with specific medications."
        },
        "HLA-B*15:02": {
            "frequency": "4-8% in Indian populations",
            "relevance": "Associated with severe skin reactions to carbamazepine and other anticonvulsants",
            "description": "Important for screening before prescribing certain anticonvulsants."
        },
        "TPMT": {
            "frequency": "TPMT*3C (1-2%)",
            "relevance": "Affects metabolism of thiopurines (azathioprine, mercaptopurine)",
            "description": "Lower frequency of deficiency alleles compared to Western populations."
        }
    }

def build_gene_drug_network(genomic_data):
    """
    Build a simple gene-drug network from genomic data.
    
    Args:
        genomic_data: DataFrame containing genomic data
        
    Returns:
        NetworkX graph object
    """
    G = nx.Graph()
    
    # Determine data format
    if 'Gene' in genomic_data.columns and 'Drug(s)' in genomic_data.columns:
        # Clinical annotations format
        for _, row in genomic_data.iterrows():
            gene = row['Gene']
            drugs = str(row['Drug(s)']).split(';')
            
            # Add gene node
            if not G.has_node(gene):
                G.add_node(gene, type='gene')
            
            # Add drug nodes and edges
            for drug in drugs:
                drug = drug.strip()
                if not drug:
                    continue
                    
                if not G.has_node(drug):
                    G.add_node(drug, type='drug')
                    
                # Get weight from score if available
                weight = 0.5  # Default weight
                if 'Score' in row:
                    try:
                        weight = float(row['Score']) / 10.0  # Normalize to 0-1
                        weight = min(1.0, max(0.1, weight))  # Bound between 0.1-1.0
                    except (ValueError, TypeError):
                        pass
                        
                G.add_edge(gene, drug, weight=weight)
                
    elif 'gene' in genomic_data.columns and 'chemicals' in genomic_data.columns:
        # Clinical variants format
        for _, row in genomic_data.iterrows():
            gene = row['gene']
            chemicals = str(row['chemicals']).split(';')
            
            # Add gene node
            if not G.has_node(gene):
                G.add_node(gene, type='gene')
            
            # Add drug nodes and edges
            for drug in chemicals:
                drug = drug.strip()
                if not drug:
                    continue
                    
                if not G.has_node(drug):
                    G.add_node(drug, type='drug')
                    
                # Calculate weight based on evidence level
                weight = 0.5  # Default weight
                if 'level of evidence' in row:
                    level = str(row['level of evidence']).strip().upper()
                    if level in ['1A', '1']:
                        weight = 1.0
                    elif level in ['1B', '2A']:
                        weight = 0.8
                    elif level in ['2B', '2']:
                        weight = 0.6
                    elif level in ['3']:
                        weight = 0.4
                    elif level in ['4']:
                        weight = 0.2
                        
                G.add_edge(gene, drug, weight=weight)
                
    elif 'Entity1_type' in genomic_data.columns and 'Entity2_type' in genomic_data.columns:
        # Relationships format
        for _, row in genomic_data.iterrows():
            entity1_type = str(row['Entity1_type']).lower()
            entity2_type = str(row['Entity2_type']).lower()
            
            # Look for gene-drug relationships
            if (entity1_type == 'gene' and entity2_type == 'chemical') or \
               (entity2_type == 'gene' and entity1_type == 'chemical'):
                
                # Determine which entity is the gene and which is the drug
                if entity1_type == 'gene':
                    gene = row['Entity1_name']
                    drug = row['Entity2_name']
                else:
                    gene = row['Entity2_name']
                    drug = row['Entity1_name']
                    
                # Add nodes
                if not G.has_node(gene):
                    G.add_node(gene, type='gene')
                    
                if not G.has_node(drug):
                    G.add_node(drug, type='drug')
                    
                # Calculate weight based on association
                weight = 0.5  # Default weight
                association = str(row.get('Association', '')).lower()
                if association == 'associated':
                    weight = 0.8
                elif association == 'ambiguous':
                    weight = 0.5
                elif association == 'not associated':
                    weight = 0.2
                    
                G.add_edge(gene, drug, weight=weight)
    
    return G

# Commenting out the process_pdf_file function to disable PDF support
def process_pdf_file(uploaded_file):
    raise NotImplementedError("PDF processing is temporarily disabled.")

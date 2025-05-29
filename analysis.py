import pandas as pd
import numpy as np
import re

def analyze_data(data):
    """Analyze the genomic data to identify relevant variants and genes."""
    if data is None or data.empty:
        return None
    
    # Try to determine what kind of data we're working with
    data_type = detect_data_type(data)
    
    if data_type == "clinical_annotations":
        processed_data = process_clinical_annotations(data)
    elif data_type == "clinical_variants":
        processed_data = process_clinical_variants(data)
    elif data_type == "relationships":
        processed_data = process_relationships(data)
    else:
        # Generic processing for unknown data format
        processed_data = process_generic_data(data)
    
    return processed_data

def detect_data_type(data):
    """Detect the type of genomic data provided."""
    columns = set(data.columns.str.lower())
    
    if {'clinical annotation id', 'variant/haplotypes', 'gene'}.issubset(columns) or \
       {'variant/haplotypes', 'gene', 'level of evidence'}.issubset(columns):
        return "clinical_annotations"
    
    if {'variant', 'gene', 'type', 'level of evidence', 'chemicals'}.issubset(columns):
        return "clinical_variants"
    
    if {'entity1_id', 'entity1_name', 'entity2_id', 'entity2_name', 'evidence'}.issubset(columns):
        return "relationships"
    
    return "unknown"

def process_clinical_annotations(data):
    """Process clinical annotations data."""
    # Ensure required columns exist
    required_columns = ['Gene', 'Drug(s)', 'Score', 'Phenotype(s)', 'Level of Evidence', 'Variant/Haplotypes']
    
    # Use standardized column names
    data.columns = [col.strip() for col in data.columns]
    
    # Check if required columns exist (case-insensitive)
    for col in required_columns:
        if not any(existing_col.lower() == col.lower() for existing_col in data.columns):
            # If the column doesn't exist, try to find an alternative or create a placeholder
            if col == 'Score':
                # If Score is missing, try to derive it or set a default
                data['Score'] = 0.0
            elif col == 'Phenotype(s)':
                data['Phenotype(s)'] = ""
            elif col == 'Drug(s)':
                data['Drug(s)'] = "Unknown"
            else:
                data[col] = "Unknown"
    
    # Extract relevant information
    result = []
    for _, row in data.iterrows():
        # Get the drug(s)
        drugs = str(row.get('Drug(s)', '')).split(';')
        
        # Get phenotypes
        phenotypes = str(row.get('Phenotype(s)', '')).split(';')
        phenotype = phenotypes[0] if phenotypes and phenotypes[0] else "Not specified"
        
        # Calculate normalized score (0-10 scale)
        try:
            score = float(row.get('Score', 0))
            # Normalize score to 0-10 scale
            normalized_score = min(10, max(0, score / 50 * 10))
        except (ValueError, TypeError):
            normalized_score = 5.0  # Default middle score
        
        # Get level of evidence
        level = str(row.get('Level of Evidence', '')).strip()
        
        # Get gene and variant
        gene = str(row.get('Gene', '')).strip()
        variant = str(row.get('Variant/Haplotypes', '')).strip()
        
        # Process each drug
        for drug in drugs:
            if drug:
                drug = drug.strip()
                result.append({
                    'drug': drug,
                    'gene': gene,
                    'variant': variant,
                    'score': normalized_score,
                    'level': level,
                    'phenotype': phenotype,
                })
    
    return result

def process_clinical_variants(data):
    """Process clinical variants data."""
    # Process each variant
    result = []
    for _, row in data.iterrows():
        # Get drug/chemical information
        chemicals = str(row.get('chemicals', '')).split(';')
        
        # Get phenotype information
        phenotypes = str(row.get('phenotypes', '')).split(';')
        phenotype = phenotypes[0] if phenotypes and phenotypes[0] else "Not specified"
        
        # Get evidence level
        level = str(row.get('level of evidence', '')).strip()
        
        # Calculate a score based on evidence level
        score = calculate_score_from_level(level)
        
        # Get gene and variant
        gene = str(row.get('gene', '')).strip()
        variant = str(row.get('variant', '')).strip()
        
        # Process each chemical/drug
        for chemical in chemicals:
            if chemical:
                chemical = chemical.strip()
                result.append({
                    'drug': chemical,
                    'gene': gene,
                    'variant': variant,
                    'score': score,
                    'level': level,
                    'phenotype': phenotype,
                })
    
    return result

def process_relationships(data):
    """Process relationships data."""
    # Focus on gene-drug relationships
    result = []
    
    for _, row in data.iterrows():
        entity1_type = str(row.get('Entity1_type', '')).strip().lower()
        entity2_type = str(row.get('Entity2_type', '')).strip().lower()
        
        # Look for gene-drug or variant-drug relationships
        if (entity1_type in ['gene', 'variant'] and entity2_type == 'chemical') or \
           (entity2_type in ['gene', 'variant'] and entity1_type == 'chemical'):
            
            # Determine which entity is the drug and which is the gene/variant
            if entity1_type == 'chemical':
                drug = str(row.get('Entity1_name', '')).strip()
                gene_or_variant = str(row.get('Entity2_name', '')).strip()
                entity_type = entity2_type
            else:
                drug = str(row.get('Entity2_name', '')).strip()
                gene_or_variant = str(row.get('Entity1_name', '')).strip()
                entity_type = entity1_type
            
            # Get association type and evidence
            association = str(row.get('Association', '')).strip().lower()
            evidence = str(row.get('Evidence', '')).strip()
            
            # Calculate a score based on association and evidence
            if association == 'ambiguous':
                score = 5.0
            elif association == 'associated':
                score = 7.5
            elif association == 'not associated':
                score = 2.5
            else:
                score = 5.0
            
            result.append({
                'drug': drug,
                'gene': gene_or_variant if entity_type == 'gene' else '',
                'variant': gene_or_variant if entity_type == 'variant' else '',
                'score': score,
                'level': evidence,
                'phenotype': "Not specified",
            })
    
    return result

def process_generic_data(data):
    """Process generic genomic data when the format is unknown."""
    result = []
    
    # Try to identify columns related to genes, variants, and drugs
    columns = data.columns.str.lower()
    
    # Look for gene-related columns
    gene_cols = [col for col in columns if 'gene' in col]
    
    # Look for variant-related columns
    variant_cols = [col for col in columns if any(term in col for term in ['variant', 'snp', 'rs', 'allele', 'mutation'])]
    
    # Look for drug-related columns
    drug_cols = [col for col in columns if any(term in col for term in ['drug', 'medication', 'treatment', 'chemical'])]
    
    # If we found relevant columns, extract the data
    if gene_cols or variant_cols:
        for _, row in data.iterrows():
            # Extract gene information
            gene = ""
            if gene_cols:
                gene = str(row[data.columns[columns == gene_cols[0]].tolist()[0]])
            
            # Extract variant information
            variant = ""
            if variant_cols:
                variant = str(row[data.columns[columns == variant_cols[0]].tolist()[0]])
            
            # Extract drug information
            drug = "Unknown"
            if drug_cols:
                drug = str(row[data.columns[columns == drug_cols[0]].tolist()[0]])
            
            result.append({
                'drug': drug,
                'gene': gene,
                'variant': variant,
                'score': 5.0,  # Default score
                'level': "",
                'phenotype': "Not specified",
            })
    
    return result

def calculate_score_from_level(level):
    """Calculate a score based on the level of evidence."""
    level = str(level).strip().upper()
    
    # Map evidence levels to scores
    if level in ['1A', '1']:
        return 10.0
    elif level in ['1B', '2A']:
        return 8.5
    elif level in ['2B', '2']:
        return 7.0
    elif level in ['3']:
        return 5.5
    elif level in ['4']:
        return 4.0
    else:
        return 5.0  # Default middle score

def get_drug_recommendations(processed_data):
    """Generate drug recommendations based on processed genomic data."""
    if not processed_data:
        return []
    
    # Group by drug and aggregate information
    drug_data = {}
    for item in processed_data:
        drug = item['drug']
        if drug not in drug_data:
            drug_data[drug] = {
                'scores': [],
                'genes': set(),
                'variants': set(),
                'phenotypes': set(),
                'levels': set()
            }
        
        drug_data[drug]['scores'].append(item['score'])
        if item['gene']:
            drug_data[drug]['genes'].add(item['gene'])
        if item['variant']:
            drug_data[drug]['variants'].add(item['variant'])
        if item['phenotype'] and item['phenotype'] != "Not specified":
            drug_data[drug]['phenotypes'].add(item['phenotype'])
        if item['level']:
            drug_data[drug]['levels'].add(item['level'])
    
    # Generate recommendations
    recommendations = []
    for drug, info in drug_data.items():
        # Calculate average score
        avg_score = sum(info['scores']) / len(info['scores']) if info['scores'] else 5.0
        
        # Skip drugs with very low scores
        if avg_score < 3.0:
            continue
        
        # Get gene interactions
        gene_interaction = ", ".join(info['genes']) if info['genes'] else "Unknown"
        
        # Get phenotype (condition)
        phenotype = ", ".join(info['phenotypes']) if info['phenotypes'] else "Not specified"
        
        # Get evidence levels
        levels = ", ".join(info['levels']) if info['levels'] else "Unknown"
        
        # Generate interpretation
        interpretation = generate_interpretation(drug, info['genes'], info['variants'], avg_score, levels)
        
        # Check for warnings
        warning = generate_warning(drug, info['genes'], avg_score)
        
        recommendations.append({
            'drug_name': drug,
            'effectiveness_score': avg_score,
            'gene_interaction': gene_interaction,
            'phenotype': phenotype,
            'interpretation': interpretation,
            'warning': warning if warning else None
        })
    
    # Sort recommendations by score (highest first)
    recommendations.sort(key=lambda x: x['effectiveness_score'], reverse=True)
    
    # Limit to top 5 recommendations
    return recommendations[:5]

def generate_interpretation(drug, genes, variants, score, levels):
    """Generate an interpretation for the drug recommendation."""
    drug_name = drug.capitalize()
    
    # Base interpretation on score
    if score >= 8.0:
        effectiveness = "highly effective"
    elif score >= 6.0:
        effectiveness = "moderately effective"
    elif score >= 4.0:
        effectiveness = "somewhat effective"
    else:
        effectiveness = "potentially effective"
    
    # Create the basic interpretation
    if genes:
        gene_list = ", ".join(genes)
        interpretation = f"{drug_name} is {effectiveness} based on your genetic profile involving the {gene_list} gene(s)."
    else:
        interpretation = f"{drug_name} is {effectiveness} based on your genetic profile."
    
    # Add variant information if available
    if variants:
        variant_count = len(variants)
        if variant_count <= 3:
            variant_list = ", ".join(variants)
            interpretation += f" Specific variants identified: {variant_list}."
        else:
            interpretation += f" Multiple relevant genetic variants were identified."
    
    # Add evidence level information
    if levels and levels != "Unknown":
        interpretation += f" This recommendation is based on evidence level {levels}."
    
    return interpretation

def generate_warning(drug, genes, score):
    """Generate warnings for potential drug risks based on genetic factors."""
    # Check for known drug-gene interactions that require warnings
    high_risk_combinations = {
        "warfarin": ["VKORC1", "CYP2C9"],
        "clopidogrel": ["CYP2C19"],
        "abacavir": ["HLA-B"],
        "carbamazepine": ["HLA-B"],
        "simvastatin": ["SLCO1B1"],
        "codeine": ["CYP2D6"],
        "azathioprine": ["TPMT", "NUDT15"],
        "mercaptopurine": ["TPMT", "NUDT15"],
        "fluorouracil": ["DPYD"],
        "capecitabine": ["DPYD"]
    }
    
    # Check if the drug is in our high-risk list
    drug_lower = drug.lower()
    
    for risk_drug, risk_genes in high_risk_combinations.items():
        if risk_drug in drug_lower:
            # Check if there's a gene overlap
            if any(gene in genes for gene in risk_genes):
                if score < 5.0:
                    return f"This medication may have reduced efficacy or increased risk of adverse effects based on your genetic profile involving {', '.join(risk_genes)} gene(s)."
                elif "HLA-B" in genes and ("abacavir" in drug_lower or "carbamazepine" in drug_lower):
                    return f"Potential risk of severe hypersensitivity reaction due to HLA-B genetic variant. Genetic testing is strongly recommended before using this medication."
                elif "DPYD" in genes and ("fluorouracil" in drug_lower or "capecitabine" in drug_lower):
                    return f"DPYD gene variants may increase risk of severe toxicity with this medication. Dose adjustment or alternative therapy may be needed."
                elif "SLCO1B1" in genes and "simvastatin" in drug_lower:
                    return f"SLCO1B1 variants may increase risk of muscle toxicity (myopathy) with this medication. Lower doses may be recommended."
                elif "CYP2D6" in genes and "codeine" in drug_lower:
                    return f"CYP2D6 genetic variations can affect how your body processes this medication, potentially affecting pain relief or increasing side effects."
    
    # General warnings based on score
    if score < 4.0:
        return f"This medication may have limited effectiveness based on your genetic profile. Consider consulting a healthcare provider for alternatives."
    
    return None

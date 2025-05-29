import pandas as pd
import numpy as np
import networkx as nx
import os
from collections import defaultdict

class PharmGKBProcessor:
    """
    Process PharmGKB data to extract gene-drug relationships
    and build knowledge graphs for genomic analysis.
    """
    def __init__(self, data_dir="attached_assets"):
        """
        Initialize PharmGKB processor.
        
        Args:
            data_dir: Directory containing PharmGKB data files
        """
        self.data_dir = data_dir
        self.clinical_annotations = None
        self.clinical_variants = None
        self.relationships = None
        self.drug_labels = None
        self.gene_drug_graph = None
        
        # Load PharmGKB data
        self.load_data()
    
    def load_data(self):
        """Load PharmGKB data files."""
        try:
            # Load clinical annotations
            clinical_annotations_file = os.path.join(self.data_dir, "clinical_annotations.tsv")
            if os.path.exists(clinical_annotations_file):
                self.clinical_annotations = pd.read_csv(clinical_annotations_file, sep='\t')
                print(f"Loaded {len(self.clinical_annotations)} clinical annotations.")
            
            # Load clinical variants
            clinical_variants_file = os.path.join(self.data_dir, "clinicalVariants.tsv")
            if os.path.exists(clinical_variants_file):
                self.clinical_variants = pd.read_csv(clinical_variants_file, sep='\t')
                print(f"Loaded {len(self.clinical_variants)} clinical variants.")
            
            # Load relationships
            relationships_file = os.path.join(self.data_dir, "relationships.tsv")
            if os.path.exists(relationships_file):
                self.relationships = pd.read_csv(relationships_file, sep='\t')
                print(f"Loaded {len(self.relationships)} relationships.")
            
            # Load drug labels
            drug_labels_file = os.path.join(self.data_dir, "drugLabels.tsv")
            if os.path.exists(drug_labels_file):
                self.drug_labels = pd.read_csv(drug_labels_file, sep='\t')
                print(f"Loaded {len(self.drug_labels)} drug labels.")
        
        except Exception as e:
            print(f"Error loading PharmGKB data: {e}")
    
    def build_gene_drug_graph(self):
        """
        Build a graph of gene-drug relationships from PharmGKB data.
        
        Returns:
            NetworkX graph of gene-drug relationships
        """
        G = nx.Graph()
        
        # Add gene-drug edges from clinical annotations
        if self.clinical_annotations is not None:
            for _, row in self.clinical_annotations.iterrows():
                gene = str(row.get('Gene', ''))
                if not gene or gene == 'nan':
                    continue
                
                drugs = str(row.get('Drug(s)', ''))
                if not drugs or drugs == 'nan':
                    continue
                
                # Add gene node
                if gene not in G:
                    G.add_node(gene, type='gene')
                
                # Add drug nodes and edges
                for drug in drugs.split(';'):
                    drug = drug.strip()
                    if not drug:
                        continue
                    
                    if drug not in G:
                        G.add_node(drug, type='drug')
                    
                    # Add edge with attributes
                    weight = float(row.get('Score', 0)) if not pd.isna(row.get('Score', 0)) else 0
                    level = str(row.get('Level of Evidence', ''))
                    
                    # Normalize weight to 0-1 range
                    normalized_weight = min(1.0, max(0.0, weight / 50.0))
                    
                    G.add_edge(gene, drug, weight=normalized_weight, level=level)
        
        # Add gene-drug edges from clinical variants
        if self.clinical_variants is not None:
            for _, row in self.clinical_variants.iterrows():
                gene = str(row.get('gene', ''))
                if not gene or gene == 'nan':
                    continue
                
                chemicals = str(row.get('chemicals', ''))
                if not chemicals or chemicals == 'nan':
                    continue
                
                # Add gene node
                if gene not in G:
                    G.add_node(gene, type='gene')
                
                # Add drug nodes and edges
                for chemical in chemicals.split(';'):
                    chemical = chemical.strip()
                    if not chemical:
                        continue
                    
                    if chemical not in G:
                        G.add_node(chemical, type='drug')
                    
                    # Add edge with attributes
                    level = str(row.get('level of evidence', ''))
                    
                    # Calculate weight based on level of evidence
                    weight = self._calculate_weight_from_level(level)
                    
                    G.add_edge(gene, chemical, weight=weight, level=level)
        
        # Add gene-drug edges from relationships
        if self.relationships is not None:
            for _, row in self.relationships.iterrows():
                entity1_type = str(row.get('Entity1_type', '')).lower()
                entity2_type = str(row.get('Entity2_type', '')).lower()
                
                # Look for gene-drug relationships
                if (entity1_type == 'gene' and entity2_type == 'chemical') or \
                   (entity2_type == 'gene' and entity1_type == 'chemical'):
                    
                    # Determine which entity is the gene and which is the drug
                    if entity1_type == 'gene':
                        gene = str(row.get('Entity1_name', ''))
                        drug = str(row.get('Entity2_name', ''))
                    else:
                        gene = str(row.get('Entity2_name', ''))
                        drug = str(row.get('Entity1_name', ''))
                    
                    if not gene or gene == 'nan' or not drug or drug == 'nan':
                        continue
                    
                    # Add nodes
                    if gene not in G:
                        G.add_node(gene, type='gene')
                    
                    if drug not in G:
                        G.add_node(drug, type='drug')
                    
                    # Add edge with attributes
                    association = str(row.get('Association', '')).lower()
                    evidence = str(row.get('Evidence', ''))
                    
                    # Calculate weight based on association
                    if association == 'associated':
                        weight = 0.75
                    elif association == 'ambiguous':
                        weight = 0.5
                    elif association == 'not associated':
                        weight = 0.25
                    else:
                        weight = 0.5
                    
                    G.add_edge(gene, drug, weight=weight, evidence=evidence)
        
        self.gene_drug_graph = G
        print(f"Built gene-drug graph with {len(G.nodes)} nodes and {len(G.edges)} edges.")
        
        return G
    
    def _calculate_weight_from_level(self, level):
        """
        Calculate edge weight based on level of evidence.
        
        Args:
            level: Level of evidence string
            
        Returns:
            Weight value between 0 and 1
        """
        level = str(level).strip().upper()
        
        # Map evidence levels to weights
        if level in ['1A', '1']:
            return 1.0
        elif level in ['1B', '2A']:
            return 0.85
        elif level in ['2B', '2']:
            return 0.7
        elif level in ['3']:
            return 0.55
        elif level in ['4']:
            return 0.4
        else:
            return 0.5  # Default middle weight
    
    def get_drug_recommendations(self, gene_variants, population='Indian', top_n=5):
        """
        Get drug recommendations based on gene variants.
        
        Args:
            gene_variants: List of gene variants
            population: Population group for tailored recommendations
            top_n: Number of top recommendations to return
            
        Returns:
            List of drug recommendations with scores
        """
        if self.gene_drug_graph is None:
            self.build_gene_drug_graph()
        
        # Extract genes from variants
        genes = []
        for variant in gene_variants:
            # Extract gene from variant
            if ',' in variant:
                # Format might be "CYP2C9*1, CYP2C9*2, CYP2C9*3"
                gene = variant.split(',')[0].split('*')[0].strip()
            elif '*' in variant:
                # Format might be "CYP2C9*2"
                gene = variant.split('*')[0].strip()
            elif 'rs' in variant:
                # Format might be "rs1799853"
                gene = self._find_gene_for_variant(variant)
            else:
                gene = variant
            
            if gene:
                genes.append(gene)
        
        # Get drug recommendations for these genes
        recommendations = []
        
        for gene in genes:
            if gene in self.gene_drug_graph:
                # Get connected drugs
                for neighbor in self.gene_drug_graph.neighbors(gene):
                    if self.gene_drug_graph.nodes[neighbor]['type'] == 'drug':
                        edge_data = self.gene_drug_graph.get_edge_data(gene, neighbor)
                        weight = edge_data.get('weight', 0.5)
                        level = edge_data.get('level', '')
                        
                        # Apply population-specific adjustment
                        adjusted_weight = self._apply_population_adjustment(neighbor, gene, weight, population)
                        
                        # Get phenotype information
                        phenotype = self._get_drug_phenotype(neighbor, gene)
                        
                        # Create recommendation
                        rec = {
                            'drug': neighbor,
                            'gene': gene,
                            'original_score': weight * 10,  # Scale to 0-10
                            'adjusted_score': adjusted_weight * 10,  # Scale to 0-10
                            'level': level,
                            'phenotype': phenotype
                        }
                        
                        recommendations.append(rec)
        
        # Sort by adjusted score (descending)
        recommendations.sort(key=lambda x: x['adjusted_score'], reverse=True)
        
        # Get top N recommendations
        top_recommendations = recommendations[:top_n]
        
        # Format recommendations
        formatted_recommendations = []
        for rec in top_recommendations:
            # Generate interpretation and warning
            interpretation = self._generate_interpretation(rec)
            warning = self._generate_warning(rec)
            
            formatted_rec = {
                'drug_name': rec['drug'],
                'effectiveness_score': rec['adjusted_score'],
                'gene_interaction': rec['gene'],
                'phenotype': rec['phenotype'],
                'interpretation': interpretation,
                'warning': warning if warning else None
            }
            
            formatted_recommendations.append(formatted_rec)
        
        return formatted_recommendations
    
    def _find_gene_for_variant(self, variant):
        """
        Find the gene associated with an rsID variant.
        
        Args:
            variant: rsID variant string
            
        Returns:
            Gene name or empty string if not found
        """
        variant = variant.strip()
        
        # Try clinical annotations
        if self.clinical_annotations is not None:
            if 'Variant/Haplotypes' in self.clinical_annotations.columns and 'Gene' in self.clinical_annotations.columns:
                matches = self.clinical_annotations[self.clinical_annotations['Variant/Haplotypes'] == variant]
                if not matches.empty:
                    return str(matches.iloc[0]['Gene'])
        
        # Try clinical variants
        if self.clinical_variants is not None:
            if 'variant' in self.clinical_variants.columns and 'gene' in self.clinical_variants.columns:
                matches = self.clinical_variants[self.clinical_variants['variant'] == variant]
                if not matches.empty:
                    return str(matches.iloc[0]['gene'])
        
        return ""
    
    def _apply_population_adjustment(self, drug, gene, base_weight, population):
        """
        Apply population-specific adjustment to drug recommendation weight.
        
        Args:
            drug: Drug name
            gene: Gene name
            base_weight: Base weight for the recommendation
            population: Population group
            
        Returns:
            Adjusted weight
        """
        # This would be based on population-specific genetic factors
        # For now, using a simplified approach with predefined adjustments for Indian population
        
        # Common genetic variants in Indian population that affect drug metabolism
        indian_genetic_adjustments = {
            # CYP2C9 variants common in Indian population
            'CYP2C9': {
                'warfarin': 0.9,  # Reduced effectiveness
                'phenytoin': 0.85,
                'diclofenac': 0.95
            },
            # CYP2D6 variants common in Indian population
            'CYP2D6': {
                'codeine': 0.8,  # Reduced effectiveness
                'tramadol': 0.85,
                'amitriptyline': 0.9
            },
            # TPMT variants
            'TPMT': {
                'azathioprine': 0.8,
                'mercaptopurine': 0.8
            },
            # VKORC1 variants
            'VKORC1': {
                'warfarin': 0.85
            },
            # SLCO1B1 variants
            'SLCO1B1': {
                'simvastatin': 0.75,
                'atorvastatin': 0.8,
                'rosuvastatin': 0.9
            },
            # G6PD variants (common in India)
            'G6PD': {
                'rasburicase': 0.7,
                'primaquine': 0.6,
                'nitrofurantoin': 0.75
            }
        }
        
        # Apply adjustment for Indian population
        if population.lower() == 'indian' and gene in indian_genetic_adjustments:
            drug_adjustments = indian_genetic_adjustments[gene]
            if drug in drug_adjustments:
                adjustment_factor = drug_adjustments[drug]
                return base_weight * adjustment_factor
        
        # No adjustment for other populations or gene-drug pairs not in the table
        return base_weight
    
    def _get_drug_phenotype(self, drug, gene):
        """
        Get phenotype information for a drug-gene pair.
        
        Args:
            drug: Drug name
            gene: Gene name
            
        Returns:
            Phenotype string
        """
        phenotypes = []
        
        # Try clinical annotations
        if self.clinical_annotations is not None:
            matches = self.clinical_annotations[
                (self.clinical_annotations['Drug(s)'].str.contains(drug, na=False, regex=False)) & 
                (self.clinical_annotations['Gene'] == gene)
            ]
            
            if not matches.empty:
                for _, row in matches.iterrows():
                    if 'Phenotype(s)' in row and row['Phenotype(s)'] and not pd.isna(row['Phenotype(s)']):
                        phenotypes.extend([p.strip() for p in str(row['Phenotype(s)']).split(';')])
        
        # Try clinical variants
        if not phenotypes and self.clinical_variants is not None:
            matches = self.clinical_variants[
                (self.clinical_variants['chemicals'].str.contains(drug, na=False, regex=False)) & 
                (self.clinical_variants['gene'] == gene)
            ]
            
            if not matches.empty:
                for _, row in matches.iterrows():
                    if 'phenotypes' in row and row['phenotypes'] and not pd.isna(row['phenotypes']):
                        phenotypes.extend([p.strip() for p in str(row['phenotypes']).split(';')])
        
        # Remove duplicates and empty strings
        phenotypes = list(set([p for p in phenotypes if p]))
        
        if phenotypes:
            return "; ".join(phenotypes)
        else:
            return "Not specified"
    
    def _generate_interpretation(self, rec):
        """
        Generate interpretation for drug recommendation.
        
        Args:
            rec: Drug recommendation dictionary
            
        Returns:
            Interpretation string
        """
        drug = rec['drug'].capitalize()
        gene = rec['gene']
        score = rec['adjusted_score']
        level = rec['level']
        
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
        interpretation = f"{drug} is {effectiveness} based on your genetic profile involving the {gene} gene."
        
        # Add level of evidence information
        if level:
            interpretation += f" This recommendation is based on evidence level {level}."
        
        # Add phenotype information if available
        phenotype = rec['phenotype']
        if phenotype and phenotype != "Not specified":
            interpretation += f" Indicated for: {phenotype}."
        
        # Add population-specific information
        interpretation += " The effectiveness has been adjusted for Indian genetic factors."
        
        return interpretation
    
    def _generate_warning(self, rec):
        """
        Generate warning for drug recommendation if needed.
        
        Args:
            rec: Drug recommendation dictionary
            
        Returns:
            Warning string or None
        """
        drug = rec['drug'].lower()
        gene = rec['gene']
        score = rec['adjusted_score']
        
        # High-risk gene-drug combinations
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
            "capecitabine": ["DPYD"],
            "rasburicase": ["G6PD"],
            "primaquine": ["G6PD"]
        }
        
        # Check if drug is in high-risk list and gene matches
        for risk_drug, risk_genes in high_risk_combinations.items():
            if risk_drug in drug and gene in risk_genes:
                if score < 5.0:
                    return f"This medication may have reduced efficacy or increased risk of adverse effects based on your genetic profile involving the {gene} gene."
                elif gene == "HLA-B" and ("abacavir" in drug or "carbamazepine" in drug):
                    return f"Potential risk of severe hypersensitivity reaction due to HLA-B genetic variant. Genetic testing is strongly recommended before using this medication."
                elif gene == "DPYD" and ("fluorouracil" in drug or "capecitabine" in drug):
                    return f"DPYD gene variants may increase risk of severe toxicity with this medication. Dose adjustment or alternative therapy may be needed."
                elif gene == "SLCO1B1" and "simvastatin" in drug:
                    return f"SLCO1B1 variants may increase risk of muscle toxicity (myopathy) with this medication. Lower doses may be recommended."
                elif gene == "G6PD" and ("rasburicase" in drug or "primaquine" in drug):
                    return f"G6PD deficiency, which is common in certain Indian populations, can cause hemolytic anemia with this medication. Alternative therapy may be needed."
                elif gene == "CYP2D6" and "codeine" in drug:
                    return f"CYP2D6 genetic variations can affect how your body processes this medication, potentially affecting pain relief or increasing side effects."
        
        # General warnings based on score
        if score < 4.0:
            return f"This medication may have limited effectiveness based on your genetic profile. Consider consulting a healthcare provider for alternatives."
        
        return None
    
    def get_indian_genetic_disorder_info(self):
        """
        Get information about common Indian genetic disorders.
        
        Returns:
            Dictionary of genetic disorder information
        """
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
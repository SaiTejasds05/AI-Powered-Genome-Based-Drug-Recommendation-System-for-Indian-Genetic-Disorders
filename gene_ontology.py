import pandas as pd
import numpy as np
from goatools import obo_parser
from goatools.associations import read_ncbi_gene2go
from goatools.semantic import TermCounts, get_info_content
from goatools.semantic import semantic_similarity
from goatools.gosubdag.gosubdag import GoSubDag
import os
import requests
from io import BytesIO
import zipfile

class GeneOntologyProcessor:
    """
    Process Gene Ontology data and calculate semantic similarity between genes
    based on their GO annotations.
    """
    def __init__(self, go_obo_file=None):
        """
        Initialize Gene Ontology processor.
        
        Args:
            go_obo_file: Path to the GO OBO file. If not provided, will download from GO website.
        """
        self.go_obo_file = go_obo_file
        self.go = None
        self.gene2go = None
        self.tcntobj = None
        
        # Load Gene Ontology
        self.load_gene_ontology()
    
    def load_gene_ontology(self):
        """Load the Gene Ontology from OBO file or download if not available."""
        if not self.go_obo_file:
            # Download GO OBO file if not provided
            self.go_obo_file = "go-basic.obo"
            if not os.path.exists(self.go_obo_file):
                print("Downloading Gene Ontology OBO file...")
                url = "http://purl.obolibrary.org/obo/go/go-basic.obo"
                response = requests.get(url)
                if response.status_code == 200:
                    with open(self.go_obo_file, 'wb') as f:
                        f.write(response.content)
                    print("Downloaded GO OBO file successfully.")
                else:
                    print(f"Failed to download GO OBO file: {response.status_code}")
                    return
        
        # Parse the OBO file
        print(f"Loading Gene Ontology from {self.go_obo_file}...")
        self.go = obo_parser.GODag(self.go_obo_file)
        print(f"Loaded {len(self.go)} GO terms.")
        
    def load_gene_associations(self, gene2go_file=None, taxid=9606):
        """
        Load gene-to-GO associations.
        
        Args:
            gene2go_file: Path to gene2go file. If not provided, will download from NCBI.
            taxid: Taxonomy ID for filtering associations (default: 9606 for human)
        """
        if not gene2go_file:
            # Download gene2go file if not provided
            gene2go_file = "gene2go"
            if not os.path.exists(gene2go_file):
                print("Downloading gene2go file from NCBI...")
                url = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz"
                response = requests.get(url)
                if response.status_code == 200:
                    with open(gene2go_file + ".gz", 'wb') as f:
                        f.write(response.content)
                    # Uncompress file
                    os.system(f"gunzip {gene2go_file}.gz")
                    print("Downloaded gene2go file successfully.")
                else:
                    print(f"Failed to download gene2go file: {response.status_code}")
                    # Create a minimal gene2go file for demonstration
                    self._create_minimal_gene2go(gene2go_file)
        
        # Read gene-to-GO associations
        print(f"Loading gene-to-GO associations from {gene2go_file}...")
        try:
            self.gene2go = read_ncbi_gene2go(gene2go_file, taxids=[taxid])
            print(f"Loaded associations for {len(self.gene2go)} genes.")
            
            # Calculate term counts for semantic similarity
            self.tcntobj = TermCounts(self.go, self.gene2go)
        except Exception as e:
            print(f"Error loading gene2go file: {e}")
            # Create a minimal gene2go file for demonstration
            self._create_minimal_gene2go(gene2go_file)
            # Try loading again
            self.gene2go = read_ncbi_gene2go(gene2go_file, taxids=[taxid])
            self.tcntobj = TermCounts(self.go, self.gene2go)
    
    def _create_minimal_gene2go(self, filename):
        """Create a minimal gene2go file for demonstration when download fails."""
        print("Creating minimal gene2go file for demonstration...")
        
        # Common genes related to drug metabolism
        gene_go_mapping = {
            "1544": {  # CYP1A2
                "GO:0005506": "molecular_function",  # iron ion binding
                "GO:0016705": "molecular_function",  # oxidoreductase activity
                "GO:0055114": "biological_process",  # oxidation-reduction process
            },
            "1565": {  # CYP2D6
                "GO:0005506": "molecular_function",  # iron ion binding
                "GO:0016705": "molecular_function",  # oxidoreductase activity
                "GO:0055114": "biological_process",  # oxidation-reduction process
            },
            "2147": {  # F2 (Coagulation factor II)
                "GO:0005515": "molecular_function",  # protein binding
                "GO:0007596": "biological_process",  # blood coagulation
            },
            "3708": {  # ITPR1
                "GO:0005220": "molecular_function",  # inositol 1,4,5-trisphosphate-sensitive calcium-release channel activity
                "GO:0006816": "biological_process",  # calcium ion transport
            }
        }
        
        # Write to file
        with open(filename, 'w') as f:
            f.write("#tax_id\tGeneID\tGO_ID\tEvidence\tQualifier\tGO_term\tPubMed\tCategory\n")
            for gene_id, go_terms in gene_go_mapping.items():
                for go_id, category in go_terms.items():
                    f.write(f"9606\t{gene_id}\t{go_id}\tIEA\t\tterm\t\t{category}\n")
    
    def calculate_gene_similarity(self, gene1, gene2, method='lin'):
        """
        Calculate semantic similarity between two genes based on their GO annotations.
        
        Args:
            gene1: First gene ID
            gene2: Second gene ID
            method: Semantic similarity method ('lin', 'resnik', or 'rel')
            
        Returns:
            Semantic similarity score
        """
        if not self.gene2go or not self.tcntobj:
            print("Gene-to-GO associations not loaded. Call load_gene_associations first.")
            return 0.0
        
        # Get GO terms for each gene
        gene1_gos = self.gene2go.get(gene1, set())
        gene2_gos = self.gene2go.get(gene2, set())
        
        if not gene1_gos or not gene2_gos:
            return 0.0
        
        # Calculate similarities between all pairs of GO terms
        similarities = []
        for go1 in gene1_gos:
            for go2 in gene2_gos:
                try:
                    if method == 'lin':
                        sim = semantic_similarity.lin_sim(go1, go2, self.go, self.tcntobj)
                    elif method == 'resnik':
                        sim = semantic_similarity.resnik_sim(go1, go2, self.go, self.tcntobj)
                    else:  # rel
                        sim = semantic_similarity.rel_sim(go1, go2, self.go, self.tcntobj)
                    
                    if sim is not None:
                        similarities.append(sim)
                except Exception as e:
                    continue
        
        # Return best match average
        if similarities:
            return np.mean(similarities)
        else:
            return 0.0
    
    def calculate_similarity_matrix(self, gene_list, method='lin'):
        """
        Calculate a similarity matrix for a list of genes.
        
        Args:
            gene_list: List of gene IDs
            method: Semantic similarity method
            
        Returns:
            Similarity matrix as a pandas DataFrame
        """
        n_genes = len(gene_list)
        sim_matrix = np.zeros((n_genes, n_genes))
        
        # Calculate similarity for each gene pair
        for i in range(n_genes):
            for j in range(i, n_genes):
                if i == j:
                    sim_matrix[i, j] = 1.0
                else:
                    sim = self.calculate_gene_similarity(gene_list[i], gene_list[j], method)
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim
        
        return pd.DataFrame(sim_matrix, index=gene_list, columns=gene_list)
    
    def get_gene_go_terms(self, gene_id):
        """
        Get GO terms associated with a gene.
        
        Args:
            gene_id: Gene ID
            
        Returns:
            Set of GO terms
        """
        if not self.gene2go:
            print("Gene-to-GO associations not loaded. Call load_gene_associations first.")
            return set()
        
        return self.gene2go.get(gene_id, set())
    
    def get_go_term_name(self, go_term):
        """
        Get the name of a GO term.
        
        Args:
            go_term: GO term ID
            
        Returns:
            Name of the GO term
        """
        if not self.go:
            print("Gene Ontology not loaded. Call load_gene_ontology first.")
            return ""
        
        if go_term in self.go:
            return self.go[go_term].name
        else:
            return ""
    
    def get_gene_go_annotations(self, gene_id):
        """
        Get detailed GO annotations for a gene.
        
        Args:
            gene_id: Gene ID
            
        Returns:
            Dictionary of GO terms with their names and namespaces
        """
        go_terms = self.get_gene_go_terms(gene_id)
        annotations = {}
        
        for go_term in go_terms:
            if go_term in self.go:
                annotations[go_term] = {
                    'name': self.go[go_term].name,
                    'namespace': self.go[go_term].namespace
                }
        
        return annotations
    
    def enrich_gene_data(self, genes_df, gene_col='Gene', id_mapping=None):
        """
        Enrich gene data with GO annotations.
        
        Args:
            genes_df: DataFrame containing gene information
            gene_col: Column name in the DataFrame containing gene names or IDs
            id_mapping: Dictionary mapping gene names to NCBI gene IDs, if needed
            
        Returns:
            Enriched DataFrame with GO annotations
        """
        if not self.go or not self.gene2go:
            print("Gene Ontology or gene associations not loaded.")
            return genes_df
        
        # Create a copy of the input DataFrame
        enriched_df = genes_df.copy()
        
        # Add GO annotations column
        enriched_df['GO_annotations'] = None
        enriched_df['GO_molecular_function'] = None
        enriched_df['GO_biological_process'] = None
        enriched_df['GO_cellular_component'] = None
        
        # Process each gene
        for idx, row in enriched_df.iterrows():
            gene = row[gene_col]
            
            # Map gene name to ID if necessary
            gene_id = gene
            if id_mapping and gene in id_mapping:
                gene_id = id_mapping[gene]
            
            # Get GO annotations
            annotations = self.get_gene_go_annotations(gene_id)
            
            # Store all annotations
            enriched_df.at[idx, 'GO_annotations'] = annotations
            
            # Separate by namespace
            mf = [go_id for go_id, info in annotations.items() if info['namespace'] == 'molecular_function']
            bp = [go_id for go_id, info in annotations.items() if info['namespace'] == 'biological_process']
            cc = [go_id for go_id, info in annotations.items() if info['namespace'] == 'cellular_component']
            
            enriched_df.at[idx, 'GO_molecular_function'] = mf
            enriched_df.at[idx, 'GO_biological_process'] = bp
            enriched_df.at[idx, 'GO_cellular_component'] = cc
        
        return enriched_df

# Helper functions to map gene symbols to NCBI gene IDs
def load_gene_info(gene_info_file=None):
    """
    Load gene info mapping from gene symbols to NCBI gene IDs.
    
    Args:
        gene_info_file: Path to gene_info file. If not provided, will download from NCBI.
        
    Returns:
        Dictionary mapping gene symbols to gene IDs
    """
    if not gene_info_file:
        # Download gene_info file if not provided
        gene_info_file = "gene_info"
        if not os.path.exists(gene_info_file):
            print("Downloading gene_info file from NCBI...")
            url = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz"
            response = requests.get(url)
            if response.status_code == 200:
                with open(gene_info_file + ".gz", 'wb') as f:
                    f.write(response.content)
                # Uncompress file
                os.system(f"gunzip {gene_info_file}.gz")
                print("Downloaded gene_info file successfully.")
            else:
                print(f"Failed to download gene_info file: {response.status_code}")
                # Create a minimal gene_info file
                _create_minimal_gene_info(gene_info_file)
    
    # Read gene_info file
    print(f"Loading gene info from {gene_info_file}...")
    try:
        gene_info = pd.read_csv(gene_info_file, sep='\t', comment='#')
        
        # Create mapping of gene symbols to gene IDs
        symbol_to_id = {}
        for _, row in gene_info.iterrows():
            if row['tax_id'] == 9606:  # Human genes
                symbol_to_id[row['Symbol']] = str(row['GeneID'])
                # Also add synonyms
                for synonym in row['Synonyms'].split('|'):
                    if synonym:
                        symbol_to_id[synonym] = str(row['GeneID'])
        
        return symbol_to_id
    except Exception as e:
        print(f"Error loading gene_info file: {e}")
        # Create a minimal gene_info file
        _create_minimal_gene_info(gene_info_file)
        # Try loading again
        gene_info = pd.read_csv(gene_info_file, sep='\t')
        
        symbol_to_id = {}
        for _, row in gene_info.iterrows():
            symbol_to_id[row['Symbol']] = str(row['GeneID'])
        
        return symbol_to_id

def _create_minimal_gene_info(filename):
    """Create a minimal gene_info file when download fails."""
    print("Creating minimal gene_info file for demonstration...")
    
    # Common genes related to drug metabolism
    gene_info_data = {
        "GeneID": [1544, 1565, 1576, 2147, 3708, 5243, 7412],
        "Symbol": ["CYP1A2", "CYP2D6", "CYP3A4", "F2", "ITPR1", "ABCB1", "VKORC1"],
        "Synonyms": ["CP12|P3-450|P450(PA)", "CPD6|CYP2D|P450C2D|P450DB1", "CP34|CYP3A|NF-25|P450C3", "PT|THPH1|RPRGL2", "IP3R|IP3R1|SCA15|SCA29", "MDR1|P-GP|CD243|GP170", "VKOR|MSX2"],
        "tax_id": [9606, 9606, 9606, 9606, 9606, 9606, 9606],
        "description": [
            "cytochrome P450 family 1 subfamily A member 2",
            "cytochrome P450 family 2 subfamily D member 6",
            "cytochrome P450 family 3 subfamily A member 4",
            "coagulation factor II, thrombin",
            "inositol 1,4,5-trisphosphate receptor type 1",
            "ATP binding cassette subfamily B member 1",
            "vitamin K epoxide reductase complex subunit 1"
        ]
    }
    
    # Create DataFrame and write to file
    df = pd.DataFrame(gene_info_data)
    df.to_csv(filename, sep='\t', index=False)
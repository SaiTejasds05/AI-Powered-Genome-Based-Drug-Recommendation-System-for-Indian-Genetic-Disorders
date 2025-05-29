import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DrugRecommendationEnvironment:
    """
    Reinforcement Learning environment for drug recommendation adaptation.
    This environment simulates the process of recommending drugs based on
    genetic profiles and learning from feedback.
    """
    def __init__(self, genomic_data, drug_data, population_data=None, reward_scale=10.0):
        """
        Initialize the drug recommendation environment.
        
        Args:
            genomic_data: DataFrame containing genomic variant data
            drug_data: DataFrame containing drug information and interactions
            population_data: Optional DataFrame with population-specific genetic information
            reward_scale: Scaling factor for rewards
        """
        self.genomic_data = genomic_data
        self.drug_data = drug_data
        self.population_data = population_data
        self.reward_scale = reward_scale
        
        # Extract unique genes and drugs
        self.genes = self._extract_genes()
        self.drugs = self._extract_drugs()
        
        # State and action spaces
        self.state_dim = len(self.genes) + 4  # Genes + demographic features
        self.action_dim = len(self.drugs)
        
        # Current state
        self.current_state = None
        self.current_patient_id = None
        
        # Reset the environment
        self.reset()
    
    def _extract_genes(self):
        """Extract unique genes from genomic data."""
        if 'Gene' in self.genomic_data.columns:
            return list(set(self.genomic_data['Gene'].dropna().unique()))
        else:
            # Try to find alternative column names
            for col in self.genomic_data.columns:
                if 'gene' in col.lower():
                    return list(set(self.genomic_data[col].dropna().unique()))
        
        # If no gene column found, return empty list
        return []
    
    def _extract_drugs(self):
        """Extract unique drugs from drug data."""
        if 'Drug(s)' in self.drug_data.columns:
            # Split drug lists and get unique values
            all_drugs = []
            for drugs in self.drug_data['Drug(s)'].dropna():
                all_drugs.extend([d.strip() for d in str(drugs).split(';')])
            return list(set(all_drugs))
        elif 'drug' in self.drug_data.columns:
            return list(set(self.drug_data['drug'].dropna().unique()))
        else:
            # Try to find alternative column names
            for col in self.drug_data.columns:
                if 'drug' in col.lower() or 'chemical' in col.lower():
                    return list(set(self.drug_data[col].dropna().unique()))
        
        # If no drug column found, return empty list
        return []
    
    def reset(self, patient_id=None):
        """
        Reset the environment and return the initial state.
        
        Args:
            patient_id: Optional patient ID to initialize state
            
        Returns:
            Initial state as a numpy array
        """
        if patient_id is None:
            # Randomly select a patient
            if len(self.genomic_data) > 0:
                patient_idx = random.randint(0, len(self.genomic_data) - 1)
                patient_id = self.genomic_data.index[patient_idx]
        
        self.current_patient_id = patient_id
        self.current_state = self._get_patient_state(patient_id)
        
        return self.current_state
    
    def _get_patient_state(self, patient_id):
        """
        Get the state representation for a patient.
        
        Args:
            patient_id: Patient ID or index
            
        Returns:
            State vector as a numpy array
        """
        # Initialize state vector with zeros
        state = np.zeros(self.state_dim)
        
        if patient_id is not None and patient_id < len(self.genomic_data):
            # Get patient's genomic data
            if isinstance(patient_id, int):
                patient_data = self.genomic_data.iloc[patient_id]
            else:
                patient_data = self.genomic_data.loc[patient_id]
            
            # Set gene features based on presence of genes
            gene_col = 'Gene' if 'Gene' in self.genomic_data.columns else 'gene'
            if gene_col in self.genomic_data.columns:
                gene = str(patient_data[gene_col])
                if gene in self.genes:
                    gene_idx = self.genes.index(gene)
                    state[gene_idx] = 1.0
            
            # Add demographic features (placeholders for now)
            # These could be age, sex, ethnicity, etc.
            state[-4:] = np.random.random(4)  # Placeholder for demo purposes
        
        return state
    
    def step(self, action):
        """
        Take a step in the environment by recommending a drug.
        
        Args:
            action: Drug index to recommend
            
        Returns:
            next_state: New state
            reward: Reward for the action
            done: Whether the episode is done
            info: Additional information
        """
        # Ensure action is valid
        if not 0 <= action < self.action_dim:
            reward = -self.reward_scale  # Penalty for invalid action
            return self.current_state, reward, True, {"error": "Invalid action"}
        
        # Get the recommended drug
        drug = self.drugs[action]
        
        # Calculate reward based on drug-gene interactions
        reward = self._calculate_reward(drug)
        
        # Episode is done after recommending a drug
        done = True
        
        # Return the result
        info = {
            "drug": drug,
            "patient_id": self.current_patient_id,
            "effectiveness": reward / self.reward_scale
        }
        
        return self.current_state, reward, done, info
    
    def _calculate_reward(self, drug):
        """
        Calculate reward for recommending a drug based on patient's genetic profile.
        
        Args:
            drug: Drug name
            
        Returns:
            Reward value
        """
        # Get patient's gene
        patient_data = self.genomic_data.iloc[self.current_patient_id] if isinstance(self.current_patient_id, int) else self.genomic_data.loc[self.current_patient_id]
        gene_col = 'Gene' if 'Gene' in self.genomic_data.columns else 'gene'
        gene = str(patient_data[gene_col]) if gene_col in patient_data else ""
        
        # Find drug-gene interaction in drug data
        interaction_found = False
        effectiveness = 0.5  # Default middle effectiveness
        
        # Search for the drug-gene pair in the drug data
        drug_data_filtered = None
        
        if 'drug' in self.drug_data.columns and 'gene' in self.drug_data.columns:
            drug_data_filtered = self.drug_data[(self.drug_data['drug'] == drug) & (self.drug_data['gene'] == gene)]
        elif 'Drug(s)' in self.drug_data.columns and 'Gene' in self.drug_data.columns:
            # For data with drug lists
            drug_data_filtered = self.drug_data[
                (self.drug_data['Drug(s)'].str.contains(drug, na=False)) & 
                (self.drug_data['Gene'] == gene)
            ]
        
        if drug_data_filtered is not None and len(drug_data_filtered) > 0:
            interaction_found = True
            
            # Get effectiveness from score if available
            if 'score' in drug_data_filtered.columns:
                effectiveness = drug_data_filtered['score'].mean() / 10.0  # Normalize to [0, 1]
            elif 'Score' in drug_data_filtered.columns:
                effectiveness = drug_data_filtered['Score'].mean() / 10.0
        
        # Apply population-specific adjustments if available
        if self.population_data is not None:
            # This would adjust the effectiveness based on population genetics
            population_adjustment = self._get_population_adjustment(drug, gene)
            effectiveness *= population_adjustment
        
        # Convert effectiveness to reward
        reward = (effectiveness * 2 - 1) * self.reward_scale  # Scale to [-reward_scale, reward_scale]
        
        return reward
    
    def _get_population_adjustment(self, drug, gene):
        """
        Get adjustment factor based on population genetics.
        
        Args:
            drug: Drug name
            gene: Gene name
            
        Returns:
            Adjustment factor between 0.5 and 1.5
        """
        # This is a placeholder. In a real system, this would use actual population data
        # to adjust the effectiveness based on genetic variations common in specific populations.
        return 1.0  # Default no adjustment
        
    def get_state_dim(self):
        """Get the dimension of the state space."""
        return self.state_dim
    
    def get_action_dim(self):
        """Get the dimension of the action space."""
        return self.action_dim


class DQNAgent:
    """
    Deep Q-Network agent for drug recommendation.
    """
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 memory_size=2000, batch_size=32):
        """
        Initialize the DQN agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            learning_rate: Learning rate for the neural network
            gamma: Discount factor
            epsilon: Exploration rate
            epsilon_decay: Rate at which to decay epsilon
            epsilon_min: Minimum value of epsilon
            memory_size: Size of the replay memory
            batch_size: Batch size for training
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        
        # Q-Network
        self.model = self._build_model(learning_rate)
        self.target_model = self._build_model(learning_rate)
        self.update_target_model()
    
    def _build_model(self, learning_rate):
        """
        Build the neural network model.
        
        Args:
            learning_rate: Learning rate for the optimizer
            
        Returns:
            Compiled neural network model
        """
        model = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)
        )
        
        # Define loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        return model
    
    def update_target_model(self):
        """Update the target model with the weights of the main model."""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """
        Choose an action based on the current state.
        
        Args:
            state: Current state
            
        Returns:
            Chosen action
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state_tensor)
        return torch.argmax(act_values[0]).item()
    
    def replay(self):
        """Train the model on a batch of experiences from memory."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch of experiences
        minibatch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.target_model(next_state_tensor)).item()
            
            # Get current Q values
            current_q = self.model(state_tensor)
            target_q = current_q.clone()
            target_q[0][action] = target
            
            # Train the model
            self.optimizer.zero_grad()
            loss = self.criterion(current_q, target_q)
            loss.backward()
            self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load_model(self, filepath):
        """
        Load model weights from a file.
        
        Args:
            filepath: Path to the model weights file
        """
        self.model.load_state_dict(torch.load(filepath))
        self.update_target_model()
    
    def save_model(self, filepath):
        """
        Save model weights to a file.
        
        Args:
            filepath: Path to save the model weights
        """
        torch.save(self.model.state_dict(), filepath)


def train_rl_model(env, agent, episodes=100):
    """
    Train the RL agent for drug recommendation.
    
    Args:
        env: Drug recommendation environment
        agent: DQN agent
        episodes: Number of episodes to train
        
    Returns:
        Training history
    """
    history = {
        'episode_rewards': [],
        'drugs_recommended': []
    }
    
    for episode in range(episodes):
        # Reset the environment
        state = env.reset()
        
        # Choose and take an action
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        
        # Remember the experience
        agent.remember(state, action, reward, next_state, done)
        
        # Train the agent
        agent.replay()
        
        # Update target model periodically
        if episode % 10 == 0:
            agent.update_target_model()
        
        # Record history
        history['episode_rewards'].append(reward)
        history['drugs_recommended'].append(info['drug'])
        
        print(f"Episode {episode + 1}/{episodes}: Reward = {reward:.2f}, Drug = {info['drug']}")
    
    return history
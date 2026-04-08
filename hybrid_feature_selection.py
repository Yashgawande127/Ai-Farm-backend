"""
Hybrid Feature Selection using POA (Pelican Optimization Algorithm) and CSSOA (Chaotic Sine-Cosine Optimization Algorithm)
This implementation combines the exploration capabilities of POA with the exploitation power of CSSOA
"""

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import logging

logger = logging.getLogger(__name__)

class PelicanOptimization:
    """
    Pelican Optimization Algorithm (POA) for feature selection
    Mimics the hunting behavior of pelicans
    """
    
    def __init__(self, pop_size, dim, bounds=(0, 1)):
        self.pop_size = pop_size
        self.dim = dim
        self.bounds = bounds
        self.population = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))
        self.fitness = np.zeros(pop_size)
        self.best_position = None
        self.best_fitness = float('inf')
        
    def evaluate_fitness(self, X, y, position):
        """Evaluate fitness of a feature subset"""
        # Convert continuous values to binary feature mask
        feature_mask = position > 0.5
        
        # Ensure at least one feature is selected
        if not np.any(feature_mask):
            feature_mask[np.argmax(position)] = True
        
        # Select features
        X_selected = X[:, feature_mask]
        
        # Calculate fitness using cross-validation
        try:
            rf = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
            scores = cross_val_score(rf, X_selected, y, cv=3, scoring='accuracy')
            accuracy = scores.mean()
            
            # Multi-objective: maximize accuracy, minimize features
            num_features = np.sum(feature_mask)
            fitness = -(accuracy - 0.01 * num_features / self.dim)  # Minimize this
            
            return fitness, feature_mask, accuracy, num_features
        except:
            return float('inf'), feature_mask, 0, np.sum(feature_mask)
    
    def update_best(self, position, fitness):
        """Update global best position"""
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_position = position.copy()
            return True
        return False
    
    def phase1_exploration(self, t, max_iter):
        """Phase 1: Exploration (moving towards prey)"""
        new_population = np.zeros_like(self.population)
        
        for i in range(self.pop_size):
            # Random pelican for interaction
            j = np.random.randint(0, self.pop_size)
            while j == i:
                j = np.random.randint(0, self.pop_size)
            
            # Exploration coefficient
            P = 2 * np.random.random() - 1
            
            # Update position
            if abs(P) < 1:
                # Move towards prey
                new_population[i] = self.population[i] + P * (self.best_position - self.population[i])
            else:
                # Random movement
                new_population[i] = self.population[i] + np.random.uniform(-1, 1, self.dim) * self.population[i]
            
            # Boundary control
            new_population[i] = np.clip(new_population[i], self.bounds[0], self.bounds[1])
        
        return new_population
    
    def phase2_exploitation(self, t, max_iter):
        """Phase 2: Exploitation (attacking prey)"""
        new_population = np.zeros_like(self.population)
        
        for i in range(self.pop_size):
            # Exploitation coefficient decreasing over time
            E = 2 * np.random.random() - 1
            E = E * (1 - t / max_iter)
            
            if abs(E) < 1:
                # Attack prey (exploitation)
                D = abs(self.best_position - self.population[i])
                new_population[i] = self.best_position - E * D
            else:
                # Search for new prey
                rand_pelican = self.population[np.random.randint(0, self.pop_size)]
                new_population[i] = rand_pelican - E * abs(rand_pelican - self.population[i])
            
            # Boundary control
            new_population[i] = np.clip(new_population[i], self.bounds[0], self.bounds[1])
        
        return new_population

class ChaoticSineCosineOptimization:
    """
    Chaotic Sine-Cosine Optimization Algorithm (CSSOA) for feature selection
    Enhanced version of SCA with chaotic maps for better exploration
    """
    
    def __init__(self, pop_size, dim, bounds=(0, 1)):
        self.pop_size = pop_size
        self.dim = dim
        self.bounds = bounds
        self.population = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))
        self.fitness = np.zeros(pop_size)
        self.best_position = None
        self.best_fitness = float('inf')
        self.chaotic_value = 0.7  # Initial chaotic value
        
    def chaotic_map(self, x):
        """Logistic chaotic map"""
        return 4 * x * (1 - x)
    
    def evaluate_fitness(self, X, y, position):
        """Same as POA fitness evaluation"""
        feature_mask = position > 0.5
        
        if not np.any(feature_mask):
            feature_mask[np.argmax(position)] = True
        
        X_selected = X[:, feature_mask]
        
        try:
            rf = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
            scores = cross_val_score(rf, X_selected, y, cv=3, scoring='accuracy')
            accuracy = scores.mean()
            
            num_features = np.sum(feature_mask)
            fitness = -(accuracy - 0.01 * num_features / self.dim)
            
            return fitness, feature_mask, accuracy, num_features
        except:
            return float('inf'), feature_mask, 0, np.sum(feature_mask)
    
    def update_best(self, position, fitness):
        """Update global best position"""
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_position = position.copy()
            return True
        return False
    
    def update_positions(self, t, max_iter):
        """Update positions using chaotic sine-cosine operator"""
        new_population = np.zeros_like(self.population)
        
        # Update chaotic value
        self.chaotic_value = self.chaotic_map(self.chaotic_value)
        
        for i in range(self.pop_size):
            # Adaptive parameters
            a = 2 - 2 * t / max_iter  # Linearly decreasing from 2 to 0
            
            for j in range(self.dim):
                # Chaotic parameters
                r1 = self.chaotic_value
                r2 = 2 * np.pi * np.random.random()
                r3 = np.random.random()
                r4 = np.random.random()
                
                if r4 < 0.5:
                    # Sine component with chaos
                    new_population[i, j] = (self.population[i, j] + 
                                          r1 * np.sin(r2) * abs(r3 * self.best_position[j] - self.population[i, j]))
                else:
                    # Cosine component with chaos
                    new_population[i, j] = (self.population[i, j] + 
                                          r1 * np.cos(r2) * abs(r3 * self.best_position[j] - self.population[i, j]))
            
            # Boundary control
            new_population[i] = np.clip(new_population[i], self.bounds[0], self.bounds[1])
        
        return new_population

def hybrid_feature_selection(X, y, pop_size=30, max_iter=50, poa_ratio=0.6):
    """
    Hybrid POA-CSSOA feature selection algorithm
    
    Args:
        X: Feature matrix
        y: Target vector
        pop_size: Population size
        max_iter: Maximum iterations
        poa_ratio: Ratio of population using POA vs CSSOA
    
    Returns:
        Dictionary with selection results
    """
    logger.info(f"Starting Hybrid POA-CSSOA feature selection with {X.shape[1]} features")
    
    # Split population between POA and CSSOA
    poa_size = int(pop_size * poa_ratio)
    cssoa_size = pop_size - poa_size
    
    # Initialize algorithms
    poa = PelicanOptimization(poa_size, X.shape[1])
    cssoa = ChaoticSineCosineOptimization(cssoa_size, X.shape[1])
    
    # Global best tracking
    global_best_position = None
    global_best_fitness = float('inf')
    global_best_mask = None
    global_best_accuracy = 0
    global_best_features = 0
    
    # Evolution history
    history = {'fitness': [], 'accuracy': [], 'features': []}
    
    for t in range(max_iter):
        logger.info(f"Iteration {t+1}/{max_iter}")
        
        # Evaluate POA population
        for i in range(poa_size):
            fitness, mask, acc, n_feat = poa.evaluate_fitness(X, y, poa.population[i])
            poa.fitness[i] = fitness
            
            if poa.update_best(poa.population[i], fitness):
                logger.info(f"POA: New best fitness: {fitness:.4f}, Accuracy: {acc:.4f}, Features: {n_feat}")
        
        # Evaluate CSSOA population
        for i in range(cssoa_size):
            fitness, mask, acc, n_feat = cssoa.evaluate_fitness(X, y, cssoa.population[i])
            cssoa.fitness[i] = fitness
            
            if cssoa.update_best(cssoa.population[i], fitness):
                logger.info(f"CSSOA: New best fitness: {fitness:.4f}, Accuracy: {acc:.4f}, Features: {n_feat}")
        
        # Update global best
        if poa.best_fitness < global_best_fitness:
            global_best_fitness = poa.best_fitness
            global_best_position = poa.best_position.copy()
            _, global_best_mask, global_best_accuracy, global_best_features = poa.evaluate_fitness(X, y, poa.best_position)
        
        if cssoa.best_fitness < global_best_fitness:
            global_best_fitness = cssoa.best_fitness
            global_best_position = cssoa.best_position.copy()
            _, global_best_mask, global_best_accuracy, global_best_features = cssoa.evaluate_fitness(X, y, cssoa.best_position)
        
        # Information sharing between algorithms
        if t % 5 == 0 and t > 0:  # Every 5 iterations
            # Share best solutions
            if poa.best_fitness < cssoa.best_fitness:
                cssoa.best_position = poa.best_position.copy()
                cssoa.best_fitness = poa.best_fitness
            else:
                poa.best_position = cssoa.best_position.copy()
                poa.best_fitness = cssoa.best_fitness
        
        # Update positions
        poa.population = poa.phase1_exploration(t, max_iter) if t < max_iter // 2 else poa.phase2_exploitation(t, max_iter)
        cssoa.population = cssoa.update_positions(t, max_iter)
        
        # Record history
        history['fitness'].append(global_best_fitness)
        history['accuracy'].append(global_best_accuracy)
        history['features'].append(global_best_features)
        
        # Early stopping if no improvement for 10 iterations
        if t > 10 and len(set(history['fitness'][-10:])) == 1:
            logger.info(f"Early stopping at iteration {t+1} - no improvement")
            break
    
    logger.info(f"Hybrid POA-CSSOA completed. Best accuracy: {global_best_accuracy:.4f} with {global_best_features} features")
    
    return {
        'selected_features': global_best_mask,
        'num_selected': global_best_features,
        'fitness': -global_best_fitness,  # Convert back to positive
        'accuracy': global_best_accuracy,
        'feature_indices': np.where(global_best_mask)[0].tolist(),
        'history': history,
        'algorithm': 'Hybrid POA-CSSOA',
        'iterations': t + 1
    }

def evaluate_feature_subset(X, y, feature_mask):
    """
    Evaluate a specific feature subset
    
    Args:
        X: Feature matrix
        y: Target vector  
        feature_mask: Boolean mask for selected features
    
    Returns:
        Dictionary with evaluation metrics
    """
    if not np.any(feature_mask):
        return {'accuracy': 0, 'fitness': float('inf')}
    
    X_selected = X[:, feature_mask]
    
    try:
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        scores = cross_val_score(rf, X_selected, y, cv=5, scoring='accuracy')
        accuracy = scores.mean()
        std = scores.std()
        
        num_features = np.sum(feature_mask)
        fitness = accuracy - 0.01 * num_features / len(feature_mask)
        
        return {
            'accuracy': accuracy,
            'accuracy_std': std,
            'fitness': fitness,
            'num_features': num_features,
            'feature_ratio': num_features / len(feature_mask)
        }
    except Exception as e:
        logger.error(f"Error evaluating feature subset: {e}")
        return {'accuracy': 0, 'fitness': float('inf')}
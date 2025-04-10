import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler

class FuzzyCMeans:
    """
    Fuzzy C-Means clustering implementation
    """
    
    def __init__(self, n_clusters: int = 3, max_iter: int = 100,
                 m: float = 2.0, error: float = 1e-6):
        """
        Initialize Fuzzy C-Means clustering.
        
        Args:
            n_clusters: Number of clusters
            max_iter: Maximum number of iterations
            m: Fuzziness parameter
            error: Convergence threshold
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.error = error
        self.scaler = StandardScaler()
        
    def initialize_membership(self, n_samples: int) -> np.ndarray:
        """
        Initialize random membership matrix.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            Initial membership matrix
        """
        membership = np.random.rand(n_samples, self.n_clusters)
        # Normalize rows to sum to 1
        return membership / membership.sum(axis=1, keepdims=True)
        
    def update_centroids(self, X: np.ndarray, membership: np.ndarray) -> np.ndarray:
        """
        Update cluster centroids.
        
        Args:
            X: Input data (n_samples, n_features)
            membership: Membership matrix
            
        Returns:
            Updated centroids
        """
        # Raise membership to power m
        powered_membership = membership ** self.m
        
        # Calculate centroids
        numerator = powered_membership.T @ X
        denominator = powered_membership.sum(axis=0).reshape(-1, 1)
        
        return numerator / denominator
        
    def update_membership(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Update membership matrix.
        
        Args:
            X: Input data
            centroids: Cluster centroids
            
        Returns:
            Updated membership matrix
        """
        n_samples = X.shape[0]
        membership = np.zeros((n_samples, self.n_clusters))
        
        for i in range(n_samples):
            distances = np.linalg.norm(X[i] - centroids, axis=1)
            
            # Handle zero distances
            if np.any(distances == 0):
                membership[i] = np.where(distances == 0, 1, 0)
            else:
                # Calculate membership values
                sum_term = np.sum([(distances[j]/distances[k])**(2/(self.m-1))
                                 for k in range(self.n_clusters)], axis=0)
                membership[i] = 1 / sum_term
                
        return membership
        
    def fit(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit Fuzzy C-Means clustering.
        
        Args:
            X: Input data (n_samples, n_features)
            
        Returns:
            Tuple of (membership matrix, centroids)
        """
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize membership matrix
        membership = self.initialize_membership(X_scaled.shape[0])
        
        for _ in range(self.max_iter):
            old_membership = membership.copy()
            
            # Update centroids and membership
            centroids = self.update_centroids(X_scaled, membership)
            membership = self.update_membership(X_scaled, centroids)
            
            # Check convergence
            if np.linalg.norm(membership - old_membership) < self.error:
                break
                
        # Transform centroids back to original scale
        centroids = self.scaler.inverse_transform(centroids)
        
        return membership, centroids
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster memberships for new data.
        
        Args:
            X: Input data
            
        Returns:
            Membership matrix for new data
        """
        X_scaled = self.scaler.transform(X)
        centroids = self.scaler.transform(self.centroids)
        
        return self.update_membership(X_scaled, centroids) 
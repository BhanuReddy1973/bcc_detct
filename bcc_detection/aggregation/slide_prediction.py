import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
import logging

class SlidePredictor:
    def __init__(self, confidence_threshold=0.3, min_cluster_size=5):
        self.confidence_threshold = confidence_threshold
        self.min_cluster_size = min_cluster_size
    
    def assign_patch_labels(self, predictions, confidences):
        """Assign labels to patches based on predictions and confidences"""
        try:
            # Convert predictions to binary labels
            labels = (predictions > self.confidence_threshold).astype(int)
            return labels, confidences
            
        except Exception as e:
            logging.error(f"Error assigning patch labels: {str(e)}")
            raise
    
    def enhance_spatial_coherence(self, labels, confidences, coordinates):
        """Enhance spatial coherence using weighted majority voting"""
        try:
            # Create a grid of predictions
            max_x = max(coord[0] for coord in coordinates)
            max_y = max(coord[1] for coord in coordinates)
            
            prediction_grid = np.zeros((max_y + 1, max_x + 1))
            confidence_grid = np.zeros((max_y + 1, max_x + 1))
            
            # Fill the grid
            for (x, y), label, conf in zip(coordinates, labels, confidences):
                prediction_grid[y, x] = label
                confidence_grid[y, x] = conf
            
            # Apply Gaussian filter for smoothing
            smoothed_predictions = gaussian_filter(prediction_grid * confidence_grid, sigma=1)
            smoothed_confidences = gaussian_filter(confidence_grid, sigma=1)
            
            # Normalize and threshold
            normalized_predictions = smoothed_predictions / (smoothed_confidences + 1e-6)
            enhanced_labels = (normalized_predictions > self.confidence_threshold).astype(int)
            
            return enhanced_labels
            
        except Exception as e:
            logging.error(f"Error enhancing spatial coherence: {str(e)}")
            raise
    
    def identify_clusters(self, labels, coordinates):
        """Identify clusters of positive predictions"""
        try:
            # Get coordinates of positive predictions
            positive_coords = np.array([coord for coord, label in zip(coordinates, labels) if label == 1])
            
            if len(positive_coords) == 0:
                return []
            
            # Use DBSCAN to identify clusters
            clustering = DBSCAN(eps=2, min_samples=self.min_cluster_size).fit(positive_coords)
            
            # Get unique cluster labels
            unique_clusters = np.unique(clustering.labels_)
            clusters = []
            
            for cluster_id in unique_clusters:
                if cluster_id != -1:  # Ignore noise points
                    cluster_points = positive_coords[clustering.labels_ == cluster_id]
                    clusters.append(cluster_points)
            
            return clusters
            
        except Exception as e:
            logging.error(f"Error identifying clusters: {str(e)}")
            raise
    
    def calculate_slide_score(self, labels, confidences, clusters):
        """Calculate final slide-level prediction score"""
        try:
            # Calculate base score from predictions
            base_score = np.mean(confidences[labels == 1]) if np.any(labels == 1) else 0
            
            # Calculate cluster-based score
            cluster_score = 0
            if clusters:
                cluster_sizes = [len(cluster) for cluster in clusters]
                max_cluster_size = max(cluster_sizes)
                cluster_score = max_cluster_size / len(labels)
            
            # Combine scores
            final_score = 0.7 * base_score + 0.3 * cluster_score
            
            return final_score
            
        except Exception as e:
            logging.error(f"Error calculating slide score: {str(e)}")
            raise
    
    def predict_slide(self, predictions, confidences, coordinates):
        """Generate slide-level prediction"""
        try:
            # Assign patch labels
            labels, confidences = self.assign_patch_labels(predictions, confidences)
            
            # Enhance spatial coherence
            enhanced_labels = self.enhance_spatial_coherence(labels, confidences, coordinates)
            
            # Identify clusters
            clusters = self.identify_clusters(enhanced_labels, coordinates)
            
            # Calculate final score
            slide_score = self.calculate_slide_score(enhanced_labels, confidences, clusters)
            
            # Make final prediction
            slide_prediction = 1 if slide_score > self.confidence_threshold else 0
            
            return {
                'prediction': slide_prediction,
                'confidence': slide_score,
                'num_clusters': len(clusters),
                'cluster_sizes': [len(cluster) for cluster in clusters]
            }
            
        except Exception as e:
            logging.error(f"Error predicting slide: {str(e)}")
            raise 
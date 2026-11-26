import numpy as np

def calculate_lof_iterative(neighborhood_indices: np.ndarray, lrd_array: np.ndarray) -> np.ndarray:
    """
    Calculate Local Outlier Factor (LOF) for each point using explicit loops.
    
    Args:
        neighborhood_indices (np.ndarray): An N x k array containing indices of k-nearest neighbors.
        lrd_array (np.ndarray): A 1D array of length N containing the Local Reachability Density.
        
    Returns:
        np.ndarray: A 1D array of length N containing the LOF scores.
    """
    n_points, k = neighborhood_indices.shape
    lof_scores = np.zeros(n_points)
    
    # 1. For each point p...
    for p in range(n_points):
        lrd_p = lrd_array[p]
        sum_ratios = 0.0
        
        # ...iterate through its k neighbors (o in N_MinPts[p])
        for o in neighborhood_indices[p]:
            lrd_o = lrd_array[o]
            
            # 2. Calculate the ratio of the neighbor's LRD to the point's own LRD
            # 3. Sum all these ratios
            sum_ratios += lrd_o / lrd_p
            
        # 4. Divide the sum by the number of neighbors (k)
        lof_scores[p] = sum_ratios / k
        
    # 5. Return array of LOF scores
    return lof_scores


def calculate_lof_vectorized(neighborhood_indices: np.ndarray, lrd_array: np.ndarray) -> np.ndarray:
    """
    Calculate Local Outlier Factor (LOF) using efficient vectorized NumPy operations.
    
    Args:
        neighborhood_indices (np.ndarray): An N x k array containing indices of k-nearest neighbors.
        lrd_array (np.ndarray): A 1D array of length N containing the Local Reachability Density.
        
    Returns:
        np.ndarray: A 1D array of length N containing the LOF scores.
    """
    k = neighborhood_indices.shape[1]
    
    # Advanced indexing: fetch the LRD values for all neighbors at once.
    # Resulting shape is (N, k)
    lrd_neighbors = lrd_array[neighborhood_indices]
    
    # Sum the LRD values of the neighbors across the rows (axis=1).
    # Resulting shape is (N,)
    sum_lrd_neighbors = np.sum(lrd_neighbors, axis=1)
    
    # Compute the final LOF score for all points at once.
    # We broadcast the division over the (N,) arrays.
    lof_scores = sum_lrd_neighbors / (k * lrd_array)
    
    return lof_scores
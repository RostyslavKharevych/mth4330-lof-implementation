import numpy as np

def calculate_lrd_loops(
    neightborhood_indices: np.ndarray,
    reachability_distances: np.ndarray,
    epsilon: float = 1e-10
) -> np.ndarray:

    """
    Calculates the Local Reachability Density (LRD) for each point using loops.
    
    Args:
        neighborhood_indices: An (N, k) array of neighbor indices for each point.
        reachability_distances: An (N, N) distance matrix where entry [p, o] is 
                                the reachability distance between p and o.
        epsilon: A small constant to prevent division by zero.
        
    Returns:
        An array of size N containing the LRD value for each point.
    """

    n_points, k_neighbors = neighborhood_indices.shape

    lrd = np.zeros(n_points)

    for p in range(n_points):
        sum_reach_dist = 0.0
        
        # 1. Sum reachability distances to all k neighbors
        for o in neighborhood_indices[p]:
            sum_reach_dist += reachability_distances[p, o]

        # 2 & 3. Stabilize denominator with epsilon and calculate LRD
        lrd[p] = k_neighbors / (sum_reach_dist + epsilon)
    
    # 4. Return array of LRD values
    return lrd
   
def calculate_lrd_loops(
    neightborhood_indices: np.ndarray,
    reachability_distances: np.ndarray,
    epsilon: float = 1e-10
) -> np.ndarray:

    """
    Calculates the Local Reachability Density (LRD) for each point using vectorized NumPy operations.
    
    Args:
        neighborhood_indices: An (N, k) array of neighbor indices for each point.
        reachability_distances: An (N, N) distance matrix where entry [p, o] is 
                                the reachability distance between p and o.
        epsilon: A small constant to prevent division by zero.
        
    Returns:
        An array of size N containing the LRD value for each point.
    """

    n_points, k_neighbors = neighborhood_indices.shape

    # Create an (N, 1) column vector of row indices to pair with the (N, k) neighbor indices
    row_indices = np.arrange(n_points)[:, np.newaxis]

    # 1. Extract the reachability distances for all neighbors simultaneously
    # This results in an (N, k) matrix of distances
    neighborhood_reach_dists = reachability_distances[row_indices, neightborhood_indices]

    # Sum across the rows (axis=1) to get the total reachability distance per point
    sum_reach_dist = np.sum(neighborhood_reach_dists, axis = 1)

    # 2 & 3. Stabilize denominator with epsilon and calculate LRD for all points at once
    lrd = k_neighbors / (sum_reach_dist + epsilon)

    # 4. Return array of LRD values
    return lrd
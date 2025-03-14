from collections import deque

def bfs_shortest_path(G, source, target):
    """
    Implement BFS algorithm to find the shortest path between source and target nodes.
    
    Parameters:
    -----------
    G : networkx.Graph
        Input network
    source : node
        Starting node
    target : node
        Target node
        
    Returns:
    --------
    tuple
        (path_length, path) where:
        - path_length is the length of the shortest path (int)
        - path is the list of nodes in the shortest path (list)
        
    If no path exists, return (float('inf'), [])
    """
    visited = set()
    queue = deque()

    # Add the first node to the queue and hashset
    visited.add(source)
    queue.append((source, [source], 0))

    while queue:
        node, path, distance = queue.popleft()

        if node == target:
            return distance, path

        for n in G[node]:
            if n not in visited:
                queue.append((n, path + [n], distance + 1))
                visited.add(n)

    # No path found
    return float('inf'), []
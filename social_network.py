import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from bfs_algorithm import (bfs_shortest_path)


# 1. Corporate/Organizational Network
def generate_corporate_network(num_employees=300, num_departments=5, hierarchy_levels=4):
    """
    Generate a hierarchical corporate network with departments and management layers.
    
    Parameters:
    -----------
    num_employees : int
        Total number of employees
    num_departments : int
        Number of departments
    hierarchy_levels : int
        Number of management levels
        
    Returns:
    --------
    networkx.Graph
        The generated corporate network
    """
    G = nx.Graph()
    
    # Set up basic structure
    employees_per_dept = num_employees // num_departments
    remaining = num_employees % num_departments
    
    # Add CEO (node 0)
    G.add_node(0, level=0, department="Executive", position="CEO")
    
    # Department heads (level 1)
    dept_heads = []
    node_id = 1
    
    for d in range(num_departments):
        dept_name = f"Dept_{d}"
        G.add_node(node_id, level=1, department=dept_name, position="Department Head")
        G.add_edge(0, node_id)  # Connect to CEO
        dept_heads.append(node_id)
        node_id += 1
    
    # Assign managers and employees to departments
    for dept_id, dept_head in enumerate(dept_heads):
        dept_name = f"Dept_{dept_id}"
        
        # Calculate employees in this department
        dept_size = employees_per_dept + (1 if dept_id < remaining else 0)
        
        # Allocate managers (levels 2 to hierarchy_levels-1)
        managers_per_level = {}
        
        # Determine number of managers at each level
        for level in range(2, hierarchy_levels):
            if level == 2:
                # Mid-level managers
                managers_per_level[level] = max(1, dept_size // 20)
            else:
                # Lower-level managers
                managers_per_level[level] = max(1, dept_size // 10)
        
        # Add managers level by level
        current_level_nodes = [dept_head]
        for level in range(2, hierarchy_levels):
            next_level_nodes = []
            managers_at_level = managers_per_level[level]
            
            # Distribute managers among higher-level managers
            for i, higher_mgr in enumerate(current_level_nodes):
                # Calculate how many managers report to this higher manager
                n_reports = managers_at_level // len(current_level_nodes)
                if i < managers_at_level % len(current_level_nodes):
                    n_reports += 1
                
                for _ in range(n_reports):
                    if managers_at_level > 0:
                        G.add_node(node_id, level=level, department=dept_name, position=f"Manager_L{level}")
                        G.add_edge(higher_mgr, node_id)
                        next_level_nodes.append(node_id)
                        node_id += 1
                        managers_at_level -= 1
            
            current_level_nodes = next_level_nodes
        
        # Calculate remaining employees (individual contributors)
        remaining_employees = dept_size - sum(managers_per_level.values())
        
        # Add individual contributors (level = hierarchy_levels)
        for manager in current_level_nodes:
            # Distribute employees among managers
            n_reports = remaining_employees // len(current_level_nodes)
            extra = 0
            if current_level_nodes.index(manager) < remaining_employees % len(current_level_nodes):
                extra = 1
            
            for _ in range(n_reports + extra):
                if remaining_employees > 0:
                    G.add_node(node_id, level=hierarchy_levels, department=dept_name, position="Employee")
                    G.add_edge(manager, node_id)
                    
                    # Add some horizontal connections (coworkers in same department)
                    potential_connections = [n for n in G.nodes() 
                                           if G.nodes[n]['level'] == hierarchy_levels and 
                                              G.nodes[n]['department'] == dept_name and 
                                              n != node_id]
                    
                    # Connect to 1-3 colleagues
                    for _ in range(min(random.randint(1, 3), len(potential_connections))):
                        colleague = random.choice(potential_connections)
                        G.add_edge(node_id, colleague)
                        potential_connections.remove(colleague)
                    
                    node_id += 1
                    remaining_employees -= 1
    
    # Add cross-departmental connections for collaboration
    for _ in range(int(num_employees * 0.05)):  # 5% of network size
        dept1, dept2 = random.sample(range(num_departments), 2)
        
        nodes_dept1 = [n for n in G.nodes() if G.nodes[n].get('department') == f"Dept_{dept1}"]
        nodes_dept2 = [n for n in G.nodes() if G.nodes[n].get('department') == f"Dept_{dept2}"]
        
        if nodes_dept1 and nodes_dept2:
            node1 = random.choice(nodes_dept1)
            node2 = random.choice(nodes_dept2)
            G.add_edge(node1, node2)
    
    # Add node attributes for visualization
    max_degree = max(dict(G.degree()).values())
    
    for node in G.nodes():
        level = G.nodes[node]['level']
        
        # Node size based on hierarchy (higher = larger)
        level_size = (hierarchy_levels - level + 1) * 5
        G.nodes[node]['size'] = level_size + G.degree(node)
        
        # Node importance
        G.nodes[node]['importance'] = (hierarchy_levels - level) / hierarchy_levels
    
    return G

# 2. Academic Collaboration Network
def generate_academic_network(num_researchers=300, num_disciplines=6, avg_collaborators=8):
    """
    Generate an academic collaboration network with disciplines and research clusters.
    
    Parameters:
    -----------
    num_researchers : int
        Total number of researchers
    num_disciplines : int
        Number of academic disciplines
    avg_collaborators : int
        Average number of collaborators per researcher
        
    Returns:
    --------
    networkx.Graph
        The generated academic network
    """
    G = nx.Graph()
    
    # Create disciplines
    disciplines = [f"Discipline_{i}" for i in range(num_disciplines)]
    
    # Create subdisciplines within each discipline
    subdisciplines = {}
    for disc in disciplines:
        n_subdisciplines = random.randint(2, 4)
        subdisciplines[disc] = [f"{disc}_Sub_{i}" for i in range(n_subdisciplines)]
    
    # Flatten subdisciplines for assignment
    all_subdisciplines = []
    for sublist in subdisciplines.values():
        all_subdisciplines.extend(sublist)
    
    # Create researchers and assign to subdisciplines
    for i in range(num_researchers):
        # Primary subdiscipline
        primary_subdiscipline = random.choice(all_subdisciplines)
        
        # Find parent discipline
        parent_discipline = None
        for disc, sub_list in subdisciplines.items():
            if primary_subdiscipline in sub_list:
                parent_discipline = disc
                break
        
        # Assign 1-2 secondary subdisciplines (sometimes from other disciplines)
        secondary_subdisciplines = []
        for _ in range(random.randint(1, 2)):
            if random.random() < 0.7:  # 70% chance to pick from same discipline
                same_disc_subs = subdisciplines[parent_discipline]
                sec_sub = random.choice(same_disc_subs)
            else:
                # Pick from different discipline
                other_discs = [d for d in disciplines if d != parent_discipline]
                if other_discs:
                    other_disc = random.choice(other_discs)
                    sec_sub = random.choice(subdisciplines[other_disc])
                else:
                    sec_sub = random.choice(all_subdisciplines)
            
            if sec_sub != primary_subdiscipline and sec_sub not in secondary_subdisciplines:
                secondary_subdisciplines.append(sec_sub)
        
        # Create node with attributes
        G.add_node(i, 
                   discipline=parent_discipline,
                   primary_subdiscipline=primary_subdiscipline,
                   secondary_subdisciplines=secondary_subdisciplines,
                   papers=random.randint(5, 50),
                   seniority=random.choice(["Junior", "Mid-Career", "Senior", "Distinguished"]))
    
    # Create collaborative relationships
    for i in range(num_researchers):
        # Get researcher's information
        researcher_discipline = G.nodes[i]['discipline']
        researcher_primary = G.nodes[i]['primary_subdiscipline']
        researcher_secondary = G.nodes[i]['secondary_subdisciplines']
        all_researcher_subdisciplines = [researcher_primary] + researcher_secondary
        
        # Determine number of collaborators for this researcher
        n_collaborators = int(np.random.normal(avg_collaborators, avg_collaborators/3))
        n_collaborators = max(1, min(n_collaborators, num_researchers-1))
        
        # Find potential collaborators
        potential_collaborators = []
        
        # Collaboration within same primary subdiscipline (highly likely)
        same_primary = [j for j in range(num_researchers) 
                       if j != i and G.nodes[j]['primary_subdiscipline'] == researcher_primary]
        potential_collaborators.extend([(j, 0.7) for j in same_primary])
        
        # Collaboration from secondary subdisciplines
        for sec_sub in researcher_secondary:
            matches = [j for j in range(num_researchers) 
                      if j != i and (G.nodes[j]['primary_subdiscipline'] == sec_sub or 
                                    sec_sub in G.nodes[j]['secondary_subdisciplines'])]
            potential_collaborators.extend([(j, 0.5) for j in matches])
        
        # Some cross-disciplinary collaboration
        other_researchers = [j for j in range(num_researchers) 
                           if j != i and G.nodes[j]['discipline'] != researcher_discipline]
        sampled_others = random.sample(other_researchers, min(10, len(other_researchers)))
        potential_collaborators.extend([(j, 0.1) for j in sampled_others])
        
        # Remove duplicates
        seen = set()
        unique_potential = []
        for collab, weight in potential_collaborators:
            if collab not in seen and not G.has_edge(i, collab):
                seen.add(collab)
                unique_potential.append((collab, weight))
        
        # Select actual collaborators
        for collab, weight in unique_potential:
            if random.random() < weight and G.degree(i) < n_collaborators:
                G.add_edge(i, collab)
                
                # Sometimes add co-collaborators (triadic closure)
                if random.random() < 0.3:
                    mutual_candidates = list(set(G.neighbors(i)) & set(G.neighbors(collab)))
                    for mutual in mutual_candidates[:2]:  # Add up to 2 mutual collaborators
                        if not G.has_edge(collab, mutual):
                            G.add_edge(collab, mutual)
    
    # Create "star" researchers with many collaborations
    senior_researchers = [i for i in range(num_researchers) 
                        if G.nodes[i]['seniority'] in ["Senior", "Distinguished"]]
    
    for star in random.sample(senior_researchers, min(5, len(senior_researchers))):
        # Add extra collaborations to create academic stars
        potential_new_collabs = [j for j in range(num_researchers) 
                              if j != star and not G.has_edge(star, j)]
        
        new_collabs = random.sample(potential_new_collabs, 
                                  min(random.randint(10, 20), len(potential_new_collabs)))
        
        for collab in new_collabs:
            G.add_edge(star, collab)
    
    # Add node attributes for visualization
    centrality = nx.degree_centrality(G)
    
    for node in G.nodes():
        # Node size based on number of papers and centrality
        papers = G.nodes[node]['papers']
        G.nodes[node]['size'] = 10 + (papers / 5) + (centrality[node] * 100)
        
        # Node importance based on seniority and connections
        seniority_value = {"Junior": 0.25, "Mid-Career": 0.5, "Senior": 0.75, "Distinguished": 1.0}
        G.nodes[node]['importance'] = (seniority_value[G.nodes[node]['seniority']] + centrality[node]) / 2
    
    return G

# 3. Small-World Friend Network
def generate_friend_network(num_people=300, k=10, p=0.05):
    """
    Generate a small-world friendship network based on Watts-Strogatz model.
    
    Parameters:
    -----------
    num_people : int
        Number of people in the network
    k : int
        Each node is connected to k nearest neighbors in ring topology (must be even)
    p : float
        Probability of rewiring each edge
        
    Returns:
    --------
    networkx.Graph
        The generated friend network
    """
    # Create base Watts-Strogatz small-world graph
    G = nx.watts_strogatz_graph(n=num_people, k=k, p=p, seed=42)
    
    # Add geographical regions
    num_regions = 8
    people_per_region = num_people // num_regions
    
    for i in range(num_people):
        region = i // people_per_region
        if region >= num_regions:
            region = num_regions - 1
            
        G.nodes[i]['region'] = f"Region_{region}"
        
        # Add some demographic attributes
        G.nodes[i]['age'] = random.randint(18, 80)
        G.nodes[i]['interests'] = random.sample(
            ["Sports", "Music", "Art", "Technology", "Travel", "Food", "Books", "Movies"],
            random.randint(2, 4)
        )
    
    # Add some strong friendship cliques within regions
    for region in range(num_regions):
        region_people = [i for i in range(num_people) if G.nodes[i]['region'] == f"Region_{region}"]
        
        # Create 2-4 friend groups per region
        num_groups = random.randint(2, 4)
        for _ in range(num_groups):
            group_size = random.randint(3, 8)
            if len(region_people) >= group_size:
                friend_group = random.sample(region_people, group_size)
                
                # Create a clique (everyone connected to everyone)
                for i in range(len(friend_group)):
                    for j in range(i+1, len(friend_group)):
                        G.add_edge(friend_group[i], friend_group[j])
    
    # Add interest-based connections (across regions)
    for interest in ["Sports", "Music", "Art", "Technology", "Travel", "Food", "Books", "Movies"]:
        interest_people = [i for i in range(num_people) if interest in G.nodes[i]['interests']]
        
        # Create some connections between people with the same interest
        num_connections = min(len(interest_people) // 2, 20)
        for _ in range(num_connections):
            person1, person2 = random.sample(interest_people, 2)
            G.add_edge(person1, person2)
    
    # Calculate betweenness for finding "brokers" between social circles
    betweenness = nx.betweenness_centrality(G, k=min(50, num_people))
    
    # Add node attributes for visualization
    for node in G.nodes():
        # Node size based on number of connections
        G.nodes[node]['size'] = 10 + 2 * G.degree(node)
        
        # Node importance based on betweenness centrality (social brokers)
        G.nodes[node]['importance'] = betweenness[node] / max(betweenness.values())
    
    return G

# 4. Scale-Free Online Social Network
def generate_online_social_network(num_users=300, m=2, growth_steps=None):
    """
    Generate a scale-free online social network based on Barabási–Albert model
    with controlled sparsity to ensure realistic path lengths.
    
    Parameters:
    -----------
    num_users : int
        Number of users in the network
    m : int
        Number of edges to attach from a new node to existing nodes
    growth_steps : int
        Additional growth steps to add after initial network formation
        
    Returns:
    --------
    networkx.Graph
        The generated online social network
    """
    # Start with a small initial connected graph
    initial_nodes = max(m+1, 5)  # Need at least m+1 initial nodes
    G = nx.Graph()
    
    # Create initial nodes with a sparse connection pattern
    for i in range(initial_nodes):
        G.add_node(i)
    
    # Connect initial nodes in a ring topology (sparse)
    for i in range(initial_nodes):
        G.add_edge(i, (i+1) % initial_nodes)
    
    # Add a few random edges to ensure connectivity but keep it sparse
    for _ in range(initial_nodes // 2):
        i, j = random.sample(range(initial_nodes), 2)
        if not G.has_edge(i, j):
            G.add_edge(i, j)
    
    # Now grow the network using preferential attachment (Barabási–Albert principle)
    # but maintain better control over edge count
    degrees = dict(G.degree())
    
    # Add remaining nodes with preferential attachment
    for i in range(initial_nodes, num_users):
        # Add node
        G.add_node(i)
        
        # Connect to m existing nodes with preferential attachment
        targets = list(range(i))  # All existing nodes
        
        # Calculate connection probability based on degree
        weights = [degrees.get(t, 1) for t in targets]
        total_weight = sum(weights)
        probs = [w/total_weight for w in weights]
        
        # Select m targets without replacement
        if len(targets) >= m:
            selected_targets = np.random.choice(targets, size=m, replace=False, p=probs)
            
            # Add edges
            for target in selected_targets:
                G.add_edge(i, target)
                
                # Update degrees
                degrees[i] = degrees.get(i, 0) + 1
                degrees[target] = degrees.get(target, 0) + 1
    
    # Add user attributes
    for i in range(num_users):
        G.nodes[i]['join_date'] = f"2023-{random.randint(1, 12)}-{random.randint(1, 28)}"
        G.nodes[i]['activity_level'] = random.choice(["Low", "Medium", "High", "Very High"])
        G.nodes[i]['content_creator'] = random.random() < 0.2  # 20% are content creators
    
    # Add a controlled number of additional connections through growth steps
    if growth_steps is None:
        growth_steps = num_users // 20  # Reduced from //10 to //20
    
    # Add sparse additional growth
    edge_count = 0
    max_additional_edges = min(growth_steps, num_users // 5)  # Limit additional edges
    
    while edge_count < max_additional_edges:
        # Select source randomly from all nodes
        source = random.randint(0, num_users-1)
        
        # Select target based on degree (preferential attachment)
        targets = list(G.nodes())
        targets.remove(source)  # Can't connect to self
        
        # Remove already connected nodes
        targets = [t for t in targets if not G.has_edge(source, t)]
        
        if not targets:
            continue
            
        # Calculate connection probability based on degree
        weights = [degrees.get(t, 1) for t in targets]
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]
        
        # Select target based on degree
        target = np.random.choice(targets, p=weights)
        
        # Add edge
        G.add_edge(source, target)
        
        # Update degrees
        degrees[source] = degrees.get(source, 0) + 1
        degrees[target] = degrees.get(target, 0) + 1
        
        edge_count += 1
    
    # Add some clustering among nearby users (friend-of-friend connections)
    # But limit to prevent excessive connectivity
    triadic_closures = 0
    max_triadic_closures = num_users // 10  # Limit triadic closures
    
    for node in list(G.nodes()):
        if triadic_closures >= max_triadic_closures:
            break
            
        neighbors = list(G.neighbors(node))
        if len(neighbors) < 2:
            continue
            
        # Limit the number of potential triadic closures per node
        max_closures_per_node = min(3, len(neighbors) * (len(neighbors) - 1) // 2)
        closures_for_node = 0
        
        # Try to create some triadic closures
        pairs = []
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                if not G.has_edge(neighbors[i], neighbors[j]):
                    pairs.append((neighbors[i], neighbors[j]))
        
        # Randomly select a limited number of pairs
        random.shuffle(pairs)
        for i, j in pairs[:max_closures_per_node]:
            if triadic_closures >= max_triadic_closures:
                break
                
            if random.random() < 0.2 and not G.has_edge(i, j):  # Reduced probability from 0.3 to 0.2
                G.add_edge(i, j)
                triadic_closures += 1
                closures_for_node += 1
                
                if closures_for_node >= max_closures_per_node:
                    break
    
    # Add "influencer" attribute based on centrality
    # Using degree centrality is faster for large networks than betweenness
    centrality = nx.degree_centrality(G)
    top_influencers = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:int(num_users*0.05)]
    for node, _ in top_influencers:
        G.nodes[node]['influencer'] = True
    
    # Add visualization attributes
    for node in G.nodes():
        # Size based on degree
        G.nodes[node]['size'] = 10 + 3 * G.degree(node)
        
        # Importance based on centrality and content creator status
        creator_bonus = 0.3 if G.nodes[node].get('content_creator', False) else 0
        influencer_bonus = 0.5 if G.nodes[node].get('influencer', False) else 0
        G.nodes[node]['importance'] = centrality[node] + creator_bonus + influencer_bonus
    
    return G

# Visualization function (no node names shown)
def visualize_network(G, title="Social Network", output_file=None, figsize=(12, 10)):
    """
    Visualize the social network with node sizes based on centrality
    and colors based on importance/influence.
    
    Parameters:
    -----------
    G : networkx.Graph
        The social network to visualize
    title : str
        Title for the visualization
    output_file : str, optional
        If provided, save the visualization to this file
    figsize : tuple
        Figure size in inches (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Use a spring layout for natural-looking graph
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    
    # Extract node attributes for visualization
    node_sizes = [G.nodes[node].get('size', 20) for node in G.nodes()]
    
    # Create color map (yellow for highest importance)
    node_colors = []
    for node in G.nodes():
        importance = G.nodes[node].get('importance', 0)
        
        if importance > 0.8:
            # Yellow for high-importance nodes
            node_colors.append('#FFD700')  # Gold
        elif importance > 0.5:
            # Light blue for medium-high importance
            node_colors.append('#00BFFF')  # Deep sky blue
        else:
            # Aqua for regular nodes
            node_colors.append('#00CED1')  # Dark turquoise
    
    # Draw nodes with size and color
    nx.draw_networkx_nodes(G, pos, 
                          node_size=node_sizes, 
                          node_color=node_colors,
                          alpha=0.9)
    
    # Draw edges with transparency
    nx.draw_networkx_edges(G, pos, alpha=0.15)
    
    # Don't draw labels (as requested)
    
    # Remove axes
    plt.axis('off')
    plt.title(title, fontsize=16)
    
    # Save if output file provided
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Visualization saved to {output_file}")
    
    plt.tight_layout()
    plt.show()

def calculate_network_statistics(G, use_all_pairs=True):
    """
    Calculate statistics about the network to demonstrate six degrees of separation.
    
    Parameters:
    -----------
    G : networkx.Graph
        The social network
    use_all_pairs : bool
        If True, calculate for all possible node pairs
        
    Returns:
    --------
    dict
        Dictionary containing various statistics
    """
    stats = {}
    
    # Basic network statistics
    stats['num_nodes'] = G.number_of_nodes()
    stats['num_edges'] = G.number_of_edges()
    stats['avg_degree'] = sum(dict(G.degree()).values()) / G.number_of_nodes()
    
    # Calculate path lengths between all pairs of nodes
    path_lengths = []
    degrees_of_separation = []
    
    all_nodes = list(G.nodes())
    total_pairs = 0
    reachable_pairs = 0
    
    # Create all node pairs to process
    node_pairs = []
    if use_all_pairs:
        # Create all possible pairs (n²-n)/2 pairs
        for i in range(len(all_nodes)):
            for j in range(i+1, len(all_nodes)):
                node_pairs.append((all_nodes[i], all_nodes[j]))
    else:
        # Use a sample if the network is too large (> 1000 nodes)
        sample_size = min(1000000, len(all_nodes) * (len(all_nodes) - 1) // 2)
        for _ in range(sample_size):
            i, j = random.sample(range(len(all_nodes)), 2)
            node_pairs.append((all_nodes[i], all_nodes[j]))
    
    # Process in batches with progress bar
    pair_degrees = {}  # Store all pair degrees for frequency counting
    
    for source, target in tqdm(node_pairs, desc="Calculating all pair distances"):
        total_pairs += 1
        
        # Use the provided BFS function
        path_length, _ = bfs_shortest_path(G, source, target)
        
        if path_length != float('inf'):
            reachable_pairs += 1
            degrees = path_length - 1
            path_lengths.append(path_length)
            degrees_of_separation.append(degrees)
            
            # Store for pair degree distribution
            pair_degrees[degrees] = pair_degrees.get(degrees, 0) + 1
    
    # Calculate statistics about path lengths
    stats['total_pairs'] = total_pairs
    stats['reachable_pairs'] = reachable_pairs
    stats['avg_path_length'] = sum(path_lengths) / len(path_lengths) if path_lengths else float('inf')
    stats['avg_degrees_of_separation'] = sum(degrees_of_separation) / len(degrees_of_separation) if degrees_of_separation else float('inf')
    stats['max_degrees_of_separation'] = max(degrees_of_separation) if degrees_of_separation else 0
    
    # Store complete pair degree distribution
    stats['pair_degrees'] = pair_degrees
    
    # Calculate distribution of degrees of separation with frequency counts
    # (not percentages this time)
    stats['degrees_of_separation_distribution'] = pair_degrees
    
    # Calculate percentage of nodes reachable within 6 degrees
    within_six_degrees = sum(pair_degrees.get(i, 0) for i in range(1, 7))
    stats['percentage_within_six_degrees'] = within_six_degrees / reachable_pairs * 100 if reachable_pairs else 0
    
    # Clustering coefficient (measures how nodes tend to cluster together)
    stats['avg_clustering_coefficient'] = nx.average_clustering(G)
    
    return stats

def plot_degrees_distribution(stats, title="Distribution of Degrees of Separation", output_file=None):
    """
    Plot the distribution of degrees of separation.
    
    Parameters:
    -----------
    stats : dict
        Dictionary containing the degrees of separation distribution
    title : str
        Title for the plot
    output_file : str, optional
        If provided, save the plot to this file
    """
    plt.figure(figsize=(12, 8))
    
    # Get raw counts for each degree of separation
    pair_degrees = stats['pair_degrees']
    degrees = sorted(pair_degrees.keys())
    counts = [pair_degrees[d] for d in degrees]
    
    # Convert to DataFrame for better plotting
    df = pd.DataFrame({
        'Degrees of Separation': degrees,
        'Number of Pairs': counts
    })
    
    # Create bar plot with exact counts
    ax = sns.barplot(x='Degrees of Separation', y='Number of Pairs', data=df)
    
    # Add count labels on top of each bar
    for i, count in enumerate(counts):
        ax.text(i, count + (max(counts) * 0.01), f"{count:,}", 
                ha='center', va='bottom', fontsize=10)
    
    # Add a horizontal line for the mean
    mean_degree = stats['avg_degrees_of_separation']
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add descriptive title and labels
    plt.title(title, fontsize=14)
    plt.xlabel('Degrees of Separation', fontsize=12)
    plt.ylabel('Number of Pairs', fontsize=12)
    
    # Add detailed statistics
    info_text = (
        f"Network Size: {stats['num_nodes']} nodes, {stats['num_edges']} edges\n"
        f"Total Pairs Analyzed: {stats['total_pairs']:,}\n"
        f"Mean Degree of Separation: {mean_degree:.2f}\n"
        f"Maximum Degree of Separation: {stats['max_degrees_of_separation']}\n"
        f"Pairs within 6 Degrees: {stats['percentage_within_six_degrees']:.1f}%"
    )
    
    # Place text box with statistics
    plt.annotate(info_text, xy=(0.97, 0.97), xycoords='axes fraction',
                ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Distribution plot saved to {output_file}")
    
    plt.show()

def print_statistics(stats, network_name="Network"):
    """
    Print the statistics in a readable format.
    
    Parameters:
    -----------
    stats : dict
        Dictionary containing the statistics
    network_name : str
        Name of the network for reporting
    """
    print(f"\n===== {network_name.upper()} STATISTICS =====")
    print(f"Number of nodes: {stats['num_nodes']}")
    print(f"Number of edges: {stats['num_edges']}")
    print(f"Average node degree: {stats['avg_degree']:.2f}")
    print(f"Average clustering coefficient: {stats['avg_clustering_coefficient']:.4f}")
    
    print("\n=== SIX DEGREES OF SEPARATION ANALYSIS ===")
    print(f"Total node pairs analyzed: {stats['total_pairs']:,}")
    print(f"Reachable pairs: {stats['reachable_pairs']:,}")
    print(f"Mean path length: {stats['avg_path_length']:.3f}")
    print(f"Mean degrees of separation: {stats['avg_degrees_of_separation']:.3f}")
    print(f"Maximum degrees of separation: {stats['max_degrees_of_separation']}")
    print(f"Percentage of pairs within 6 degrees: {stats['percentage_within_six_degrees']:.2f}%")
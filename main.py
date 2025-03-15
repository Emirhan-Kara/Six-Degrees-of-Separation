import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import networkx as nx
import itertools
from social_network import (
    generate_corporate_network,
    generate_academic_network,
    generate_friend_network,
    generate_online_social_network,
    calculate_network_statistics,
    print_statistics
)

def ensure_directory_exists(file_path):
    """
    Ensure that the directory for the given file path exists.
    
    Parameters:
    -----------
    file_path : str
        Path to a file
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return os.path.normpath(file_path)  # Normalize path for platform compatibility

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
        # Ensure directory exists and normalize path
        output_file = ensure_directory_exists(output_file)
        try:
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            print(f"Visualization saved to {output_file}")
        except Exception as e:
            print(f"Error saving visualization: {e}")
            # Try with a simplified filename as fallback
            try:
                simple_filename = os.path.join(os.path.dirname(output_file), 
                                              f"network_{title.replace(' ', '_').lower()}.png")
                plt.savefig(simple_filename, bbox_inches='tight', dpi=300)
                print(f"Visualization saved to {simple_filename} (fallback)")
            except Exception as e2:
                print(f"Failed to save visualization with fallback: {e2}")
    
    plt.tight_layout()
    plt.show()

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
    
    # Calculate the actual maximum degree from the data
    actual_max_degree = max(degrees) if degrees else 0
    
    # Convert to DataFrame for better plotting
    df = pd.DataFrame({
        'Degrees of Separation': degrees,
        'Number of Pairs': counts
    })
    
    # Create bar plot with exact counts
    ax = sns.barplot(x='Degrees of Separation', y='Number of Pairs', data=df)
    
    # Add count labels on top of each bar
    for i, count in enumerate(counts):
        ax.text(i, count + (max(counts) * 0.01), f"{int(count):,}", 
                ha='center', va='bottom', fontsize=10)
    
    # Add a horizontal line for the mean
    mean_degree = stats['avg_degrees_of_separation']
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add descriptive title and labels
    plt.title(title, fontsize=14)
    plt.xlabel('Degrees of Separation', fontsize=12)
    plt.ylabel('Number of Pairs', fontsize=12)
    
    # Calculate the total pairs from the actual data
    total_pairs = sum(counts)
    
    # Calculate percentage within 6 degrees from the actual data
    within_six = sum(pair_degrees.get(d, 0) for d in range(7))
    percentage_within_six = (within_six / total_pairs * 100) if total_pairs > 0 else 0
    
    # Add detailed statistics using the actual data from the plot
    info_text = (
        f"Network Size: {int(stats['num_nodes'])} nodes, {int(stats['num_edges'])} edges\n"
        f"Total Pairs Analyzed: {int(total_pairs):,}\n"
        f"Mean Degree of Separation: {mean_degree:.2f}\n"
        f"Maximum Degree of Separation: {actual_max_degree}\n"
        f"Pairs within 6 Degrees: {percentage_within_six:.1f}%"
    )
    
    # Place text box with statistics
    plt.annotate(info_text, xy=(0.97, 0.97), xycoords='axes fraction',
                ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if output_file:
        # Ensure directory exists and normalize path
        output_file = ensure_directory_exists(output_file)
        try:
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            print(f"Distribution plot saved to {output_file}")
        except Exception as e:
            print(f"Error saving plot: {e}")
            # Try with a simplified filename as fallback
            try:
                simple_filename = os.path.join(os.path.dirname(output_file), 
                                              f"distribution_{title.replace(' ', '_').lower()}.png")
                plt.savefig(simple_filename, bbox_inches='tight', dpi=300)
                print(f"Distribution plot saved to {simple_filename} (fallback)")
            except Exception as e2:
                print(f"Failed to save distribution plot with fallback: {e2}")
    
    plt.show()

def plot_comparative_results(avg_stats, output_file=None):
    """
    Create a comparative plot showing key metrics across different network types.
    
    Parameters:
    -----------
    avg_stats : dict
        Dictionary containing averaged statistics for each network type
    output_file : str, optional
        If provided, save the plot to this file
    """
    # Create a DataFrame for plotting
    data = []
    for network_type, stats in avg_stats.items():
        data.append({
            'Network Type': network_type,
            'Average Degree': stats['avg_degree'],
            'Average Path Length': stats['avg_path_length'],
            'Average Degrees of Separation': stats['avg_degrees_of_separation'],
            'Percentage Within 6 Degrees': stats['percentage_within_six_degrees']
        })
    
    df = pd.DataFrame(data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Average Degree
    sns.barplot(x='Network Type', y='Average Degree', data=df, ax=axes[0, 0])
    axes[0, 0].set_title('Average Node Degree by Network Type')
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 2: Average Path Length
    sns.barplot(x='Network Type', y='Average Path Length', data=df, ax=axes[0, 1])
    axes[0, 1].set_title('Average Path Length by Network Type')
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 3: Average Degrees of Separation
    sns.barplot(x='Network Type', y='Average Degrees of Separation', data=df, ax=axes[1, 0])
    axes[1, 0].set_title('Average Degrees of Separation by Network Type')
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 4: Percentage Within 6 Degrees
    sns.barplot(x='Network Type', y='Percentage Within 6 Degrees', data=df, ax=axes[1, 1])
    axes[1, 1].set_title('Percentage of Pairs Within 6 Degrees')
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars for all plots
    for i, ax in enumerate(axes.flat):
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom')
    
    plt.suptitle('Comparative Analysis of Different Network Types\n(Parameter Pool Variations)', 
                 fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if output_file:
        # Ensure directory exists and normalize path
        output_file = ensure_directory_exists(output_file)
        try:
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            print(f"Comparative plot saved to {output_file}")
        except Exception as e:
            print(f"Error saving comparative plot: {e}")
            # Try with a simplified filename as fallback
            try:
                simple_filename = os.path.join(os.path.dirname(output_file), "comparative_results.png")
                plt.savefig(simple_filename, bbox_inches='tight', dpi=300)
                print(f"Comparative plot saved to {simple_filename} (fallback)")
            except Exception as e2:
                print(f"Failed to save comparative plot with fallback: {e2}")
    
    plt.show()

def aggregate_statistics(stats_list):
    """
    Aggregate statistics across multiple parameter combinations.
    
    Parameters:
    -----------
    stats_list : list
        List of statistics dictionaries from multiple runs
    
    Returns:
    --------
    dict
        Averaged statistics across all runs
    """
    if not stats_list:
        return {}
    
    # Create a new aggregated stats dictionary
    agg_stats = {}
    
    # Get keys from the first stats dictionary
    keys = stats_list[0].keys()
    
    # Calculate average for numeric values
    for key in keys:
        if key == 'pair_degrees' or key == 'degrees_of_separation_distribution':
            # Combine the degree distribution dictionaries
            combined_dict = {}
            for stats in stats_list:
                for degree, count in stats[key].items():
                    if degree in combined_dict:
                        combined_dict[degree] += count
                    else:
                        combined_dict[degree] = count
            
            # Use sums for pair degrees instead of averages
            agg_stats[key] = combined_dict
        elif isinstance(stats_list[0][key], (int, float)) and key != 'max_degrees_of_separation':
            # Average numeric values
            agg_stats[key] = sum(stats[key] for stats in stats_list) / len(stats_list)
        elif key == 'max_degrees_of_separation':
            # For max degrees, take the max of the maximums
            agg_stats[key] = max(stats[key] for stats in stats_list)
        else:
            # For non-numeric values, just use the first one
            agg_stats[key] = stats_list[0][key]
    
    return agg_stats

def main():
    """
    Generate networks with different parameter combinations for each type
    and calculate aggregate statistics.
    """
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up parameter pools for each network type
    corporate_params = [
        {'num_employees': 150, 'num_departments': 3, 'hierarchy_levels': 3},
        {'num_employees': 150, 'num_departments': 5, 'hierarchy_levels': 4},
        {'num_employees': 200, 'num_departments': 4, 'hierarchy_levels': 3},
        {'num_employees': 200, 'num_departments': 6, 'hierarchy_levels': 5},
        {'num_employees': 250, 'num_departments': 5, 'hierarchy_levels': 4},
        {'num_employees': 300, 'num_departments': 7, 'hierarchy_levels': 5},
        {'num_employees': 300, 'num_departments': 4, 'hierarchy_levels': 3}
    ]
    
    academic_params = [
        {'num_researchers': 150, 'num_disciplines': 3, 'avg_collaborators': 4},
        {'num_researchers': 150, 'num_disciplines': 6, 'avg_collaborators': 7},
        {'num_researchers': 200, 'num_disciplines': 4, 'avg_collaborators': 5},
        {'num_researchers': 200, 'num_disciplines': 7, 'avg_collaborators': 8},
        {'num_researchers': 250, 'num_disciplines': 5, 'avg_collaborators': 6},
        {'num_researchers': 300, 'num_disciplines': 4, 'avg_collaborators': 6},
        {'num_researchers': 300, 'num_disciplines': 7, 'avg_collaborators': 4}
    ]
    
    friend_params = [
        {'num_people': 300, 'k': 4, 'p': 0.01},
        {'num_people': 300, 'k': 8, 'p': 0.1},
        {'num_people': 400, 'k': 6, 'p': 0.05},
        {'num_people': 400, 'k': 10, 'p': 0.2},
        {'num_people': 500, 'k': 4, 'p': 0.05},
        {'num_people': 500, 'k': 10, 'p': 0.01},
        {'num_people': 600, 'k': 8, 'p': 0.2}
    ]
    
    online_params = [
        {'num_users': 150, 'm': 1, 'growth_steps': 10},
        {'num_users': 150, 'm': 3, 'growth_steps': 25},
        {'num_users': 200, 'm': 2, 'growth_steps': 15},
        {'num_users': 200, 'm': 3, 'growth_steps': 35},
        {'num_users': 250, 'm': 1, 'growth_steps': 25},
        {'num_users': 250, 'm': 2, 'growth_steps': 35},
        {'num_users': 300, 'm': 3, 'growth_steps': 15}
    ]
    
    # Dictionary to store statistics for each network type
    all_stats = {
        "Corporate": [],
        "Academic": [],
        "Friend": [],
        "Online Social": []
    }
    
    # ================== Corporate Networks ==================
    print("\n===== GENERATING CORPORATE NETWORKS =====")
    first_corporate = True
    
    for i, params in enumerate(corporate_params):
        print(f"\nCorporate Network {i+1}/{len(corporate_params)}")
        print(f"Parameters: {params}")
        
        # Generate network
        corporate_network = generate_corporate_network(**params)
        
        # Only visualize the first network
        if first_corporate:
            visualize_network(
                corporate_network, 
                title=f"Corporate Network (employees={params['num_employees']}, depts={params['num_departments']}, levels={params['hierarchy_levels']})",
                output_file=os.path.join(output_dir, "corporate_network.png")
            )
            first_corporate = False
        
        # Calculate statistics
        stats_corporate = calculate_network_statistics(
            corporate_network, 
            use_all_pairs=True
        )
        
        # Store statistics
        all_stats["Corporate"].append(stats_corporate)
    
    # ================== Academic Networks ==================
    print("\n===== GENERATING ACADEMIC NETWORKS =====")
    first_academic = True
    
    for i, params in enumerate(academic_params):
        print(f"\nAcademic Network {i+1}/{len(academic_params)}")
        print(f"Parameters: {params}")
        
        # Generate network
        academic_network = generate_academic_network(**params)
        
        # Only visualize the first network
        if first_academic:
            visualize_network(
                academic_network, 
                title=f"Academic Network (researchers={params['num_researchers']}, disc={params['num_disciplines']}, collab={params['avg_collaborators']})",
                output_file=os.path.join(output_dir, "academic_network.png")
            )
            first_academic = False
        
        # Calculate statistics
        stats_academic = calculate_network_statistics(
            academic_network, 
            use_all_pairs=True
        )
        
        # Store statistics
        all_stats["Academic"].append(stats_academic)
    
    # ================== Friend Networks ==================
    print("\n===== GENERATING FRIEND NETWORKS =====")
    first_friend = True
    
    for i, params in enumerate(friend_params):
        print(f"\nFriend Network {i+1}/{len(friend_params)}")
        print(f"Parameters: {params}")
        
        # Generate network
        friend_network = generate_friend_network(**params)
        
        # Only visualize the first network
        if first_friend:
            visualize_network(
                friend_network, 
                title=f"Friend Network (people={params['num_people']}, k={params['k']}, p={params['p']})",
                output_file=os.path.join(output_dir, "friend_network.png")
            )
            first_friend = False
        
        # Calculate statistics
        stats_friend = calculate_network_statistics(
            friend_network, 
            use_all_pairs=True
        )
        
        # Store statistics
        all_stats["Friend"].append(stats_friend)
    
    # ================== Online Social Networks ==================
    print("\n===== GENERATING ONLINE SOCIAL NETWORKS =====")
    first_online = True
    
    for i, params in enumerate(online_params):
        print(f"\nOnline Social Network {i+1}/{len(online_params)}")
        print(f"Parameters: {params}")
        
        # Generate network
        online_network = generate_online_social_network(**params)
        
        # Only visualize the first network
        if first_online:
            visualize_network(
                online_network, 
                title=f"Online Social Network (users={params['num_users']}, m={params['m']}, growth={params['growth_steps']})",
                output_file=os.path.join(output_dir, "online_network.png")
            )
            first_online = False
        
        # Calculate statistics
        stats_online = calculate_network_statistics(
            online_network, 
            use_all_pairs=True
        )
        
        # Store statistics
        all_stats["Online Social"].append(stats_online)
    
    # Calculate aggregate statistics across all parameter combinations
    print("\n===== CALCULATING AGGREGATE STATISTICS =====")
    
    avg_stats = {}
    for network_type, stats_list in all_stats.items():
        avg_stats[network_type] = aggregate_statistics(stats_list)
        
        # Print average statistics for each network type
        print_statistics(avg_stats[network_type], network_name=f"Aggregate {network_type}")
        
        # Create distribution plot for each network type
        plot_degrees_distribution(
            avg_stats[network_type],
            title=f"Degrees of Separation in {network_type} Networks",
            output_file=os.path.join(output_dir, f"{network_type.lower().replace(' ', '_')}_degrees.png")
        )
    
    # Print comparative summary
    print("\n===== COMPARATIVE SUMMARY (ACROSS PARAMETER VARIATIONS) =====")
    print("{:<15} | {:<10} | {:<15} | {:<15} | {:<15}".format(
        "Network Type", "Avg Degree", "Avg Path Length", "Mean Degrees Sep.", "% Within 6 Degrees"))
    print("-" * 75)
    for name, stats in avg_stats.items():
        print("{:<15} | {:<10.2f} | {:<15.2f} | {:<15.2f} | {:<15.2f}%".format(
            name, 
            stats['avg_degree'], 
            stats['avg_path_length'], 
            stats['avg_degrees_of_separation'], 
            stats['percentage_within_six_degrees']))
    
    # Create a combined plot of all network types
    plot_comparative_results(avg_stats, output_file=os.path.join(output_dir, "comparative_results.png"))
    
    # Save a report on the parameter variations
    save_parameter_report(
        corporate_params, 
        academic_params, 
        friend_params, 
        online_params, 
        output_dir
    )

def save_parameter_report(corporate_params, academic_params, friend_params, online_params, output_dir):
    """Save a report on the parameter variations used in the experiment."""
    report_file = os.path.join(output_dir, "parameter_variations_report.txt")
    
    with open(ensure_directory_exists(report_file), 'w') as f:
        f.write("SIX DEGREES OF SEPARATION SIMULATION - PARAMETER VARIATIONS REPORT\n")
        f.write("=================================================================\n\n")
        
        f.write("1. CORPORATE NETWORKS\n")
        f.write("-----------------------\n")
        for i, params in enumerate(corporate_params):
            f.write(f"Variation {i+1}: employees={params['num_employees']}, departments={params['num_departments']}, hierarchy_levels={params['hierarchy_levels']}\n")
        
        f.write("\n2. ACADEMIC NETWORKS\n")
        f.write("-----------------------\n")
        for i, params in enumerate(academic_params):
            f.write(f"Variation {i+1}: researchers={params['num_researchers']}, disciplines={params['num_disciplines']}, avg_collaborators={params['avg_collaborators']}\n")
        
        f.write("\n3. FRIEND NETWORKS\n")
        f.write("-----------------------\n")
        for i, params in enumerate(friend_params):
            f.write(f"Variation {i+1}: people={params['num_people']}, k={params['k']}, p={params['p']}\n")
        
        f.write("\n4. ONLINE SOCIAL NETWORKS\n")
        f.write("-----------------------\n")
        for i, params in enumerate(online_params):
            f.write(f"Variation {i+1}: users={params['num_users']}, m={params['m']}, growth_steps={params['growth_steps']}\n")
        
        f.write("\n\nTotal number of network variations: ")
        f.write(f"{len(corporate_params) + len(academic_params) + len(friend_params) + len(online_params)}\n")
    
    print(f"Parameter variations report saved to {report_file}")

if __name__ == "__main__":
    main()

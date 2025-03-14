import os
from social_network import (
    generate_corporate_network,
    generate_academic_network,
    generate_friend_network,
    generate_online_social_network,
    visualize_network,
    calculate_network_statistics,
    plot_degrees_distribution,
    print_statistics
)

def main():
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Corporate/Organizational Network
    print("\n===== GENERATING CORPORATE NETWORK =====")
    corporate_network = generate_corporate_network(
        num_employees=250,
        num_departments=6,
        hierarchy_levels=4
    )
    
    # Visualize
    visualize_network(
        corporate_network, 
        title="Corporate Network Structure",
        output_file=os.path.join(output_dir, "corporate_network.png")
    )
    
    # Calculate statistics with higher path lengths
    stats_corporate = calculate_network_statistics(
        corporate_network, 
        use_all_pairs=True
    )
    
    # Print and plot
    print_statistics(stats_corporate, network_name="Corporate Network")
    plot_degrees_distribution(
        stats_corporate,
        title="Degrees of Separation in Corporate Networks",
        output_file=os.path.join(output_dir, "corporate_degrees.png")
    )
    
    # 2. Academic Collaboration Network
    print("\n===== GENERATING ACADEMIC NETWORK =====")
    academic_network = generate_academic_network(
        num_researchers=250,
        num_disciplines=6,
        avg_collaborators=6  # Lower for higher path lengths
    )
    
    # Visualize
    visualize_network(
        academic_network, 
        title="Academic Collaboration Network",
        output_file=os.path.join(output_dir, "academic_network.png")
    )
    
    # Calculate statistics
    stats_academic = calculate_network_statistics(
        academic_network, 
        use_all_pairs=True
    )
    
    # Print and plot
    print_statistics(stats_academic, network_name="Academic Network")
    plot_degrees_distribution(
        stats_academic,
        title="Degrees of Separation in Academic Networks",
        output_file=os.path.join(output_dir, "academic_degrees.png")
    )
    
    # 3. Small-World Friend Network
    print("\n===== GENERATING FRIEND NETWORK =====")
    friend_network = generate_friend_network(
        num_people=500,
        k=6,  # Lower k = fewer direct connections
        p=0.05  # Controls small-world property
    )
    
    # Visualize
    visualize_network(
        friend_network, 
        title="Small-World Friendship Network",
        output_file=os.path.join(output_dir, "friend_network.png")
    )
    
    # Calculate statistics
    stats_friend = calculate_network_statistics(
        friend_network, 
        use_all_pairs=True
    )
    
    # Print and plot
    print_statistics(stats_friend, network_name="Friend Network")
    plot_degrees_distribution(
        stats_friend,
        title="Degrees of Separation in Friendship Networks",
        output_file=os.path.join(output_dir, "friend_degrees.png")
    )
    
    # 4. Scale-Free Online Social Network
    print("\n===== GENERATING ONLINE SOCIAL NETWORK =====")
    online_network = generate_online_social_network(
        num_users=250,
        m=1,  # Minimally connected - 1 edge per new node
        growth_steps=25  # Fewer additional connections
    )
    
    # Visualize
    visualize_network(
        online_network, 
        title="Scale-Free Online Social Network",
        output_file=os.path.join(output_dir, "online_network.png")
    )
    
    # Calculate statistics
    stats_online = calculate_network_statistics(
        online_network, 
        use_all_pairs=True
    )
    
    # Print and plot
    print_statistics(stats_online, network_name="Online Social Network")
    plot_degrees_distribution(
        stats_online,
        title="Degrees of Separation in Online Social Networks",
        output_file=os.path.join(output_dir, "online_degrees.png")
    )
    
    # Print comparative summary
    print("\n===== COMPARATIVE SUMMARY =====")
    networks = {
        "Corporate": stats_corporate,
        "Academic": stats_academic,
        "Friend": stats_friend,
        "Online Social": stats_online
    }
    
    print("{:<15} | {:<10} | {:<15} | {:<15} | {:<15}".format(
        "Network Type", "Avg Degree", "Avg Path Length", "Mean Degrees Sep.", "% Within 6 Degrees"))
    print("-" * 75)
    for name, stats in networks.items():
        print("{:<15} | {:<10.2f} | {:<15.2f} | {:<15.2f} | {:<15.2f}%".format(
            name, 
            stats['avg_degree'], 
            stats['avg_path_length'], 
            stats['avg_degrees_of_separation'], 
            stats['percentage_within_six_degrees']))

if __name__ == "__main__":
    main()
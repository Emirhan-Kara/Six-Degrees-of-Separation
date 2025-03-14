# Six Degrees of Separation

This project demonstrates the "Six Degrees of Separation" theory using different types of social network models. It generates four distinct network types and analyzes the degrees of separation between all node pairs.

## Features

- Multiple social network models:
  - Corporate/Organizational Network
  - Academic Collaboration Network
  - Small-World Friend Network
  - Scale-Free Online Social Network
- Comprehensive analysis of path lengths and degrees of separation
- Visualization of network structures
- Statistical comparison between different network types

## Requirements

- Python 3.6+
- Dependencies listed in `requirements.txt`

## Installation

```bash
# Clone the repository
git clone https://github.com/Emirhan-Kara/Six-Degrees-of-Separation
cd Six_Degrees_of_Separation

# Set up a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the main script to generate all four network types and their analyses:

```bash
python main.py
```

The script will:

1. Generate four different social network models
2. Visualize each network
3. Calculate path lengths between all node pairs
4. Generate distribution plots showing degrees of separation
5. Print comparative statistics

## Project Structure

```
six-degrees-of-separation/
├── social_network.py   # Network generation and analysis functions
├── bfs_algorithm.py    # BFS implementation for path finding
├── main.py             # Main script to run the demonstration
├── requirements.txt    # Project dependencies
└── output/             # Generated visualizations and results
```

## Theory Background

The "Six Degrees of Separation" theory suggests that all people are six or fewer social connections away from each other. This project simulates different social structures to test and visualize this concept across various network types.

## License

[MIT License](LICENSE)

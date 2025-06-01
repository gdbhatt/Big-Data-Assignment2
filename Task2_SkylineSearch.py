# SKYLINE SEARCH IMPLEMENTATION - CITY1 DATASET

# This program is applying three algorithms to identify skyline points in a real estate dataset.(CITY1 dataset)
# Skyline points are representing homes that are optimal in both cost and size (i.e., cheaper and larger).
# The three methods being implemented are:
# 1. Implementation of Sequential Scan Method
# 2. Implementation of Branch and Bound Skyline (BBS) using a bulk-loaded R-tree
# 3. Implementation of BBS with Divide-and-Conquer

import time
import sys
import os

# heapq has been used in order to maintain efficient priority queue for expansion of node in Branch and Bound Skyline
import heapq  

# --- STEP 1: LOADING DATASET ---

# Reading and parsing the dataset from the text file and converting each line into a tuple with the structure(ID, cost, size)

def load_dataset(filename):
    """Loading dataset from file with error handling"""
    try:
        # Checking if file exists before attempting to read
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Dataset file '{filename}' not found")
        
        # Opening file and reading data safely
        with open(filename, "r") as f:
            # Converting each line into tuple while handling potential parsing errors
            dataset = []
            for line_num, line in enumerate(f, 1):
                try:
                    # Parsing each line and converting to appropriate data types
                    parts = line.strip().split()
                    if len(parts) != 3:
                        raise ValueError(f"Line {line_num}: Expected 3 values, got {len(parts)}")
                    
                    # Converting to proper data types with validation
                    point_id = int(parts[0])
                    cost = float(parts[1])
                    size = float(parts[2])
                    
                    # Validating that cost and size are positive values
                    if cost <= 0 or size <= 0:
                        raise ValueError(f"Line {line_num}: Cost and size must be positive")
                    
                    dataset.append((point_id, cost, size))
                    
                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping invalid line {line_num} - {e}")
                    continue
            
            # Checking if dataset has enough points for meaningful analysis
            if len(dataset) < 2:
                raise ValueError("Dataset must contain at least 2 points for skyline analysis")
            
            print(f"Loaded {len(dataset)} points from {filename}")
            return dataset
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

# Loading the dataset with error handling
dataset = load_dataset("city1.txt")

# --- STEP 2: CHECKING TO SEE IF A POINT BELONGS IN THE SKYLINE SET ---

# Iterating through current skyline to compare and identify if the new point is dominated
# A point is added if only it is not dominated by any of the current skyline points

def is_skyline_point(point, skyline):
    """
    Checking if a point belongs in skyline by comparing with existing skyline points
    Returning False if point is dominated, True otherwise
    """
    # Validating input parameters
    if not point or len(point) < 3:
        return False
    
    # Iterating through each skyline point to check for domination
    for s in skyline:
        # Checking if skyline point s dominates the test point
        # A point is dominated if s has lower/equal cost AND higher/equal size
        # with at least one dimension being strictly better
        if s[1] <= point[1] and s[2] >= point[2] and (s[1] < point[1] or s[2] > point[2]):
            # Skipping points that are dominated
            return False
    # Returning when point is not dominated    
    return True

# --- STEP 3: IMPLEMENTING BRUTE-FORCE SKYLINE SEARCH ---

# Iterating through each point and comparing with all others to find non-dominated points

def skyline_sequential(dataset):
    """
    Implementing sequential scan method for finding skyline points
    Using brute-force O(n²) approach as baseline algorithm
    """
    # Validating input dataset
    if not dataset:
        return []
    
    # Initializing empty list to store skyline points
    skyline = []  
    
    # Looping through each point in the dataset
    for i, p in enumerate(dataset): 
        # Assuming current point is not dominated initially
        dominated = False 
        
        # Comparing with all other points in dataset
        for j, q in enumerate(dataset):  
            # Skipping comparison with itself
            if i != j and q[1] <= p[1] and q[2] >= p[2]:
                # Checking if point q dominates point p
                if q[1] < p[1] or q[2] > p[2]:
                    # Marking as dominated when better point found
                    dominated = True 
                    # Stopping as soon as domination is found for efficiency
                    break  
        
        # Adding point to skyline if not dominated by any other point
        if not dominated:
            skyline.append(p)  
    
    return skyline

# --- STEP 4: DEFINING NODE STRUCTURE FOR R-TREE ---

# Each node is storing either child nodes or actual data points along with their MBR (Minimum Bounding Rectangle)

class Node:
    """
    Representing a node in the R-tree structure
    Storing entries (points or child nodes) and computing MBR boundaries
    """
    def __init__(self, entries, is_leaf=True):
        # Validating entries parameter
        if entries is None:
            entries = []
        
        self.entries = entries  # Storing child nodes or data points
        self.is_leaf = is_leaf  # Flagging node type (leaf or internal)
        self.mbr = self.compute_mbr()  # Computing MBR upon initialization

    # Computing the Minimum Bounding Rectangle (MBR) for a node
    def compute_mbr(self):
        """
        Computing MBR by finding min/max boundaries of all contained entries
        Handling both leaf nodes (with points) and internal nodes (with child MBRs)
        """
        # Returning extreme MBR values if node is empty
        if not self.entries:
            return (float('inf'), float('-inf'), float('inf'), float('-inf'))

        # Computing MBR differently based on node type
        if self.is_leaf:
            # For leaf nodes: extracting cost and size values from data points
            xs = [e[1] for e in self.entries]  # Collecting cost values from points
            ys = [e[2] for e in self.entries]  # Collecting size values from points
        else:
            # For internal nodes: using MBRs from child nodes
            xs = [e.mbr[0] for e in self.entries] + [e.mbr[1] for e in self.entries]  # Combining min/max costs
            ys = [e.mbr[2] for e in self.entries] + [e.mbr[3] for e in self.entries]  # Combining min/max sizes

        # Returning complete bounding box with (min_cost, max_cost, min_size, max_size)
        return (min(xs), max(xs), min(ys), max(ys))

    # Calculating mindist to guide BBS search prioritization
    def mindist(self):
        """
        Calculating minimum distance heuristic for priority queue ordering
        Using sum of minimum cost and minimum size as search priority
        """
        # Returning sum of minimum cost and minimum size for node prioritization
        return self.mbr[0] + self.mbr[2]

    def __lt__(self, other):
        """Enabling heapq comparison by mindist values"""
        # Allowing heapq to compare nodes by their mindist values
        return self.mindist() < other.mindist()

# --- STEP 5: CONSTRUCTING R-TREE USING BULK LOADING ---

# Sorting dataset by cost and chunking it into leaf nodes, then recursively building the tree

def bulk_load(dataset, max_entries=100):
    """
    Building R-tree using bulk loading method for better performance
    Creating balanced tree structure by sorting and chunking data
    """
    # Handling empty dataset case
    if not dataset:
        return Node([], is_leaf=True)
    
    # Creating copy to avoid modifying original dataset
    data_copy = dataset.copy()
    
    # Sorting dataset based on cost for better spatial locality
    data_copy.sort(key=lambda x: x[1])
    
    # Creating leaf level by chunking dataset into groups
    leaf_nodes = [Node(data_copy[i:i + max_entries], is_leaf=True) 
                  for i in range(0, len(data_copy), max_entries)]
    
    # Building tree level by level from bottom up
    current_level = leaf_nodes
    
    # Continuing until single root node remains
    while len(current_level) > 1:
        # Creating next level by grouping current nodes
        new_level = [Node(current_level[i:i + max_entries], is_leaf=False)
                     for i in range(0, len(current_level), max_entries)]
        
        current_level = new_level  # Updating current level for next iteration

    # Returning root node or empty node if no data
    root = current_level[0] if current_level else Node([], is_leaf=True)
    return root

# --- STEP 6: CHECKING IF MBR IS FULLY DOMINATED BY SKYLINE ---

# Helping in pruning branches during BBS traversal

def mbr_dominated(mbr, skyline):
    """
    Checking if MBR is dominated by any skyline point for pruning
    Enabling early termination of branches that cannot contain skyline points
    """
    # Validating MBR parameter
    if not mbr or len(mbr) < 4:
        return False
    
    # Iterating through skyline points to check for MBR domination
    for p in skyline:
        # Checking if skyline point p dominates the entire MBR region
        # MBR is dominated if point has cost <= min_cost AND size >= max_size
        # with at least one dimension being strictly better
        if p[1] <= mbr[0] and p[2] >= mbr[3] and (p[1] < mbr[0] or p[2] > mbr[3]):
            return True  # Returning true if MBR is dominated
    
    # Returning false if no skyline point dominates the MBR
    return False

# --- STEP 7: RUNNING BBS ALGORITHM USING HEAP ---

# Traversing the tree by expanding most promising (lowest mindist) node first

def skyline_bbs(dataset):
    """
    Implementing Branch and Bound Skyline algorithm using R-tree
    Using priority queue to explore most promising nodes first
    """
    # Validating input dataset
    if not dataset:
        return []
    
    root = bulk_load(dataset)  # Building R-tree using bulk loading
    
    skyline = []  # Initializing empty skyline list
    queue = [(root.mindist(), root)]  # Initializing priority queue with root node
    
    # Processing nodes while priority queue is not empty
    while queue:
        # Removing node with lowest mindist from priority queue
        _, node = heapq.heappop(queue)
        
        # Skipping node if its MBR is dominated by current skyline
        if mbr_dominated(node.mbr, skyline):
            continue
        
        # Processing leaf nodes by examining their points
        if node.is_leaf:
            # Scanning all points in leaf node
            for p in node.entries:
                # Checking if point should be added to skyline
                if is_skyline_point(p, skyline):
                    skyline.append(p)  # Adding valid skyline point
        else:
            # Processing internal nodes by examining their children
            for child in node.entries:  # Checking each child node
                # Adding non-dominated children to priority queue
                if not mbr_dominated(child.mbr, skyline):
                    heapq.heappush(queue, (child.mindist(), child))  # Inserting promising child node
    
    # Ensuring final post-filtering to remove any remaining dominated points
    final_skyline = [p for p in skyline if is_skyline_point(p, skyline)]
    
    return final_skyline

# --- STEP 8: RUNNING DIVIDE-AND-CONQUER BBS ---

# Dividing dataset into halves then computing skyline independently for each half, and finally merging the results

def skyline_bbs_divide_and_conquer(dataset):
    """
    Implementing BBS with Divide-and-Conquer approach
    Splitting dataset, finding subspace skylines, then merging results
    """
    # Validating input dataset
    if not dataset:
        return []
    
    # Handling small datasets by using regular BBS
    if len(dataset) <= 100:
        return skyline_bbs(dataset)
    
    # Sorting dataset by cost for consistent splitting
    sorted_data = sorted(dataset, key=lambda p: p[1])
    
    # Finding midpoint for dividing dataset
    mid = len(sorted_data) // 2
    left = sorted_data[:mid]  # Splitting left half (lower costs)
    right = sorted_data[mid:]  # Splitting right half (higher costs)
    
    # Running BBS on left subspace
    left_sky = skyline_bbs(left)
    
    # Running BBS on right subspace  
    right_sky = skyline_bbs(right)
    
    # Merging skylines from both subspaces
    merged = left_sky + right_sky
    
    # Filtering final skyline points using dominance screening
    final_skyline = [p for p in merged if is_skyline_point(p, merged)]
    
    return final_skyline

# --- STEP 9: RUNNING ALL THE REQUIRED METHODS AND RECORDING THEIR RESULTS ---

def run_all_algorithms():
    """
    Running all three skyline algorithms and collecting performance results
    Measuring execution times and validating results consistency
    """
    print("-" * 50)
    print("STARTING SKYLINE SEARCH ANALYSIS")
    print("-" * 50)
    
    results = []

    # Running Sequential Scan and measuring execution time
    print("\n1. Sequential Scan Method")
    start = time.time()
    
    try:
        seq = skyline_sequential(dataset)
        seq_time = time.time() - start
        results.append(("Sequential Scan", seq, seq_time))
        print(f"   Completed in {seq_time:.4f} seconds")
    except Exception as e:
        print(f"   Error: Sequential Scan failed - {e}")
        return None

    # Running BBS and measuring execution time
    print("\n2. Branch and Bound Skyline (BBS)")
    start = time.time()
    
    try:
        bbs = skyline_bbs(dataset)
        bbs_time = time.time() - start  
        results.append(("BBS", bbs, bbs_time))
        print(f"   Completed in {bbs_time:.4f} seconds")
    except Exception as e:
        print(f"   Error: BBS failed - {e}")
        return None

    # Running Divide-and-Conquer BBS and measuring execution time
    print("\n3. BBS with Divide-and-Conquer")
    start = time.time()
    
    try:
        bbs_dc = skyline_bbs_divide_and_conquer(dataset)
        bbs_dc_time = time.time() - start
        results.append(("BBS with Divide-and-Conquer", bbs_dc, bbs_dc_time))
        print(f"   Completed in {bbs_dc_time:.4f} seconds")
    except Exception as e:
        print(f"   Error: Divide-and-Conquer BBS failed - {e}")
        return None
    
    # Calculating and displaying performance improvements
    print(f"\n" + "-" * 50)
    print("PERFORMANCE ANALYSIS")
    print("-" * 50)
    seq_time = results[0][2]
    for method, skyline, exec_time in results:
        speedup = seq_time / exec_time if exec_time > 0 else float('inf')
        print(f"{method}: {exec_time:.4f}s ({len(skyline)} points, {speedup:.1f}x speedup)")
    
    return results

# Running all algorithms with error handling
try:
    results = run_all_algorithms()
    if results is None:
        sys.exit(1)
except Exception as e:
    print(f"Critical error during algorithm execution: {e}")
    sys.exit(1)

# --- STEP 10: WRITING OUTPUT TO FILE ---

# Writing execution time and skyline points for each method

def write_results_to_file(results, filename="Task2_SkylineSearch_Output.txt"):
    """
    Writing algorithm results to output file with proper formatting
    Including execution times and skyline point details
    """
    try:
        print(f"\nWriting results to {filename}...")
        
        # Opening output file for writing
        with open(filename, "w") as f:
            # Writing results for each algorithm
            for method, skyline, t in results:
                # Writing method header and execution time
                f.write(f"=== {method} ===\n")
                f.write(f"Execution Time: {t:.4f} seconds\n")
                f.write("ID\tX (Cost)\tY (Size)\n")
                
                # Writing each skyline point with proper formatting
                for p in skyline:
                    f.write(f"{p[0]}\t{p[1]:.2f}\t{p[2]:.2f}\n")
                
                # Adding blank line between methods for readability
                f.write("\n")
        
        print(f"Results successfully written to {filename}")
        return True
        
    except IOError as e:
        print(f"Error: Could not write to file {filename} - {e}")
        return False
    except Exception as e:
        print(f"Error: Unexpected error during file writing - {e}")
        return False

# Writing results to file with error handling
if not write_results_to_file(results):
    print("Warning: Results may not have been saved properly")

print("\n" + "=" * 60)
print("SKYLINE SEARCH ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)
print("Skyline Search Completed.")
# SKYLINE SEARCH IMPLEMENTATION - CITY1 DATASET

# This program is applying three algorithms to identify skyline points in a real estate dataset.(CITY1 dataset)
# Skyline points are representing homes that are optimal in both cost and size (i.e., cheaper and larger).
# The three methods being implemented are:
# 1. Implementation of Sequential Scan Method
# 2. Implementation of Branch and Bound Skyline (BBS) using a bulk-loaded R-tree
# 3. Implementation of  BBS with Divide-and-Conquer

import time

# heapq has been used in order to maintain efficient priority queue for expansion of node in Branch and Bound Skyline
import heapq  

# --- STEP 1: LOADING DATASET ---

# Reading and parsing the dataset from the text file and converting each line into a tuple  with the structure(ID, cost, size)

with open("city1.txt", "r") as f:
    dataset = [(int(p[0]), float(p[1]), float(p[2])) for p in (line.strip().split() for line in f)]

# --- STEP 2: CHECKING TO SEE IF A POINT BELONGS IN THE SKYLINE SET ---

# Iterating through current skyline to compare and identify if the new point is dominated
# A point is added if only it is not dominated by any of the current skyline points

def is_skyline_point(point, skyline):
    for s in skyline:
        if s[1] <= point[1] and s[2] >= point[2] and (s[1] < point[1] or s[2] > point[2]):
            # Skipping points that are dominated
            return False
    # Returning when point is not dominated    
    return True

# --- STEP 3: IMPLEMENTING BRUTE-FORCE SKYLINE SEARCH ---

# Iterating through each point and comparing with all others to find non-dominated points

def skyline_sequential(dataset):
    # Initializing empty list to store skyline points
    skyline = []  
    # Looping through each point
    for i, p in enumerate(dataset): 
        # Assuming current point is not dominated
        dominated = False 
        # Comparing with all other points
        for j, q in enumerate(dataset):  
            if i != j and q[1] <= p[1] and q[2] >= p[2]:
                if q[1] < p[1] or q[2] > p[2]:
                    # Marking as dominated
                    dominated = True 
                    # Stopping as soon as domination is found
                    break  
        if not dominated:
            # Adding to skyline if not dominated by any other
            skyline.append(p)  
    return skyline

# --- STEP 4: DEFINING NODE STRUCTURE FOR R-TREE ---

# Each node is storing either child nodes or actual data points along with their MBR (Minimum Bounding Rectangle)

class Node:
    def __init__(self, entries, is_leaf=True):
        self.entries = entries  # Storing child nodes or data points
        self.is_leaf = is_leaf  # Flagging node type
        self.mbr = self.compute_mbr()  # Computing MBR upon initialization

    # Computing the Minimum Bounding Rectangle (MBR) for a node
    def compute_mbr(self):
        if not self.entries:
            return (float('inf'), float('-inf'), float('inf'), float('-inf'))  # Returning extreme MBR if empty

        if self.is_leaf:
            xs = [e[1] for e in self.entries]  # Collecting cost values
            ys = [e[2] for e in self.entries]  # Collecting size values
        else:
            xs = [e.mbr[0] for e in self.entries] + [e.mbr[1] for e in self.entries]  # Using MBRs from children
            ys = [e.mbr[2] for e in self.entries] + [e.mbr[3] for e in self.entries]

        return (min(xs), max(xs), min(ys), max(ys))  # Returning complete bounding box

    # Calculating mindist to guide BBS search prioritization
    def mindist(self):
        return self.mbr[0] + self.mbr[2]  # Returning sum of min cost and min size

    def __lt__(self, other):
        return self.mindist() < other.mindist()  # Allowing heapq to compare nodes by mindist

# --- STEP 5: CONSTRUCTING R-TREE USING BULK LOADING ---

# Sorting dataset by cost and chunking it into leaf nodes, then recursively building the tree

def bulk_load(dataset, max_entries=100):
    dataset.sort(key=lambda x: x[1])  # Sorting dataset based on cost
    leaf_nodes = [Node(dataset[i:i + max_entries], is_leaf=True) for i in range(0, len(dataset), max_entries)]  # Creating leaf level

    # Building tree level by level
    while len(leaf_nodes) > 1:
        new_level = [Node(leaf_nodes[i:i + max_entries], is_leaf=False)
                     for i in range(0, len(leaf_nodes), max_entries)]
        leaf_nodes = new_level  # Updating current level

    return leaf_nodes[0] if leaf_nodes else Node([], is_leaf=True)  # Returning root or empty root

# --- STEP 6: CHECKING IF MBR IS FULLY DOMINATED BY SKYLINE ---

# Helping in pruning branches during BBS traversal

def mbr_dominated(mbr, skyline):
    for p in skyline:
        if p[1] <= mbr[0] and p[2] >= mbr[3] and (p[1] < mbr[0] or p[2] > mbr[3]):
            return True  # Returning true if dominated
    return False

# --- STEP 7: RUNNING BBS ALGORITHM USING HEAP ---

# Traversing the tree by expanding most promising (lowest mindist) node first

def skyline_bbs(dataset):
    root = bulk_load(dataset)  # Building R-tree
    skyline = []  # Initializing skyline list
    queue = [(root.mindist(), root)]  # Initializing priority queue

    while queue:
        _, node = heapq.heappop(queue)  # Removing node with lowest mindist

        if node.is_leaf:
            for p in node.entries:  # Scanning all points
                if is_skyline_point(p, skyline):
                    skyline.append(p)  # Adding valid point
        else:
            for child in node.entries:  # Checking child nodes
                if not mbr_dominated(child.mbr, skyline):
                    heapq.heappush(queue, (child.mindist(), child))  # Inserting non-dominated node

    return [p for p in skyline if is_skyline_point(p, skyline)]  # Ensuring final post-filtering


# --- STEP 8: RUNNING DIVIDE-AND-CONQUER BBS ---

# Dividing dataset into halves then computing skyline independently fo each half, and finally merging the results

def skyline_bbs_divide_and_conquer(dataset):
    sorted_data = sorted(dataset, key=lambda p: p[1])  # Sorting dataset by cost
    mid = len(sorted_data) // 2
    left = sorted_data[:mid]  # Splitting left half
    right = sorted_data[mid:]  # Splitting right half

    left_sky = skyline_bbs(left)  # Running BBS on left
    right_sky = skyline_bbs(right)  # Running BBS on right

    merged = left_sky + right_sky  # Merging skylines
    return [p for p in merged if is_skyline_point(p, merged)]  # Filtering final skyline points

# --- STEP 9: RUNNING ALL THE REQUIRED METHODS AND RECORDING THEIR RESULTS ---

results = []

start = time.time()
# Running Sequential Scan and noting the time 
 
seq = skyline_sequential(dataset) 
results.append(("Sequential Scan", seq, time.time() - start))

start = time.time()
# Running BBS and noting the time

bbs = skyline_bbs(dataset)  
results.append(("BBS", bbs, time.time() - start))

start = time.time()
# Running Divide-and-Conquer BBS and timing it

bbs_dc = skyline_bbs_divide_and_conquer(dataset)  
results.append(("BBS with Divide-and-Conquer", bbs_dc, time.time() - start))

# --- STEP 10: WRITING OUTPUT TO FILE ---

# Writing execution time and skyline points for each method

with open("Task2_SkylineSearch_Output.txt", "w") as f:
    for method, skyline, t in results:
        f.write(f"=== {method} ===\n")
        f.write(f"Execution Time: {t:.4f} seconds\n")
        f.write("ID\tX (Cost)\tY (Size)\n")
        for p in skyline:
            f.write(f"{p[0]}\t{p[1]:.2f}\t{p[2]:.2f}\n")
        f.write("\n")

print("Skyline Search Completed.")

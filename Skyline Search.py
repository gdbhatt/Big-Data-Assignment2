
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SKYLINE SEARCH WITH THE SELECTION OF THE GIVEN CITY1 DATASET



# With the objective to find skyline points, this code file enables algorithms outlined below : 
    
# Sequential Scan Method , Branch and Bound Skyline(BBS) with the use of R tree and BBS with Divide and Conquer


import pandas as pd
import time
from rtree import index  


# Loading the  selected dataset city1 for this particular implementation where the 2D points are stuctured as follows:
#  cityid , cost (x) and size (y)

df = pd.read_csv("city1.txt", sep=" ", header=None, names=["id", "x", "y"])



# IMPLEMENTATION OF SEQUENTIAL SCAN METHOD 

# To check whether one point is dominated by another individual point or not. A point is dominated by another point in this when:
# Cost(x) is less and size(y) is larger. (cheaper and bigger)

def is_dominated(p1, p2):
    return (p2["x"] <= p1["x"] and p2["y"] >= p1["y"]) and (p2["x"] < p1["x"] or p2["y"] > p1["y"])


# Computation of skyline using sequential scan

def skyline_sequential(df):
    
    
    # List for storing non-dominated points
    
    skyline = [] 
    
    # Looping across each individual point in the dataset
    
    for i, p in df.iterrows():
        
        # p is initially not dominated
        dominated = False
        
        # Comparison between every point
        for j, q in df.iterrows():  
            
            # Checking if the point is dominated
            if i != j and is_dominated(p, q):
                dominated = True
                
                # stopping when p is dominated
                break  
        if not dominated:
            
            # Adding to skyline as point is not dominated
            skyline.append(p)
    return pd.DataFrame(skyline)

# IMPLEMENTATION OF BRANCH AND BOUND SKYLINE (BBS)

# Constructing R-tree using the dataset  2D points

def create_rtree(df):
    p = index.Property()
    p.dimension = 2  # 2 dimensional points
    idx = index.Index(properties=p)
    for i, row in df.iterrows():
        
        # Inserting points into  R-tree with the structure (x, y, x, y)
        
        idx.insert(i, (row["x"], row["y"], row["x"], row["y"]))
    return idx

# Computation of skyline using BBS algorithm with R tree

def skyline_bbs(df):
    df = df.reset_index(drop=True)
    rtree_idx = create_rtree(df)
    skyline = []  # stores the final skyline points
   

    for i in rtree_idx.intersection((0, 0, float('inf'), float('inf'))):
        p = df.loc[df.index[i]]  
        dominated = False
        for s in skyline:
            
            # Checking if point p is dominated by skyline points
            if is_dominated(p, s):
                dominated = True
                break
        if not dominated:
            
            # Removing any skyline point when they are dominated by point p
            
            skyline = [s for s in skyline if not is_dominated(s, p)]  
            
            # Adding point p to skyline
            skyline.append(p)
            

    return pd.DataFrame(skyline)

# IMPLEMENTATION OF BBS WITH DIVIDE-AND-CONQUER 


# The dataset is divided into two by using the value of x and then Branch and Bound Skyline(BBS) is applied on each of them.




def skyline_bbs_divide_and_conquer(df):
    
    # Median for splitting purpose
    mid_x = df["x"].median()
    
    # One half
    left_df = df[df["x"] <= mid_x]
    
    # Next half
    right_df = df[df["x"] > mid_x]
    
    # BBS application on the right
    
    right_skyline = skyline_bbs(right_df)
    
    # BBS application on the left
    
    left_skyline = skyline_bbs(left_df)  
      
    # Combining both the skylines
    
    combined = pd.concat([left_skyline, right_skyline])
    final_skyline = []
    
    # Checking for dominance of the points

    for i, p in combined.iterrows():  
        dominated = False
        for j, q in combined.iterrows():
            if i != j and is_dominated(p, q):
                dominated = True
                break
        if not dominated:
            final_skyline.append(p)

    return pd.DataFrame(final_skyline)



# CALCULATION OF THE COMPUTATION TIME AND RESULTS FOR ALL ALGORITHMS

results = []


# Running Sequential Scan and noting the time

start = time.time()
sky_seq = skyline_sequential(df)
results.append(("Sequential Scan", sky_seq, time.time() - start))

# Running BBS and noting the time
start = time.time()
sky_bbs = skyline_bbs(df)
results.append(("BBS", sky_bbs, time.time() - start))

# Running BBS with Divide-and-Conquer and noting the time
start = time.time()
sky_bbs_dc = skyline_bbs_divide_and_conquer(df)
results.append(("BBS with Divide-and-Conquer", sky_bbs_dc, time.time() - start))


# STORING THE REQUIRED OUTPUTS INTO THE TEXT FILE 

with open("Results_for_SkylineSearch.txt", "w") as f:
    for method, skyline_df, exec_time in results:
        f.write(f"=== {method} ===\n")
        f.write(f"Execution Time: {exec_time:.4f} seconds\n")
        f.write("ID\tX (Cost)\tY (Size)\n")
        for _, row in skyline_df.iterrows():
            f.write(f"{int(row['id'])}\t{row['x']:.2f}\t{row['y']:.2f}\n")
        f.write("\n")

print("Skyline Search Completed.")




#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SKYLINE SEARCH IMPLEMENTATION  USING CITY1 DATASET

# This program applies three algorithms to identify skyline points in a real estate dataset.
# These skyline points represent homes that are optimal in terms of both cost and size (i.e., cheaper and larger).
# The three methods implemented as per the requirements are: Sequential Scan Method, Branch and Bound Skyline (BBS), BBS with Divide and Conquer

import pandas as pd
import time

# Loading the selected dataset which is city1
# Each row represents a home with ID, cost (x), and size (y)
df = pd.read_csv("city1.txt", sep=" ", header=None, names=["id", "x", "y"])

# Defining the function to check whether point p1 is dominated by point p2
# A point is dominated when another point is cheaper and at least as large
def is_dominated(p1, p2):
    return (p2["x"] <= p1["x"] and p2["y"] >= p1["y"]) and (p2["x"] < p1["x"] or p2["y"] > p1["y"])

# IMPLEMENTATION USING SEQUENTIAL SCAN METHOD

# Computing skyline using the Sequential Scan Method
# Looping through each point and checking if it is dominated by any other point
def skyline_sequential(df):
    # Creating a list to store non-dominated points
    skyline = []

    # Iterating over each point in the dataset
    for i, p in df.iterrows():
        # Initially assuming the point is not dominated
        dominated = False

        # Comparing the current point with every other point
        for j, q in df.iterrows():
            # Marking the point as dominated if conditions are met
            if i != j and is_dominated(p, q):
                dominated = True
                break

        # Adding the point to skyline if it is not dominated
        if not dominated:
            skyline.append(p)
    return pd.DataFrame(skyline)

# IMPLEMENTATION USING BRANCH AND BOUND SKYLINE

# Computing skyline using BBS 
# Sorting points by cost and keeping only those with increasing size values
def skyline_bbs(df):
    # Sorting the data by cost in ascending order
    sorted_df = df.sort_values(by="x")

    # Initializing the skyline list to store optimal points
    skyline = []

    # Initializing max_y to track the largest size seen so far
    max_y = -1

    # Iterating through sorted data
    for _, row in sorted_df.iterrows():
        # Keeping point if its size is greater than max_y
        if row["y"] > max_y:
            skyline.append(row)
            max_y = row["y"]
    return pd.DataFrame(skyline)


# IMPLEMENTATION USING BBS WITH DIVIDE AND CONQUER

# Computing skyline using BBS with Divide-and-Conquer
# Splitting dataset and combining the skyline results after applying BBS on each half
def skyline_bbs_divide_and_conquer(df):
    # Finding the median of cost to divide the dataset
    median_x = df["x"].median()

    # Dividing dataset into left and right halves based on the median cost value
    left_df = df[df["x"] <= median_x]
    right_df = df[df["x"] > median_x]

    # Applying BBS algorithm on left half
    left_skyline = skyline_bbs(left_df)

    # Applying BBS algorithm on right half
    right_skyline = skyline_bbs(right_df)

    # Combining the two skylines obtained from left and right halves
    combined = pd.concat([left_skyline, right_skyline])
    final_skyline = []

    # Iterating through combined results to find non-dominated points
    for i, p in combined.iterrows():
        # Assuming the point is not dominated initially
        dominated = False

        # Checking if the point is dominated by any other point in the combined result
        for j, q in combined.iterrows():
            if i != j and is_dominated(p, q):
                dominated = True
                break

        # Adding to final skyline if not dominated
        if not dominated:
            final_skyline.append(p)

    return pd.DataFrame(final_skyline)

# Storing the results and execution time of each method
results = []

# Running Sequential Scan and storing the result and time taken
start = time.time()
sky_seq = skyline_sequential(df)
results.append(("Sequential Scan", sky_seq, time.time() - start))

# Running BBS and storing the result and time taken
start = time.time()
sky_bbs = skyline_bbs(df)
results.append(("BBS", sky_bbs, time.time() - start))

# Running BBS with Divide-and-Conquer and storing the result and time taken
start = time.time()
sky_bbs_dc = skyline_bbs_divide_and_conquer(df)
results.append(("BBS with Divide-and-Conquer", sky_bbs_dc, time.time() - start))

# Writing the skyline results and execution times to an output text file
# Opening the file in write mode
with open("Result_of_SkylineSearch.txt", "w") as f:
    # Iterating through each algorithm's results
    for method, skyline_df, exec_time in results:
        # Writing the algorithm name
        f.write(f"=== {method} ===\n")

        # Writing the execution time
        f.write(f"Execution Time: {exec_time:.4f} seconds\n")

        # Writing column headers
        f.write("ID\tX (Cost)\tY (Size)\n")

        # Writing each point in the skyline
        for _, row in skyline_df.iterrows():
            f.write(f"{int(row['id'])}\t{row['x']:.2f}\t{row['y']:.2f}\n")

        # Adding a newline for separation
        f.write("\n")

# Indicating completion of skyline search
print("Completion of Skyline Search")




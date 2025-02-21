import math
import os
import time
from tqdm import tqdm

# Ensure Graphviz is in the PATH
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

import pydot
from gvgen import GvGen
import numpy as np

# Constants used for visualization
POWER_TWOS = [-32768, -16384, -8192, -4096, -2048, -1024, -512, -256, -128, -64,
              -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64, 128, 256,
              512, 1024, 2048, 4096, 8192, 16384, 32768]
POWER_TWOS_ABS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
                  8192, 16384, 32768]


def show_graph(g):
    with open("my.dot", "w") as w:
        g.dot(w)
    (graph,) = pydot.graph_from_dot_file("my.dot")
    graph.del_node('"\\n"')
    graph.write_png('somefile.png')
    os.startfile('somefile.png')


def fixed_unique(array_or_arrays, additions):
    """
    Filter unique entries from a list of lists (or arrays) and return both the unique lists
    and their corresponding addition info. This function converts each inner list to a tuple,
    which allows for proper hashing and comparison.
    """
    seen = set()
    unique_arrays = []
    unique_additions = []
    for arr, add in zip(array_or_arrays, additions):
        arr_tuple = tuple(arr)
        if arr_tuple not in seen:
            seen.add(arr_tuple)
            unique_arrays.append(arr)
            unique_additions.append(add)
    return unique_arrays, unique_additions


def gen_K_0_l(l: int):
    result = [0]
    for i in range(l + 1):
        result.append(2 ** i)
        result.append(-(2 ** i))
    return sorted(result)


def gen_K_1_l(l: int):
    K_0 = gen_K_0_l(l)
    B_1_l = []
    B_1_l_additions = []
    for k_a in K_0:
        for k_b in K_0:
            new_ele = np.unique([1, k_a + k_b]).tolist()
            if new_ele not in B_1_l:
                B_1_l.append(new_ele)
                B_1_l_additions.append([[k_a, k_b]])
    K_1 = []
    K_1_additions = []
    for b_index, B in enumerate(B_1_l):
        new_K = [0]
        for n in range(l + 1):
            for b in B:
                new_value = b * (2 ** n)
                # Clamp the value between -2^l and 2^l
                new_value = max(min(2 ** l, new_value), -2 ** l)
                new_K.append(new_value)
                new_K.append(-new_value)
        K_1.append(np.unique(new_K).tolist())
        K_1_additions.append(B_1_l_additions[b_index])

    K_1, K_1_additions = fixed_unique(K_1, K_1_additions)
    return K_1, K_1_additions


def gen_K_2_l(l: int):
    K_0 = gen_K_0_l(l)
    K_1, K_1_l_additions = gen_K_1_l(l)
    B_2_l = []
    B_2_l_additions = []
    max_val = 2 ** l
    print("Iterating over half of", len(K_0))
    for k_a_index, k_a in tqdm(enumerate(K_0), total=len(K_0)):
        if k_a_index > len(K_0) / 2:
            break
        for K_1_addition_item, K_1_item in zip(K_1_l_additions, K_1):
            for k_b in K_1_item:
                new_ele = np.unique([1, k_a + k_b,
                                     K_1_addition_item[0][0] + K_1_addition_item[0][1]]).tolist()
                if new_ele not in B_2_l:
                    B_2_l.append(new_ele)
                    B_2_l_additions.append(K_1_addition_item + [[k_a, k_b]])
    print(f"GOT ALL FOR B sets for K_2_{l}")
    K_2 = []
    K_2_additions = []
    for b_index, B in tqdm(enumerate(B_2_l), total=len(B_2_l)):
        new_K = [0]
        for n in range(l + 1):
            for b in B:
                new_value = b * (2 ** n)
                new_value = max(min(2 ** l, new_value), -2 ** l)
                new_K.append(new_value)
                new_K.append(-new_value)
        K_2.append(np.unique(new_K).tolist())
        K_2_additions.append(B_2_l_additions[b_index])

    K_2, K_2_additions = fixed_unique(K_2, K_2_additions)
    return K_2, K_2_additions


def draw_coeff_add_set(coeffs, additions, l, k):
    """
    Debugging utility to visualize coefficient additions as a graph.
    """
    from gvgen import GvGen
    print("got coeffs", coeffs)
    print("with additions", additions)
    g = GvGen()
    coeff_node_map = {}
    # Create nodes for each coefficient
    for coeff in coeffs:
        coeff_node_map[coeff] = g.newItem(str(coeff))
    # Create start node
    coeff_node_map["start"] = g.newItem("Start")

    # Draw connections from start node to K_0 coefficients
    K_0 = gen_K_0_l(l)
    for k_0 in K_0:
        if k_0 not in coeffs:
            raise Exception("Not containing all K_0 coeffs!")
        g.newLink(coeff_node_map["start"], coeff_node_map[k_0])

    # Nodes that are not directly derived from the start node
    rest = [x for x in coeffs if x not in K_0]
    rest_done = []
    # Draw all additions
    for a, b in additions:
        if a + b in coeff_node_map:
            g.newLink(coeff_node_map[a], coeff_node_map[a + b])
            g.newLink(coeff_node_map[b], coeff_node_map[a + b])
            for r in rest:
                if r / (a + b) == r // (a + b):  # if divisible
                    divisor = r // (a + b)
                    if divisor in POWER_TWOS:  # if divisor is representable as a shift
                        if a + b != r:
                            link = g.newLink(coeff_node_map[a + b], coeff_node_map[r])
                            g.propertyAppend(link, "label", str(POWER_TWOS_ABS.index(abs(divisor)) - 1))
                        rest_done.append(r)

    # Mark nodes with missing connections in red
    missing_connections = [x for x in rest if x not in rest_done]
    for very_wrong in missing_connections:
        g.propertyAppend(coeff_node_map[very_wrong], "fontcolor", "red")
    show_graph(g)


def writeAddFile(coeffs, prefix, l):
    coeffs = sorted(coeffs, key=lambda x: len(x), reverse=True)
    result = ""
    for index, x in enumerate(coeffs):
        result += " |  " + " ".join([str(_) for _ in x])
        if index + 1 < len(coeffs):
            result += "\n"
    # Create the folder if it doesn't exist
    if not os.path.exists("coeffs"):
        os.makedirs("coeffs")
    filename = f"coeffs/{prefix}Add{l+1}.txt"
    with open(filename, "w") as f:
        f.write(result)


# Main execution
for l in tqdm(range(16), desc="cost-0"):
    cur = gen_K_0_l(l)
    print(f"K_0_{l} = Got {cur}")
    writeAddFile([cur], "10", l - 1)

print()

for l in tqdm(range(16)):  # range(16):
    cur, cur_adds = gen_K_1_l(l)
    # Uncomment the line below for debugging visualization:
    # for i in range(len(cur)):
    #     draw_coeff_add_set(cur[i], cur_adds[i], l, 1)
    print(f"K_1_{l} = Got {len(cur)} sets: {cur}")
    writeAddFile(cur, "11", l - 1)

print()

for l in tqdm(range(16), desc="cost-2"):  # 14,16
    start = time.time()
    cur, cur_adds = gen_K_2_l(l)
    # Uncomment the line below for debugging visualization:
    # for i in range(len(cur)):
    #     draw_coeff_add_set(cur[i], cur_adds[i], l, 2)
    print(f"K_2_{l} = Got {len(cur)} sets: ", "took: ", time.time() - start)
    writeAddFile(cur, "12", l - 1)

# Why these coeffs?
# The fully parallel architecture allows some tricks!
# Negative coeffs are for free. We can just subtract rather than add the result in the summation.
# Same with zero. Just have one less value for the summation. Easy!

import numpy as np
import scipy
import networkx as nx
from collections import defaultdict

#from gi.overrides.keysyms import Armenian_mijaket
from numpy.linalg import inv
#from collections import deque

#from torch.cuda.nvtx import range_end

def print_partitioned_matrix(matrix, partition, decimals=2):
    """
    Prints a matrix with entries formatted to `decimals` decimal places,
    and visual separators showing the blocks determined by `partition`.
    """
    nrows, ncols = matrix.shape
    row_lines = np.cumsum(partition)[:-1]  # Row/column lines to draw
    format_entry = lambda x: f"{x.real:.{decimals}f}" + (f"+{x.imag:.{decimals}f}j" if np.iscomplexobj(x) and abs(x.imag) > 1e-09 else "")

    # Determine column widths for clean formatting
    col_widths = []
    for j in range(ncols):
        max_width = max(len(format_entry(matrix[i, j])) for i in range(nrows))
        col_widths.append(max_width + 2)

    def row_separator():
        parts = []
        for i, w in enumerate(col_widths):
            sep = '-' * w
            parts.append(sep)
            if i + 1 in row_lines:
                parts.append('+')
        return ''.join(parts)

    for i in range(nrows):
        row_parts = []
        for j in range(ncols):
            entry = format_entry(matrix[i, j]).center(col_widths[j])
            row_parts.append(entry)
            if j + 1 in row_lines:
                row_parts.append('|')
        print(''.join(row_parts))
        if i + 1 in row_lines:
            print(row_separator())

    return

def print_complex_matrix(M, decimals=2, title=None):
    """
    Nicely prints a complex matrix with real and imaginary parts aligned.

    Parameters:
    - M : numpy.ndarray
        The complex matrix to print.
    - decimals : int
        Number of decimal places to round to.
    - title : str or None
        Optional title to print before the matrix.
    """
    if title:
        print(title)

    M = np.asarray(M)
    rows, cols = M.shape
    fmt = f"{{:>{3 + decimals}.{decimals}f}}"  # space for sign and decimal
    for i in range(rows):
        row_str = ""
        for j in range(cols):
            val = M[i, j]
            real_str = fmt.format(val.real)
            imag_str = fmt.format(val.imag)
            row_str += f"{real_str} + {imag_str}j  "
        print(row_str)
    print()  # Extra newline for spacing
    return

def u_induced_graph_partition(matrix_pairs, sizes):
    """
    Constructs a graph from {(A_l, B_l)} using non-zero blocks in A_l only,
    where edge exists between i and j if A_l[i,j] or A_l[j,i] is non-zero for any l.
    Parameters:
        matrix_pairs: List of tuples (A_l, B_l), each A_l and B_l are (n x n) matrices.
        sizes: List of partition sizes (n1, ..., nd)
    Returns:
        A dictionary with:
            - number of components (part_no)
            - components as lists
            - paths from representative to other members
    """
    n = sum(sizes)
    d = len(sizes)
    indices = np.cumsum([0] + sizes)

    # Step 1: Build the undirected graph
    G = nx.Graph()
    G.add_nodes_from(range(1, d + 1))

    for l, (A, _) in enumerate(matrix_pairs):
        for i in range(d):
            for j in range(i + 1, d):
                rows_i = slice(indices[i], indices[i + 1])
                cols_j = slice(indices[j], indices[j + 1])
                cols_i = slice(indices[i], indices[i + 1])
                rows_j = slice(indices[j], indices[j + 1])

                A_ij = A[rows_i, cols_j]
                A_ji = A[rows_j, cols_i]

                if not np.allclose(A_ij, 0,atol=1e-09) or not np.allclose(A_ji, 1e-09):
                    G.add_edge(i + 1, j + 1)

    # Step 2: Find connected components
    components = list(nx.connected_components(G))
    part_no = len(components)

    u_partition = {"part_no": part_no, 'clsses':[], "paths": {},"c(i)":{}}
    for idx, comp in enumerate(components, start=1):
        #print("clsses",u_partition)
        label = str(idx)
        nodes = sorted(list(comp))
        u_partition[label] = nodes
        # BFS for paths from first element
        root = nodes[0]
        u_partition['clsses'].append(root)
        u_partition['c(i)'][int(root)]=root
        paths = nx.single_source_shortest_path(G, root)
        for target in nodes[1:]:
            path = paths[target]
            step_list = []
            for u, v in zip(path, path[1:]):
                for l, (A, _) in enumerate(matrix_pairs):
                    u_idx = u - 1
                    v_idx = v - 1
                    rows_u = slice(indices[u_idx], indices[u_idx + 1])
                    cols_v = slice(indices[v_idx], indices[v_idx + 1])
                    cols_u = slice(indices[u_idx], indices[u_idx + 1])
                    rows_v = slice(indices[v_idx], indices[v_idx + 1])
                    A_uv = A[rows_u, cols_v]
                    A_vu = A[rows_v, cols_u]
                    if not np.allclose(A_uv, 0, atol=1e-09):
                        step_list.append(f"{l+1}_{u}_{v}")
                        break
                    elif not np.allclose(A_vu, 0, atol=1e-09):
                        step_list.append(f"{l+1}_{v}_{u}")
                        break
            u_partition["paths"][f"{label}_{target}"] = step_list
            u_partition["c(i)"][target]=int(label)

    return u_partition

def a_b_submatrices(cllctn,tpl_no,mat_part,i_strt,j_strt):
    #print("sub_matrix",tpl_no,i_strt,j_strt)

    A=cllctn[tpl_no-1][0]
    B=cllctn[tpl_no-1][1]

    #print("A")
    #print_partitioned_matrix(A,mat_part,2)
    #print("B")
    #print_partitioned_matrix(B,mat_part,2)

    if(i_strt==1):
        rw_strt=0
    else:
        rw_strt = sum(mat_part[:i_strt-1])
    if(j_strt==1):
        cl_strt=0
    else:
        cl_strt= sum(mat_part[:j_strt-1])

    rw_end=sum(mat_part[:i_strt])
    cl_end=sum(mat_part[:j_strt])

    #print("rw",rw_strt,rw_end)
    #print("cl",cl_strt,cl_end)
    A_ij=A[rw_strt:rw_end,cl_strt:cl_end]
    B_ij=B[rw_strt:rw_end,cl_strt:cl_end]

    #print_complex_matrix(A_ij,2,"A_ij")
    #print_complex_matrix(B_ij,2,"B_ij")

    #print_complex_matrix(A_ij@A_ij.conj().T,2,"A_ij_rU_check")
    #print_complex_matrix(B_ij@B_ij.conj().T,2, "B_ij_rU_check")

    return A_ij,B_ij

def u_solution(mat_part,graph_part,ci_i_prod):
    sz=len(mat_part)
    U_sol = [""] * sz
    #print("keys",list(ci_i_prod['A_prod'].keys()))
    for c in range(1,graph_part['part_no']+1):
        root=int(graph_part[str(c)][0])
        U_sol[root-1]=np.eye(mat_part[root-1])
        for i in graph_part[str(c)][1:]:
            pth=str(c)+"_"+str(i)
            A_ci_prod=ci_i_prod["A_prod"][pth]
            B_ci_prod=ci_i_prod["B_prod"][pth]
            U_sol[i-1]=np.linalg.inv(B_ci_prod)@A_ci_prod

    U_sol = scipy.linalg.block_diag(*U_sol)
    #print("U_sol")
    #print(U_sol.shape)
    #print(U_sol)
    #print_complex_matrix(U_sol.conj().T @ U_sol, 2, "U_sol_check")
    return U_sol


def ci_i_products(cllctn,mat_part,graph_part):

    #A_pth_prod=[""]*len(mat_part)
    #B_pth_prod=[""]*len(mat_part)

    A_part_prod = {}
    B_part_prod = {}
    for k in range(1,graph_part['part_no']+1):
        #print("partitions",k)
        clss_vrtx=graph_part[str(k)] # class
        c=clss_vrtx[0]  # class rep
        #print("class rep",c)
        mat_prt_sz=mat_part[c-1]
        sz=len(clss_vrtx)
        lst_vrtx=clss_vrtx[-1]
        #print("clss_vrtx",clss_vrtx)
        for i in range(1,sz):
            v_crnt=clss_vrtx[i]
            pth_lbl=str(c)+"_"+str(v_crnt)
            #print("pth_lbl",pth_lbl)
            A_ci_prod = np.eye(mat_part[v_crnt - 1])
            B_ci_prod = np.eye(mat_part[v_crnt - 1])
            if(pth_lbl in A_part_prod):
                A_ci_prod=A_part_prod[pth_lbl]
                B_ci_prod=B_part_prod[pth_lbl]
            else:
                pth_edge_seq = graph_part['paths'][pth_lbl]
                v_0 = c
                for edge in pth_edge_seq:
                    edge=edge.split("_")
                    edge=[int(x) for x in edge]
                    if(edge[1]==v_0):
                        v_1=edge[2]
                        cnj="no"
                    else:
                        v_1 = edge[1]
                        cnj="yes"
                    #print("c,l,i,j",c,edge[0],edge[1],edge[2])
                    pth = str(c) + "_" + str(v_1)
                    if(pth in A_part_prod):
                        A_ci_prod=A_part_prod[pth]
                        B_ci_prod=B_part_prod[pth]
                    else:
                        A_v_crnt, B_v_crnt = a_b_submatrices(cllctn, int(edge[0]), mat_part, edge[1], edge[2])
                        if(cnj=="no"):
                            A_ci_prod = A_ci_prod @ A_v_crnt
                            B_ci_prod = B_ci_prod @ B_v_crnt
                        else:
                            A_ci_prod = A_ci_prod @ np.linalg.inv(A_v_crnt)
                            B_ci_prod = B_ci_prod @ np.linalg.inv(B_v_crnt)

                        # product of class representative to vertex
                        A_part_prod[pth]=A_ci_prod
                        B_part_prod[pth]=B_ci_prod

                    v_0 = v_1


    ci_i_prod={};
    ci_i_prod["A_prod"]=A_part_prod;
    ci_i_prod["B_prod"]=B_part_prod

    return ci_i_prod


# print(result)
# Each A_l and B_l is 4x4, blocks are 2x2
'''A1 = np.block([[np.eye(2), np.ones((2,2))],
               [np.zeros((2,2)), np.eye(2)]])
B1 = np.copy(A1)

A4 = np.block([[np.eye(2), np.zeros((2,2))],
               [2*np.eye(2), np.eye(2)]])
B4 = np.copy(A4)

matrix_pairs = [(A1, B1), (A4, B4)]
path = ['1_1_2', '4_3_2', '1_3_4']  # assumes 4 blocks (1,2,3,4)'''


# # Example input for testing
# A1 = np.array([ [1, 0, 1, 0, 0],
#                 [0, 0, 0, 0, 0],
#                 [1, 0, 1, 8, 0],
#                 [0, 0, 0, 1, 0],
#                 [0, 0, 0, 0, 1]] )
#
# B1 = A1
# cllctn=[(A1,B1)]
# part = [1, 3, 1]
#
#
# #print("U_sol",U_sol)
# #print('rslt',U_sol)
# #print("eg",A1[0:1],B1[0:1])
# print(a_b_submatrices(cllctn,1,part,3,3))
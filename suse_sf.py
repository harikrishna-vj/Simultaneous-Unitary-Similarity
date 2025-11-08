import numpy as np
from  suse_grph import a_b_submatrices
from suse_grph import u_induced_graph_partition
from suse_grph import ci_i_products

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
# -12
def pre_solution_form(matrix_pairs, sizes, real=False, tol=1e-09):
    #print("In Solution form")
    n = sum(sizes)
    blocks = np.cumsum([0] + sizes)

    for l, (A, B) in enumerate(matrix_pairs, 1):
        #print("l",l)

        for i in range(len(sizes)):
            for j in range(len(sizes)):
                i_start, i_end = blocks[i], blocks[i + 1]
                j_start, j_end = blocks[j], blocks[j + 1]
                A_ij = A[i_start:i_end, j_start:j_end]
                B_ij = B[i_start:i_end, j_start:j_end]
                #print("ij",i,j)
                #print("A_ij",A_ij)
                #print("B_ij",B_ij)
                if i == j:
                    # Diagonal blocks: check if scalar * I
                    if not (
                        np.allclose(A_ij, np.eye(sizes[i]) * A_ij[0, 0], atol=tol)
                        and np.allclose(B_ij, np.eye(sizes[i]) * A_ij[0, 0], atol=tol)
                    ):
                        return ["no",l, i + 1, j + 1, (A_ij, B_ij)]
                elif sizes[i] != sizes[j]:
                    # Rectangular blocks should be zero
                    if not (
                        np.allclose(A_ij, 0, atol=tol)
                        and np.allclose(B_ij, 0, atol=tol)
                    ):
                        return ["no",l, i + 1, j + 1, (A_ij, B_ij)]
                else:
                    # Square blocks: A_ij A_ij* = r I = B_ij B_ij*
                    A_prod = A_ij @ (A_ij.T if real else A_ij.conj().T)
                    B_prod = B_ij @ (B_ij.T if real else B_ij.conj().T)
                    r_A = np.trace(A_prod) / sizes[i]
                    r_B = np.trace(B_prod) / sizes[i]
                    if not (
                        np.allclose(A_prod, np.eye(sizes[i]) * r_A, atol=tol)
                        and np.allclose(B_prod, np.eye(sizes[i]) * r_B, atol=tol)
                    ):
                        return ["no",l, i + 1, j + 1, (A_ij, B_ij)]
                    #else:
                        #print("solution_ij",l,i+1,j+1)
                    #    print_complex_matrix(A_prod,2,"A_prod")
                        #print(r_A)
                    #    print_complex_matrix(B_prod, 2,"B_prod")
                        #print(r_B)

    return ["yes"]

def mult_of_id_prA_prB(cllctn,mat_part,graph_part,ci_i_prod,tol=1e-09):
    blocks = np.cumsum([0] + mat_part)
    #print("graph",graph_part)
    for l, (A, B) in enumerate(cllctn, 1):
        #print("l", l)
        for i in range(len(mat_part)):
            for j in range(len(mat_part)):
                i_start, i_end = blocks[i], blocks[i+1]
                j_start, j_end = blocks[j], blocks[j+1]
                A_ij = A[i_start:i_end, j_start:j_end]
                B_ij = B[i_start:i_end, j_start:j_end]
                #print("grph_prt",graph_part['c(i)'])
                i_lbl = "no"; j_lbl="no"
                if(i!=j and mat_part[i]==mat_part[j]):
                    c_i=graph_part['c(i)'][i+1]

                    if(int(i+1)==int(c_i)):
                        i_lbl="yes"

                    c_j=graph_part['c(i)'][j+1]
                    if(int(j+1)==int(c_j)):
                        j_lbl="yes"

                    if(c_i==c_j):
                        if(i_lbl=="no"):
                            pth = str(c_i) + "_" + str(i + 1)
                            A_pth_prod_i = ci_i_prod['A_prod'][pth]
                            B_pth_prod_i = ci_i_prod['B_prod'][pth]
                        else:
                            A_pth_prod_i = np.eye(mat_part[i])
                            B_pth_prod_i = np.eye(mat_part[i])

                        if (j_lbl == "no"):
                            pth = str(c_i) + "_" + str(j + 1)
                            A_pth_prod_j = ci_i_prod['A_prod'][pth]
                            B_pth_prod_j = ci_i_prod['B_prod'][pth]
                        else:
                            A_pth_prod_j = np.eye(mat_part[j])
                            B_pth_prod_j = np.eye(mat_part[j])

                        pr_A= A_pth_prod_i @ A_ij @ np.linalg.inv(A_pth_prod_j)
                        pr_B= B_pth_prod_i @ B_ij @ np.linalg.inv(B_pth_prod_j)

                        if not (
                                np.allclose(pr_A, np.eye(mat_part[i]) * pr_A[0, 0], atol=tol)
                                and np.allclose(pr_B, np.eye(mat_part[i]) * pr_B[0, 0], atol=tol)
                        ):
                            #print("lij",l,i,j)
                            return ["no", l, i+1, j+1, c_i, c_j,  (pr_A, pr_B)]

    return ['yes']

#-12
def solution_form(cllctn, mat_part, tol=1e-09):
    pre_sol=pre_solution_form(cllctn,mat_part)
    if(pre_sol[0]=="no"):
        pre_sol.append('pre_sol')
        return pre_sol,{},{}
    else:
        graph_part=u_induced_graph_partition(cllctn,mat_part)
        ci_i_prod=ci_i_products(cllctn,mat_part,graph_part)
        sol_form=mult_of_id_prA_prB(cllctn,mat_part,graph_part,ci_i_prod)
        if(sol_form[0]=="no"):
            #print("form",sol_form)
            sol_form.append('sol')
        return sol_form,ci_i_prod,graph_part

    return



# import numpy as np
#
# # Partition sizes
# partition = [2, 2]  # so n = 4
#
# # Define matrices that satisfy the three conditions
# alpha1 = 3.0
# alpha2 = 5.0
# r = 2.0
#
# # Diagonal blocks: scalar * identity
# A1 = np.block([
#     [alpha1 * np.eye(2),              np.zeros((2, 2))],
#     [np.zeros((2, 2)),    alpha2 * np.eye(2)]
# ])
# B1 = np.block([
#     [alpha2 * np.eye(2),              np.zeros((2, 2))],
#     [np.zeros((2, 2)),    alpha2 * np.eye(2)]
# ])
#
# # Off-diagonal square blocks with unitary scaled
# U = np.array([[0, 1], [1, 0]]) / np.sqrt(2)
# A2 = np.block([
#     [alpha1 * np.eye(2),              np.sqrt(r) * U],
#     [np.sqrt(r) * U.T,    alpha2 * np.eye(2)]
# ])
# B2 = np.copy(A2)
#
# # Assemble the input list of matrix pairs
# matrix_pairs = [(A1, B1), (A2, B2)]
#
# # Call the function
# result = Check_Solution_Form(matrix_pairs, partition, real=False)
# print(result)

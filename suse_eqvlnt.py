import numpy as np
from scipy.linalg import block_diag, eigh
from collections import Counter
from numpy.linalg import eig

#def get_matrix_partition(A_eigs,B_eigs):
#
#    return

def is_multiple_of_identity(M, tol=1e-09):
    """Check if a Hermitian matrix is a scalar multiple of identity."""
    if not np.allclose(M, M.conj().T, atol=tol):
        return False
    diag_val = M[0, 0]
    return np.allclose(M, np.eye(M.shape[0]) * diag_val, atol=tol)


def equivalent_collection(typ,cllctn,part_sizes, l, i, j, A_N, B_N, U_eqv, V_eqv, is_real=False):
    """
    Return equivalent collection {(U A_l U*, V B_l V*)}, the diagonal matrix used,
    and distinct eigenvalues with multiplicity.

    A_list, B_list : list of np.arrays of shape (n, n)
    part_sizes : list of sizes [n_1, ..., n_d]
    l : int, matrix index in collection (1-based)
    i, j : int, block indices (1-based)
    is_real : bool, whether inputs are real
    """
    A_list, B_list = zip(*cllctn)
    A_list = list(A_list)
    B_list = list(B_list)
    thrm_2_part=""
    if(typ=="pre_sol"):
        # n = sum(part_sizes)
        # p = len(part_sizes)
        #
        # # Step 1: identify block slices
        # cum_sizes = [0] + list(np.cumsum(part_sizes))
        # slc_i = slice(cum_sizes[i - 1], cum_sizes[i])
        # slc_j = slice(cum_sizes[j - 1], cum_sizes[j])

        #A = A_list[l - 1][slc_i, slc_j]
        #B = B_list[l - 1][slc_i, slc_j]

        #print("Inside eqvlnt problem")
        A=A_N
        B=B_N
        # Step 2: Choose a Hermitian matrix to diagonalize
        if i == j:

            A_H1 = (A + A.conj().T) / 2
            A_H2 = (A - A.conj().T) / (2j)
            B_H1 = (B + B.conj().T) / 2
            B_H2 = (B - B.conj().T) / (2j)

            if not is_multiple_of_identity(A_H1):
                A_H = A_H1
            else:
                A_H = A_H2

            if not is_multiple_of_identity(B_H1):
                B_H = B_H1
            else:
                B_H = B_H2

            part_i=i

            thrm_2_part="1"

        else:
            # Off-diagonal: A*A^*, A^*A
            AAstar = A @ A.conj().T
            AstarA = A.conj().T @ A
            BBstar = B @ B.conj().T
            BstarB = B.conj().T @ B

            if not is_multiple_of_identity(AAstar):
                A_H = AAstar
                part_i=i
            else:
                A_H = AstarA
                part_i=j

            if not is_multiple_of_identity(BBstar):
                B_H = BBstar
                part_i = i
            else:
                B_H = BstarB
                part_i= j

            thrm_2_part = "2,3"

        eigvals_A, Y = eigh(A_H)
        eigvals_B, Z = eigh(B_H)

    if(typ=="sol"):
    # Step 3: Diagonalize A_H and B_H

        eigvals_A, Y = eig(A_N)
        sort_idx=eigvals_A.argsort()[::-1]
        eigvals_A=eigvals_A[sort_idx]
        Y=Y[:,sort_idx]
        eigvals_B, Z = eig(B_N)
        sort_idx = eigvals_B.argsort()[::-1]
        eigvals_B = eigvals_B[sort_idx]
        Z = Z[:, sort_idx]
        A_H=A_N ; B_H=B_N ;
        part_i=i
        thrm_2_part="4"

    #print("Y",Y)
    #print("Z",Z)

    #print("typ",typ)

    #print("eigvals_A",eigvals_A)
    #print("eigvals_B",eigvals_B)
    # Eigenvalue multiplicity
    def eig_mults(eigvals):
        vals, counts = np.unique(np.round(eigvals, decimals=9), return_counts=True)
        #return list(zip(vals, counts))
        return list(vals),list(counts)

    def eig_match(vals_A,mults_A,vals_B,mults_B):
        val_match=np.allclose(vals_A,vals_B,atol=1e-09)
        mult_match=np.allclose(mults_A,mults_B,atol=1e-09)
        return val_match,mult_match


    vals_A,mults_A = eig_mults(eigvals_A)
    vals_B,mults_B = eig_mults(eigvals_B)
    val_mtch,mult_mtch=eig_match(vals_A,mults_A,vals_B,mults_B)

    if(val_mtch and mult_mtch==True):

        matrix_part=part_sizes[:part_i-1] + mults_A + part_sizes[part_i:]

        #print("matrix_part",matrix_part)
        # Step 4: Construct U and V
        #print("i,j,part_i",i,j,part_i)
        #print("Y_sz",Y.shape)
        #print("Z_sz",Z.shape)

        U_blocks = [np.eye(nk) if k + 1 != part_i else Y for k, nk in enumerate(part_sizes)]
        V_blocks = [np.eye(nk) if k + 1 != part_i else Z for k, nk in enumerate(part_sizes)]

        #print("U_blocks")
        #print(U_blocks)

        U = block_diag(*U_blocks)
        V = block_diag(*V_blocks)

        # print("U for eqvlnt problem:")
        # print(U)
        # print("V for eqvlnt problem")
        # print(V)
        # #Step 5: Form equivalent collection
        U=U.conj().T; V=V.conj().T;

        #print("V_sz",V.shape)
        #print("U_sz",U.shape)

    #for A in A_list:
        #print("A_shape")
        #print (A.shape)

        new_A_list = [U @ A @ U.conj().T for A in A_list]
        new_B_list = [V @ B @ V.conj().T for B in B_list]

        cllctn=list(zip(new_A_list,new_B_list))

        U_eqv=U@U_eqv
        V_eqv=V@V_eqv

        return {
            "mtch":True,
            "cllctn":cllctn,
            "A_diag": A_H,
            "B_diag": B_H,
            "A_eigen": mults_A,
            "B_eigen": mults_B,
            "U_eqv": U_eqv,
            "V_eqv":V_eqv,
            "matrix_part":matrix_part,
            "thrm_part":thrm_2_part
         }
    else:
        return {"mtch":False,
                "vals_A":vals_A,
                "vals_B":vals_B,
                "mults_A": mults_A,
                "mults_B": mults_B,
                "thrm_part":thrm_2_part
                }


# # Example inputs
# A1 = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
# A1 = (A1 + A1.conj().T)/2  # make it Hermitian
# B1 = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
# B1 = (B1 + B1.conj().T)/2
#
# A_list = [A1]
# B_list = [B1]
# part_sizes = [2, 2]  # 2 blocks
#
# result = Equivalent_Collection(A_list, B_list, part_sizes, l=1, i=1, j=1, is_real=False)
#
# print("Eigenvalue multiplicities of A:", result["A_eigenvalues"])
# print("Eigenvalue multiplicities of B:", result["B_eigenvalues"])
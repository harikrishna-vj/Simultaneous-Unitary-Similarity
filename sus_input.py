import numpy as np
import scipy
from scipy.linalg import block_diag, eigh
from scipy import sparse

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

def get_diagonal_matrix(partition,values):
    """
    Given a partition [n1, n2, ..., nd], returns a diagonal matrix
    with non-negative entries corresponding to block identifiers 1, 2, ..., d.
    """
    diagonal_entries = []
    for i, count in enumerate(partition, start=1):
        diagonal_entries.extend([values[i-1]] * count)

    return np.diag(diagonal_entries)

def print_partitioned_matrix(matrix, partition, decimals=2):
    """
    Prints a matrix with entries formatted to `decimals` decimal places,
    and visual separators showing the blocks determined by `partition`.
    """
    nrows, ncols = matrix.shape
    row_lines = np.cumsum(partition)[:-1]  # Row/column lines to draw
    format_entry = lambda x: f"{x.real:.{decimals}f}" + (f"+{x.imag:.{decimals}f}j" if np.iscomplexobj(x) and abs(x.imag) > 1e-12 else "")

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


def get_zero(m,n,real=False):
    array_shape=(m,n)
    if(real==False):
        Z = np.zeros(array_shape, dtype=complex)
    else:
        Z = np.zeros(array_shape, dtype=real)
    return Z

def get_complex(m,n,real=False):
    # Generate random real and imaginary parts (e.g., floats between 0 and 1)
    R = np.random.rand(m, n)
    M = R
    if(real==False):
        I = np.random.rand(n, n)
        M=R + 1j * I
    return M

def get_hermitian(n,real=False):
    if (real == False):
        R_H=get_complex(n,n)
        I_H=get_complex(n,n)
        H = R_H + 1j * I_H
        H = (H + H.conj().T) / 2
    else:
        R=get_complex(n,n,True)
        R=(R + R.T)/2
        H=R
    return H

def get_unitary(n,real=False):
    H=get_hermitian(n,real)
    D_H, U_org = scipy.linalg.eigh(H)
    return U_org

def get_multiple_of_unitary(n,real=False):
    U=get_unitary(n,real)
    a=get_complex(1,1,True)
    a=a[0,0]
    if(a<0):
        a=-1*a
    U=a*U
    return U

def get_multiple_of_identity(n,real=False):
    I=np.eye(n)
    a = get_complex(1, 1,real)
    a=a[0,0]
    I=a*I
    return I



def get_input_7x7_sus():
    # setting up a 2 1 2 2 i.e 7x7 problem in solution form
    # first matrix diagonalizes to give this form
    # 2nd and 3rd matrix have Solution form in the U-induced partition

    U=get_unitary(7)

    D=get_diagonal_matrix([2,1,2,2],[0.1,0.101,0.3,0.4])
    #print("D",D)
    U_inp=get_unitary(7)
    A1=U_inp @ D @ U_inp.conj().T
    B1=U @ A1 @ U.conj().T

    #-------
    # 2nd set is put together using aI's and rI's to be in Solution form
    M11=get_multiple_of_identity(2); M12=get_zero(2,1) ; M13=get_multiple_of_unitary(2); M14=get_zero(2,2)
    M21=get_zero(1,2); M22=get_multiple_of_identity(1) ; M23=get_zero(1,2); M24=get_zero(1,2)
    M31=get_zero(2,2); M32=get_zero(2,1) ; M33=get_multiple_of_identity(2); M34=get_multiple_of_unitary(2)
    M41=get_zero(2,2); M42=get_zero(2,1) ; M43=get_zero(2,2); M44=get_multiple_of_identity(2)

    R1=np.hstack([M11,M12,M13,M14]);
    R2=np.hstack([M21,M22,M23,M24]);
    R3=np.hstack([M31,M32,M33,M34]);
    R4=np.hstack([M41,M42,M43,M44]);
    A2=np.vstack([R1,R2,R3,R4])
    #print("A2\n")
    #print_partitioned_matrix(A2,4)
    A2=U_inp@ A2 @U_inp.conj().T
    B2=U @ A2 @ U.conj().T

    #---------
    # similarly the 3rd collection
    M11 = get_multiple_of_identity(2);M12 = get_zero(2, 1);M13 = get_multiple_of_unitary(2);M14 = get_zero(2, 2)
    M21 = get_zero(1, 2);M22 = get_multiple_of_identity(1); M23 = get_zero(1, 2);M24 = get_zero(1, 2)
    M31 = get_zero(2, 2);M32 = get_zero(2,1);M33 = get_multiple_of_identity(2);M34 = get_multiple_of_unitary(2)
    M41 = get_zero(2, 2);M42 = get_zero(2,1);M43 = get_zero(2,2);M44 = get_multiple_of_identity(2)

    R1 = np.hstack([M11, M12, M13, M14]);
    R2 = np.hstack([M21, M22, M23, M24]);
    R3 = np.hstack([M31, M32, M33, M34]);
    R4 = np.hstack([M41, M42, M43, M44]);

    A3=np.vstack([R1,R2,R3,R4])
    #print("A3\n")
    #print_partitioned_matrix(A3,4)
    A3 = U_inp @ A3 @ U_inp.conj().T
    B3=U @ A3 @ U.conj().T

    cllctn=[(A1,B1),(A2,B2),(A3,B3)]
    #partition=[2,1,2,2]

    #eigvals_A,Y = eigh(A1)

    #print_complex_matrix(check_Y, 2, "check_Y")
    #check_D_U
    #check=Y.conj().T @ U_inp
    #print_complex_matrix(check,2,"V*U")

    return cllctn,U,7

def check_normal(A):
    if(np.allclose(A @ A.conj(),A.conj() @ A)):
        return True
    return False

def get_input_non_normal_sus():
    P = get_complex(5,5)
    D_c = get_diagonal_matrix([2,1,2],[2+1j,1+1j,0+1j])
    #print_complex_matrix(D_c,2,'D_c')
    A = P @ D_c @ np.linalg.inv(P)
    U = get_unitary(5)
    B = U @ A @ U.conj().T
    cllctn=[(A,B)]
    #if(check_normal(A)==False):
        #print("A not normal")
    return cllctn,U,5

def get_input_not_sus():
    U1 = get_unitary(5)
    U2 = get_unitary(5)

    A1=get_complex(5,5)
    B1=U1 @ A1 @ U1.conj().T
    A2=get_complex(5,5)
    B2=U2 @ A2 @ U2.conj().T

    cllctn = [(A1, B1), (A2, B2)]

    return cllctn,U1,5


# cllctn,U,n=get_input_non_normal_sus()
# A=cllctn[0][0]
# print_complex_matrix(A,2,'A')
# B=cllctn[0][1]
# print_complex_matrix(B,2,'B')




# cllctn,U,n=get_input_not_sus()
# A1=cllctn[0][0]
# print_complex_matrix(A1,2,'A1')
# B1=cllctn[0][1]
# print_complex_matrix(B1,2,'B1')
# A2=cllctn[1][0]
# print_complex_matrix(A2,2,'A2')
# B2=cllctn[1][1]
# print_complex_matrix(B2,2,'B2')
# print_complex_matrix(U,2,'U')
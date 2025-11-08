import numpy as np
#from caffe2.perfkernels.hp_emblookup_codegen import sizeof
from suse_sf import solution_form
from suse_grph import u_induced_graph_partition
from suse_eqvlnt import equivalent_collection
from suse_grph import a_b_submatrices
from suse_grph import u_solution
import scipy
from scipy.linalg import block_diag, eigh
from sus_input import print_partitioned_matrix
from sus_input import get_input_7x7_sus,get_input_non_normal_sus,get_input_not_sus
import io
import sys
from fpdf import FPDF
import textwrap


fpdf="yes"

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

def Construct_U_sol(U,products):
    U_sol=""
    return U_sol

def find_u_org(U_prod,V_prod,U_sol):
    U_org=V_prod.conj().T@U_sol@U_prod
    return U_org

def check_solution(cllctn,U_sol):
    sm=0
    result="SUS"
    #print_complex_matrix(U_sol@U_sol.conj().T,2,"U_sol_test")

    for tple in cllctn:
        sm=sm+1
        A=tple[0]
        B=tple[1]
        #print_complex_matrix(A,2,"A")
        #print_complex_matrix(B,2,"B")
        if np.allclose(U_sol @ A @ U_sol.conj().T,B , atol=1e-09):
            #print("B"+str(sm)  + " close to " "U_solA"+ str(sm) +"U_sol^*")
            #print_complex_matrix(U_sol @ A @ U_sol.conj().T - B, 4, "Error")
            result="SUS"
        else:
            #print("B" + str(sm) + " not close to " "U_solA" + str(sm) + "U_sol^*")
            #print_complex_matrix(U_sol @ A @ U_sol.conj().T-B,4,"Error")
            result="Not SUS"

    return result



# A1=np.array([[2,1,-1,0],[0,1,1,0],[1,0,3,1],[1,-1,1,1]],dtype=np.complex64)
#
# print_complex_matrix(U_org,2,"U_used")
#
# B1=U_org @ A1 @ U_org.conj().T
#
# print_complex_matrix(A1,2,"A1")
# print_complex_matrix(B1,2,"B1")
# print_complex_matrix(U_org,2,"U_org")

# Step 1: Create a string buffer to capture print statements
if(fpdf=="yes"):
    buffer = io.StringIO()
    # Redirect print statements to the buffer
    sys.stdout = buffer


# enter size of matrices n
# get input from one of the 'get_input' functions on sus_input.py

cllctn,U_usd,n=get_input_7x7_sus()
#cllctn,U_usd,n=get_input_non_normal_sus()
#cllctn,U_usd,n=get_input_not_sus()
matrix_part=[n]
org_cllctn=cllctn
U_eqv=np.eye(n)
V_eqv=np.eye(n)
p=len(cllctn)
print ("An 7x7 example showing the flow of the algorithm to solve the given S.U.S problem")
print ("It runs for 5 iterations showing that the collection is S.U.S")
print("Printing salient results of each iteration")
print("The given input collection of p=" + str(p) +", 2-tuples (A_l,B_l)")
print("Displaying 2 decimal places\n")
for l in range(0,len(cllctn)):
    print_complex_matrix(cllctn[l][0],2,"A_"+str(l+1))
    print(" ")
    print_complex_matrix(cllctn[l][1], 2, "B_" + str(l+1))

#print(sol_form[0])
#if(sol_form[0]=="no"):
#    print("tuple no",sol_form[1],"block_i",sol_form[2],"block_j",sol_form[3])
for i in range(0,n):
    print("Iteration:", " ",i+1)
    print("--------------")
    print("U-Induced Partition: ",matrix_part)
    sol_form,ci_i_prod,graph_part = solution_form(cllctn, matrix_part)
    if (sol_form[0]=="yes"):
        print("In Solution form:")
        graph_part = u_induced_graph_partition(cllctn, matrix_part)
        print("U-Induced Graph:")
        print("Note: path is seq of l_i_j triples (A_ij non-zero in l^th matrix)")
        wrp_grph_part = textwrap.fill(str(graph_part), width=120)
        print(wrp_grph_part)
        #print(graph_part)
        U_sol  = u_solution(matrix_part,graph_part,ci_i_prod)
        #print_complex_matrix(U_sol,2,"U_sol")
        result=check_solution(cllctn, U_sol)
        print(" ")
        if(result=="SUS"):
            print("Please Refer Definition 5 and Theorem 1")
            print_complex_matrix(U_sol , 2, "U_sol")
            print("U_sol checked \n ")
            print("The collection is S.U.S i.e Simultaneously Unitarily Similar ! \n")
            #print("Original problem errors")
            U_found = find_u_org(U_eqv, V_eqv, U_sol)
            result_sol=check_solution(org_cllctn,U_found)
            if(result_sol=="SUS"):
                print(" ")
                #print("Given collection is SUS")
                #print('U_org is a solution')
                print("U (U_found) that solves the original (given) problem")
                print_complex_matrix(U_found,2,"U_found")
                print('The U that was used to set-up the problem, algorithm found a different U ?')
                print_complex_matrix(U_usd,2,"U_usd")
                break;
        else:
            print("NOT S.U.S", "Reason: U^sol does not solve the problem (Theorem 1)")
            break;
    else:
        #print("Not in solution form")
        if(sol_form[-1]=="pre_sol"):
            typ="pre_sol"
            print("Not in Pre-Solution form")
            print("Please refer Definition 3")
            print("Reason: A^l_ii/B^l_ii , the l,i,j of A^l_ij which fail's the criterion as follows:")
            l = sol_form[1];i = sol_form[2];j = sol_form[3];
            A_N = sol_form[4][0]
            B_N = sol_form[4][1]
            eq_i=i; eq_j=j;
        if(sol_form[-1]=="sol"):
            typ="sol"
            print("In Pre-Solution form but Not in Solution form")
            print("Please refer Definition 3 and 5")
            print("U-Induced Graph:")
            print("Note: path is seq of l_i_j triples (A_ij non-zero in l^th matrix)")
            wrp_grph_part = textwrap.fill(str(graph_part), width=120)
            print(wrp_grph_part)
            print("Please refer Definition 5")
            print("Reason: l, i, j of Normal pr(A_ij),pr(B_ij) which is not multiple of Identity as follows")
            l = sol_form[1]; i = sol_form[2]; j = sol_form[3]; c_i=sol_form[4]; c_j=sol_form[5]
            A_N = sol_form[6][0]
            B_N = sol_form[6][1]
            eq_i=c_i;eq_j=c_j
        print("l = ",l,",i = ",i,",j = ",j)
        print("l^th partitioned matrix of collection from where sub-matrices (S,R) are picked for Diagonalization\n")
        print("A_"+str(l))
        print_partitioned_matrix(cllctn[l-1][0],matrix_part)
        print(" ")
        print("B_"+str(l))
        print_partitioned_matrix(cllctn[l-1][1],matrix_part)
        print(" ")
        print_complex_matrix(A_N,2,"S")
        print_complex_matrix(B_N,2,"R")
        print("Setting up an Equivalent Problem")
        eqvlnt_prob=equivalent_collection(typ,cllctn,matrix_part,l,eq_i,eq_j,A_N,B_N,U_eqv,V_eqv)
        if(eqvlnt_prob['mtch']==False):
            print('vals_A')
            print(eqvlnt_prob['vals_A'])
            print('vals_B')
            print(eqvlnt_prob['vals_B'])
            print('mults_A')
            print(eqvlnt_prob['mults_A'])
            print('mults_B')
            print(eqvlnt_prob['mults_B'])
            print('NOT S.U.S')
            print('Reason: The eigen-values of (S,R) used for setting up equivalent problem do not match')
            print ('Refer Theorem 2, Proof item number/s: ' + eqvlnt_prob['thrm_part'])
            break
        else:
            cllctn=eqvlnt_prob['cllctn']
            U_eqv=eqvlnt_prob['U_eqv']
            V_eqv=eqvlnt_prob['V_eqv']
            matrix_part=eqvlnt_prob['matrix_part']
            print("U `blocks further', New partition")
            print('Refer Theorem 2, Proof item number/s: ' + eqvlnt_prob['thrm_part'])
            print(matrix_part)
        print(" ")

        # Your code with print statements
        #print("This is a sentence.")
        #print("Here is a matrix:")
        #print_matrix([[1 + .2j, 0.1 + 0j, 3 + 2j], [0.2 - 0j, 1 - 0.3j, -1 - 1j], [1 + 3j, 1 + 0j, 1 + 4j]])

if(fpdf=="yes"):
    # Step 2: Reset redirect so print statements go back to console
    sys.stdout = sys.__stdout__

    # Get the content from the buffer
    content = buffer.getvalue()

    # Step 3: Write the captured content to a PDF using FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add the captured content to the PDF
    for line in content.splitlines():
        pdf.cell(200, 10, txt=line, ln=True)

    # Save the PDF file
    pdf.output("S.U.S_7x7_SUS_example_2.pdf")
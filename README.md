Licence : GNU Lesser General Public licensce

Code: Python 3.7

OS on which code was run: Ubuntu 19.04

Processor on which code was tested: 2.3Ghz x 4 pentium, 7Gib RAM

Description of the implementation:
----------------------------------

Folllowing are the details of the implementation of the work described in the paper
‘Polynomial Algorithms for Simultaneous Unitary Similarity and Simultaneous Unitary Equivalance’
The .pdf of this paper available in this repository is titled sue&sueq_sol.pdf

1) The main file to run is suse.py. 
Currently there are three `example input collections’ which can be used by calling
one of the following functions:

get_input_7x7_sus()
get_input_non_normal_sus()
get_input_non_sus()

These functions are in suse_input.py, which has other functions to help construct
a collection with intended properties.

Select fpdf==`yes’ in suse.py if the output is to be written to a pdf `no’ otherwise

There are three pdf files S.U.S_7x7_SUS_example.pdf, S.U.S_5x5_non_normal_example.pdf and 
5x5_non_sus_example.pdf which are the outputs of running suse.py with the three different 
input collections respectively.

2) The other files suse_graph.py, suse_sf.py, suse_eqvlnt.py implement the U-Induced Partition,
checking for Solution form and setting up of Equivalent problem  respectively.

3)The notation followed in the code is similar to that in the paper.


Improvements to be made to this implementation :
--------------------------------------------------

1) Addition of the implementation for the S.U.Eq problem that involves partitioning a bi-graph
as described in Section 2.5 of the paper

2) Collecting the ‘Canonical features’ as described in Section 4.3 of the paper.

3) Improving presentation of the input and output i.e results of running the algorithm.



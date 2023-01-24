################################################################################
# Created on Fri Aug 24 13:36:53 2018                                          #
#                                                                              #
# @author: olhartin@asu.edu; updates by sdm                                    #
#                                                                              #
# Program to solve resister network with voltage and/or current sources        #
################################################################################

import numpy as np                     # needed for arrays
from numpy.linalg import solve         # needed for matrices
from read_netlist import read_netlist  # supplied function to read the netlist
import comp_constants as COMP          # needed for the common constants

# this is the list structure that we'll use to hold components:
# [ Type, Name, i, j, Value ]

################################################################################
# How large a matrix is needed for netlist? This could have been calculated    #
# at the same time as the netlist was read in but we'll do it here             #
# Input:                                                                       #
#   netlist: list of component lists                                           #
# Outputs:                                                                     #
#   node_cnt: number of nodes in the netlist                                   #
#   volt_cnt: number of voltage sources in the netlist                         #
################################################################################

def get_dimensions(netlist):           # pass in the netlist
    
    volt_cnt = 0
    node_cnt = []
    
    for comp in netlist:
        node_cnt = np.append(node_cnt, comp[COMP.I])
        if ( comp[COMP.TYPE] == COMP.VS ):
            volt_cnt += 1
            
    node_cnt_max = int(max(node_cnt))    

#    print(' Nodes ', node_cnt, ' Voltage sources ', volt_cnt)
    return (node_cnt_max,volt_cnt)

################################################################################
# Function to stamp the components into the netlist                            #
# Input:                                                                       #
#   y_add:    the admittance matrix                                            #
#   netlist:  list of component lists                                          #
#   currents: the matrix of currents                                           #
#   node_cnt: the number of nodes in the netlist                               #
# Outputs:                                                                     #
#   node_cnt: the number of rows in the admittance matrix                      #
################################################################################

def stamper(y_add,netlist,currents,node_cnt):
    # return the total number of rows in the matrix for
    # error checking purposes
    # add 1 for each voltage source...

    for comp in netlist:                  # for each component...
        #print(' comp ', comp)            # which one are we handling...

        # extract the i,j and fill in the matrix...
        # subtract 1 since node 0 is GND and it isn't included in the matrix
        i = comp[COMP.I] - 1
        j = comp[COMP.J] - 1

        if ( comp[COMP.TYPE] == COMP.R ):           # a resistor
            if (i >= 0):                            # add on the diagonal
                y_add[i,i] += 1.0/comp[COMP.VAL]
            if (j >= 0):
                y_add[j,j] += 1.0/comp[COMP.VAL]
            if (i >= 0) & (j >= 0):
                y_add[i,j] += (-1.0)/comp[COMP.VAL]
                y_add[j,i] += (-1.0)/comp[COMP.VAL]

        if ( comp[COMP.TYPE] == COMP.VS ):           # a voltage source
            if (i >= 0):
                y_add[node_cnt , i] += 1
                y_add[i, node_cnt ] += 1
            if (j >= 0):
                y_add[node_cnt , j] +=(-1)
                y_add[j, node_cnt ] +=(-1)
            currents[node_cnt , 0] += comp[COMP.VAL]
            node_cnt += 1
                
        if ( comp[COMP.TYPE] == COMP.IS ):           # a current source
            if (i >= 0):
                currents[i, 0] += (-1*comp[COMP.VAL])
            if (j >= 0):
                currents[j, 0] += comp[COMP.VAL]

    return node_cnt  # should be same as number of rows!

################################################################################
# Start the main program now...                                                #
################################################################################

# Read the netlist!
netlist = read_netlist()

# Print the netlist so we can verify we've read it correctly
for index in range(len(netlist)):
    print(netlist[index])
print("\n")

dimensions = get_dimensions(netlist)

node_cnt, volt_cnt = [dimensions[i] for i in (0,1)]

n = node_cnt + volt_cnt

y_add = np.zeros((n, n))

currents = np.zeros((n, 1))

stamped = stamper(y_add, netlist, currents, node_cnt)

solution = solve(y_add, currents)


print(solution)
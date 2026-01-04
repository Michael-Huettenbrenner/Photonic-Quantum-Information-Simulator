# -*- coding: utf-8 -*-
"""
This file provides a brief introduction to how to use this package
"""

#place your file in the same folder that the folder simlulator is placed in, and you can do the import
# this demonstration will not run unless you ALSO place it in the same folder as the simulator folder 
from simulator import *
import numpy as np



#%% Define a circuit
#As example, we will look at a BSM based teleportation circuit with an entanglement swapping stage prior to the teleportation
m = 10  #specifiy the number of modes
pauli_x = np.array([[0,1],   #custom quantum gates can be specified by supplying the single photon transfer matrix
                    [1,0]])


#you can use matrices like H (Hadamard) already without specifying them. You can find these in gates.py
#examlpe: make fusion type 2 gates

Ga1 = Gate(H, [4,5], m) #the Gate constructor takes your single photon matrix, the indices of the modes between which it acts (can also be > 2), and the total number of modes in the circuit m
Ga2 = Gate(H, [6,7], m) #it generates a gate class with a unitary that has the same dimension as the circuit (mxm) and pads the original matrix with 1s (diagonal) and 0s (off-diagonal) to achieve this
Ga3 = Gate(H, [4,5], m)
Ga4 = Gate(H, [6,7], m)

SWAPa = Gate(pauli_x, [5,7], m)


#make fusion type 2 gate nr. 2
Gb1 = Gate(H, [0,1], m)
Gb2 = Gate(H, [2,3], m)
Gb3 = Gate(H, [0,1], m)
Gb4 = Gate(H, [2,3], m)

SWAPb = Gate(pauli_x, [1,3], m)

GP3 = Gate(P(90, 1), [2,3], m)
GP9 = Gate(P(270, 1), [8,9], m)

#once you have specified your gates, order them into a circuit. Make sure to provide them in the order the photons see them in. (left side of list is first gate a photon would interact with)
BSM_Purifier = Circuit([Ga1,Ga2, SWAPa, Ga3,Ga4, GP3, Gb1,Gb2, SWAPb, Gb3,Gb4, GP9], m) #this is the constructor for a Circuit class object, it takes a list of gate class objects and the number of modes


#%%states
#you can specify the states you want to send into the circuit via dictionaries:
bell1  = {(2,4): 1/np.sqrt(2), (3,5): 1/np.sqrt(2)} #make sure to use idx notation: (2,4) means photon in mode 2 and photon in mode 4.
bell2  = {(6,8): 1/np.sqrt(2), (7,9): 1/np.sqrt(2)} #this is a superposition with coefficients 1/sqrt(2) -> the n-tupel() is the key in the dict, and the coefficient in the superposition is the dict's value corresponding to that key
incoming_state =   {(0,): np.sqrt(1/3), (1,): -np.sqrt(2/3) }

s1_fail = tensorproduct(bell1, bell1) #join states via tensorproduct
source = tensorproduct(bell1, bell2)

only_resource_failed = tensorproduct(s1_fail, incoming_state)
relevant_failure = s1_fail
correct = tensorproduct(incoming_state,source)
no_input = source

#%%Investigate input being mixture 

#define coefficients for mixed state
p_fail = 0.5
p_loss = 0.5

#join states into a mixed state by using a list of touples [(state_dict, P_in_mix),...]
mix_with_bad_terms = [(correct, (1-p_fail)*(1-p_loss)),     
                      (only_resource_failed, p_fail*(1-p_loss)), 
                      (no_input, (1-p_fail)*(p_loss)),      
                      (relevant_failure, p_fail*p_loss)]         




out_mixture, P_succ = transform_mixture_list( mix_with_bad_terms, BSM_Purifier, clicks=[0,2,4,6], no_clicks=[1,3,5,7], remove_bunching = [0,2,4,6]) #calculate the output mixture
#provide: mixed state, circuit, specify detector clicks / no clicks, specify with remove_bunching if detectors can exclude events that would make 2 clicks in one mode (photon number resolution)
#obtain: output mixed state, and the probability for the click pattern given the mixed state. Be careful here with normalization. If you use a pure state here, you can get the true success probability of a protocol for the expected input from P_succ
#it will print 'WARNING: You might have lost a state' everytime a part of the mixture cannot give the heralding pattern and would be removed. The output mixture can and will contain less terms.

#%%If you do not wish to transform a mixed state, but only have one state, you can also do

out_state = transform_state(correct, BSM_Purifier, clicks=[0,2,4,6], no_clicks=[1,3,5,7], remove_bunching = [0,2,4,6])  #this just gives you a dict corresponding to the output state for your provided input state


#%%We can do a lot more:
    
#if you want more control, there are ways to work with fock states (and convert to idx and vice versa)
# also, you can specify a distinguishability matrix S and calculate the transition PROBABILITIES (not amplitudes!) via a generalized tensor permanent: exmaple



#%%specify photon and mode number
num_modes = 8
num_photons = 4

#%%Generate the gates and circuit
Gate_1 = Gate(H, np.array([0,1]), num_modes)# the array in the function's argument tells us which two modes are involved in the gate
Gate_2 = Gate(H, np.array([1,4]), num_modes)
Gate_3 = Gate(NS, np.array([1,2,3]), num_modes)
Gate_4 = Gate(NS, np.array([4,5,6]), num_modes)
Gate_5 = Gate(H, np.array([1,4]), num_modes)
Gate_6 = Gate(H, np.array([0,1]), num_modes)

Gates_in_order = [Gate_1, Gate_2, Gate_3, Gate_4, Gate_5, Gate_6] #leftmost gate in circuit is also on the left in this list

CNOT = Circuit(Gates_in_order, num_modes)
#%%Make a list (np.array) of all Input and output states you want to consider. Formatting is up to you, index-basis is recommended. 
#The index basis tells us which photon is in which mode, i.e. [2,1,0,0] means four photons in total, first photon in mode 2, second photon in mode 1 and last two photons in mode 0
#note that for example in the two photon case [0,1] and [1,0] correspond to the same fock state ([1,1] for two modes) because the photons are indistinguishable. You cannot say which photon is in which mode
#build masks to exclude states that would not lead to a heralding click / are undesired inputs.

basis = Basis(num_photons, num_modes)

Input_click_mask_fock   = np.array([0,0,1,0,0,1,0,0]) #A one here means that these modes need to have a click. Since the detectors cannot distinguish multiple photons from one, multiple photon states are also allowed
Input_noclick_mask_fock = np.array([0,0,0,1,0,0,1,0]) #A one here means that these modes need to have no click

Output_click_mask_fock   = np.array([0,0,1,0,0,1,0,0]) #A one here means that these modes need to have a click. Since the detectors cannot distinguish multiple photons from one, multiple photon states are also allowed
Output_noclick_mask_fock = np.array([0,0,0,1,0,0,1,0]) #A one here means that these modes need to have no click

#%% CNOT behavior inputs and outputs
#Extract valid rows
Inputs_fock = remove_collision_states(apply_click_mask(basis.fock, Input_click_mask_fock, Input_noclick_mask_fock))[1:5:1] #remove the inputs where there are multiple photons, and where both photons are in control or target qubit
Inputs = fock_list_to_idx_list(Inputs_fock)

Outputs_fock = remove_collision_states(apply_click_mask(basis.fock, Output_click_mask_fock, Output_noclick_mask_fock))[1:5:1]
Outputs = fock_list_to_idx_list(Outputs_fock)

display_inputs = Inputs_fock[:,[0,1,4,7]]
display_outputs = Outputs_fock[:,[0,1,4,7]]

#%%Results CNOT
Amplitudes_CNOT = Phi(Inputs, Outputs, CNOT.unitary).get_amplitudes() # if you want to work with fock states, you need to add a fock=True as last entry in the Phi class' constructor here

Ampl_rounded_arr_CNOT = np.round(Amplitudes_CNOT.real, 3) + 1j * np.round(Amplitudes_CNOT.imag, 3)


#%% do the same but for distinguishable photons:
#the question is: does this account for 1 going to 2 and 2 going to one, or just 1 to 1 and 2 to 2 (for [1,2]->[1,2])
    
S=np.eye(num_modes)
scale = 0
for i in range(num_modes):
    for j in range(num_modes):
        if(i!=j):
            S[i,j] = scale
Dist_Probs_CNOT = round_array(Dist_Prob_Phi(Inputs, Outputs, CNOT.unitary, S).get_probabilities(),3)




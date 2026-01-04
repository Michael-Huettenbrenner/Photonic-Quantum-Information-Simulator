# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 11:35:12 2025

@author: Hüttenbrenner
"""
import math
import numpy as np

#%% helper functions

#build all possible n-1 tuples from n tuple by removing one element
def drop_one(tup):
    return [tup[:i] + tup[i+1:] for i in range(len(tup))]


def tuple_in_tuple(needle, haystack):
    # Convert to lists so we can slice-check
    n, h = len(needle), len(haystack)
    for i in range(h - n + 1):
        if haystack[i:i+n] == needle:
            return True
    return False

# Subsequence (allows gaps, order preserved)
def tuple_in_tuple_subsequence(needle, haystack):
    i = j = 0
    while i < len(needle) and j < len(haystack):
        if needle[i] == haystack[j]:
            i += 1
        j += 1
    return i == len(needle)



def all_close(lst, tol=1e-5):
    if not lst:  # empty list
        return True
    first = lst[0]
    return all(math.isclose(x, first, abs_tol=tol) for x in lst)

#%% functions for finding usalble states

#find a unique pattern
def find_unique_pattern(full_state, missing_photon):
    '''
    

    Parameters
    ----------
    full_state : Dict
        Dictionary describing the output state you get when doing the regular input with all photons
    missing_photon : List
        All outputs that could arise if you input a state with any given photon missing (n-1 photon states)
        This list contains the sets of possible outputs belonging to the respective input where one of the photons was missing
        The List entries are dictionaries, with key being the idx sate and value the amplitude

    Returns
    -------
    unique_patterns : Dict[tuple, List[tuple]]
        For each full-state tuple that has at least one “unreachable” sub-pattern,
        maps it to the list of all such unique (n‑1)-tuples.
    '''
    
    unique_patterns = {}
    
    for full_key in full_state:
        # build all n-1 sub-tuples
        subs = drop_one(full_key)
        
        # keep only those sub-tuples that are in *none* of the missing_photon dicts
        uniques = [
            sub
            for sub in subs
            if all(sub not in d for d in missing_photon)
        ]
        
        if uniques:
            unique_patterns[full_key] = uniques
            for u in uniques:
                print(f"found unique sub-pattern {u} from {full_key}")
    
    return unique_patterns


#identify if there is a single pattern in the full state that can be made unique by interference

# def unique_through_interference_in_lossy_states(full_state, missing_photon):
#     '''
    

#     Parameters
#     ----------
#     full_state : Dict
#         Dictionary describing the output state you get when doing the regular input with all photons
#     missing_photon : List
#         All outputs that could arise if you input a state with any given photon missing (n-1 photon states)
#         This list contains the sets of possible outputs belonging to the respective input where one of the photons was missing
#         The List entries are dictionaries, with key being the idx sate and value the amplitude

#     Returns
#     -------
#     unique_patterns : Dict[tuple, List(tuple)]
#         For each full-state tuple that has at least one pattern that can be made unique,
#         maps it to the last (n-1) tuple it found that can be interfered away in all lossy states such that the pattern becomes unique.
#     '''
    
#     unique_patterns = {}
    
#     for full_key in full_state:
#         # build all n-1 sub-tuples
#         subs = drop_one(full_key)
        
#         for sub in subs:
#             subs2 = drop_one(sub)
#             for sub2 in subs2:
#                 container = []
#                 count = 0
#                 ratios = []
#                 for state in missing_photon:
#                     Possible = {}
#                     for basis_vector in state:
#                         if tuple_in_tuple_subsequence(sub2, basis_vector):
#                             Possible[basis_vector] = state[basis_vector]
#                     if len(Possible) == 0:
#                         count += 1
#                         container.append(Possible)
#                     if len(Possible) == 2:
#                         count += 1
#                         container.append(Possible)
#                         values = list(Possible.values())
#                         first_value = values[0]
#                         second_value = values[1]
#                         ratios.append(np.abs(first_value / second_value))
                        
#                 if count == len(missing_photon) and all_close(ratios): #only add if the ratios are the same so we can be sure interference is possible
#                     unique_patterns.setdefault(full_key, []).append(sub2) #appends sub to the list at full_key, or makes a new list with only sub as entry if there was no prior list
                    
                            
#     for key in unique_patterns:
#         unique_patterns[key] = list(dict.fromkeys(unique_patterns[key]))             
    
#     return unique_patterns


def unique_through_interference_in_lossy_states(full_state, missing_photon):
    
    #problems with this
    #it can include states where a certain ratio does not exist for all lossy states. but interference will most likely reintroduce that state into the lossy state it was missing
    #if some of the ratios are non and some arent, it might pick non in the end
    #we might only be interested in states where two photon interference is possible to remove the pattern from all but the full state. This might not be possible for some of those it found
    #   -> if you recall we cared about 3 photons across the same two modes for example.
    #if the pattern it found is included in multiple no-loss basis states, than it only works if the interference does not kill them aswell
    
    #crucially: if we care about states that have 3 photons across the same 2 modes it might not find the pattern: i.e. 2,2,3 and 3,2,2 -> do that next
    
    
    unique_patterns = {}  # full_key -> {sub2: ratio}

    for full_key in full_state:
        subs = drop_one(full_key)
        for sub in subs:
            subs2 = drop_one(sub)
            for sub2 in subs2:
                count = 0
                ratios = []
                for state in missing_photon:
                    Possible = {}
                    for basis_vector in state:
                        if tuple_in_tuple_subsequence(sub2, basis_vector):  # or your chosen test
                            Possible[basis_vector] = state[basis_vector]

                    if len(Possible) == 0:
                        count += 1
                    elif len(Possible) == 2:
                        count += 1
                        vals = list(Possible.values())
                        ratios.append(np.abs(vals[0] / vals[1]))
                    # else: 1 or >2 -> this state doesn’t count toward `count`

                # Accept if every lossy state had 0 or 2 matches, and ratios agree
                if count == len(missing_photon) and all_close(ratios):
                    # pick a representative ratio; if there were no 2-match states, store None
                    ratio = ratios[0] if ratios else None
                    bucket = unique_patterns.setdefault(full_key, {})
                    # keep the first ratio seen for this sub2; remove `.setdefault` if you want to overwrite
                    bucket.setdefault(sub2, ratio)

    # Convert inner dicts to lists of (subtuple, ratio) pairs
    return {k: list(v.items()) for k, v in unique_patterns.items()}


#identify if unique pattern is possible through interference



#%%test cases

# full_state = { #as idx state!!
         
#     (0,1,2,4,5) : 1/np.sqrt(3), 
#     (0,2,2,4,5) : 1/np.sqrt(3),
#     (0,0,2,4,5) : 1/np.sqrt(3)
    
# }

# l1 = { #as idx state!!
         
#     (0,1,2,4) : 1/np.sqrt(4), 
#     (0,2,2,4) : 1/np.sqrt(4),
#     (0,0,4,5) : 1/np.sqrt(4),
#     (1,1,2,5) : 1/np.sqrt(4)
    
# }


# l2 = { #as idx state!!
         
#     (0,1,2,4) : 1/np.sqrt(2), 
#     (0,2,2,4) : 1/np.sqrt(2),
    
# }

# l3 = { #as idx state!!
         
#     (0,1,2,4) : 1/np.sqrt(3), 
#     (0,2,2,4) : 1/np.sqrt(3),
#     (0,2,2,2) : 1/np.sqrt(3)
    
# }
# missing_photon = [l1,l2,l3]

# A = unique_through_interference_in_lossy_states(full_state, missing_photon)
# print(A)





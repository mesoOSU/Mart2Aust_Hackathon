# -*- coding: utf-8 -*-
"""
Created on Fri May 13 12:17:03 2022

@author: xucha
"""

import numpy as np
import statistics as stats 

import orix
from orix.io import load
from orix.quaternion import symmetry
from orix.vector import Vector3d


from os import path

from matplotlib import pyplot as plt

# fname = "AF_001.hdf5"
# xmap= load(fname)


#############
## Classes ##
#############
# Notes:
# These classes have hard coded values in atm
# These will eventually need to be moved to a dictionary
# Transfer of MTEX to orix
# Testing purposes only 

# Steel
class material_sys():
    def __init__(mat, info, physcial, sym, pg):
        mat.info = 'material'
        mat.physical = 'Steel'
        mat.sym = 'specimen_symmetry'        
        mat.pg = 'triclnic'

# Crystal Symmetry High Temperature (CSHT)
class CS_HT():
    def __init__(cs_ht, mineral, pg_symmetry, rgb, axes):
        # High Temp parameters
        A = [3.65, 3.65, 3.65] 
        
        # High temperature phase
        # Accepts only Austenite as the correct phase atm
        # This value is hard coded for testing
        cs_ht.mineral = 'Austenite'
        
        # Currently hard coded to only take in m-3m
        cs_ht.pg_symmetry = symmetry.get_point_group(220)
        
        # color of the phase
        # takes an input as a string
        # X11/CSS4 color name with no space
        # e.g 'aquamarine' | 'seamediumgreen'
        # Use 'LightGreen' for testing
        cs_ht.color = plt.plot(color = rgb)
        
        # Axes definition of the material
        cs_ht.axes = Vector3d(np.array([[A[0], 0, 0], [0, A[1], 0], 
                                        [0, 0, A[2]]]))
        
# Crystal Symmetry Low Temperature (CSLT)       
class CS_LT():
    def __init__(cs_lt, mineral, pg_symmetry, rgb, axes):
        # Low Temp parameters
        M = [2.87, 2.87, 2.87]
        
        # High temperature phase
        # Accepts only Martensite as the correct phase
        # This value is hard coded for testing
        cs_lt.mineral = 'Martensite'
        
        # Currently hard coded to only take in m-3m
        cs_lt.pg_symmetry = symmetry.get_point_group(220)
        
        # color of the phase
        # takes an input as a string
        # X11/CSS4 color name with no space
        # e.g 'aquamarine' | 'seamediumgreen'
        # Use 'DarkRed' for testing
        cs_lt.color = plt.plot(color = rgb)
        
        # Axes definition of the material
        cs_lt.axes = Vector3d(np.array([[M[0], 0, 0], [0, M[1], 0], 
                                        [0, 0, M[2]]]))  
        
# Crystal Symmetry Reconstructed (CSR)      
class CS_R():
    def __init__(cs_r, mineral, pg_symmetry, rgb, axes):
        # High Temp parameters
        A = [3.65, 3.65, 3.65] 
        
        # High temperature phase
        # Accepts only Reconstructed Austenite as the correct phase
        # This value is hard coded for testing
        cs_r.mineral = 'Reconstructed Austenite'
        
        # Currently hard coded to only take in m-3m
        cs_r.pg_symmetry = symmetry.get_point_group(220)
        
        # color of the phase
        # takes an input as a string
        # X11/CSS4 color name with no space
        # e.g 'aquamarine' | 'seamediumgreen'
        # Use 'DarkBlue' for testing
        cs_r.color = plt.plot(color = rgb)
        
        # Axes definition of the material
        cs_r.axes = Vector3d(np.array([[A[0], 0, 0], [0, A[1], 0], 
                                        [0, 0, A[2]]]))
        
# Crystal Symmetry Variant (CSV)       
class CS_V():
    def __init__(cs_v, mineral, pg_symmetry, rgb, axes):
        # Low Temp parameters
        M = [2.87, 2.87, 2.87] 
        
        # High temperature phase
        # Accepts only Reconstructed Variants as the correct phase 
        # This value is hard coded for testing
        cs_v.mineral = 'Reconstructed Variants'
        
        # Currently hard coded to only take in m-3m
        cs_v.pg_symmetry = symmetry.get_point_group(220)
        
        # color of the phase
        # takes an input as a string
        # X11/CSS4 color name with no space
        # e.g 'aquamarine' | 'seamediumgreen'
        # Use 'Violet' for testing
        cs_v.color = plt.colors(color = rgb)
        
        # Axes definition of the material
        cs_v.axes = Vector3d(np.array([[M[0], 0, 0], [0, M[1], 0], 
                                        [0, 0, M[2]]]))     


###############
## Functions ##
###############

# MATLAB ismember function
def ismember(d, k):
  return [1 if (i == k) else 0 for i in d]

# First step of the reconstruction prep
# Checks the user inputs to make sure phases are correct
def prep_for_recon(EBSD, options, material, HT_Id, LT_Id):
   if material == "Steel":
       if path.exist("HT_Id", "var"):
           ebsd = (EBSD, options, material, HT_Id, LT_Id)
       else:
           ebsd = (EBSD, options)
   elif material == "Titanium":
           raise Exception("Error: Titanium not yet implemented")
   else:
           raise Exception("""Error: The only valid options.material values 
                           are 'Titanium' and 'Steel'""")
   print("Loaded")
   
   return ebsd

# Returns the dictionary of all the values unindexed           
def CSList(CS_HT, CS_LT, CS_R, CS_V):
    

    return {CS_HT, CS_LT, CS_R, CS_V, 'notIndexed'}

# Scanning the CSList for mineral names or cell dimensions that make sense
# NOTE: At some point, some edge case will break the search 
#       at that point people will have to choose the HT/LT themselves
#       or write their edge case into loop
def determine_HT_LT_Steel(old_CSList):
    # n is the number of phases that we are creating
    # should be around 3 for this case specifically
    n = len(old_CSList)
    
    # Not sure this is the correct way to do a string array
    # Grab the searchable database
    
    # Creation of a string array to store the characters (??)
    names = ["" for x in range(n)] 
    
    # creation of an array of zeros size n to store characters for later
    cell_dims = np.zeros(n)
    
    # creation of an array to store the phases
    phase_linspace = list(range(1,n+1))
    
    for i in n:
        try:
            [a, b, c] = old_CSList(i).axes.double
            names[1,i] = str(old_CSList(i).phase)
            cell_dims[1,i] = stats.mean(a + b + c)
        except:
            print("An exception occurred")
            
    # First try searching by phase names. Any phase that matches, remove from
    # remaining searches
 
    # Try Martensite names
    LT_Id = min(phase_linspace.__contains__('art' in names) == 1)
    if len(LT_Id) != 0:
        names[LT_Id] = ""
        cell_dims[LT_Id] = 0
    
    # Try Austenite names    
    HT_Id = min(phase_linspace.__contains__('ust' in names) == 1)    
    if len(LT_Id) != 0:
        names[LT_Id] = ""
        cell_dims[LT_Id] = 0
        
    # If either failed
    # matching phase options
    # with the expected cell dimensions taken from the options object
    
    # Martensite first
    if len(LT_Id) == 0:
        LT_abc = [2.87, 2.87, 2.87]
       
       # [~,LT_Id] = min((cell_dims-LT_abc(1)).^2); 
        np.min(LT_Id, axis = 0)
        min((cell_dims - np.square(LT_abc(0))))
        names[LT_Id] = ""
        cell_dims[LT_Id] = 0
        
    # Austenite
    if len(HT_Id) == 0:
        HT_abc = [3.65, 3.65, 3.65] 
        [delta, HT_Id] = min((cell_dims-np.square(HT_abc(0))))
        if delta > 0.05:
            HT_Id = 60
    
    return HT_Id, LT_Id

def steel_prep(ebsd, HT_Id, LT_Id):
    phaseIds =  ebsd.phaseId + 100
    phaseMap = ebsd.phaseMap
    # CSList = make_Steel_CSList.material_sys('a', 'b', 'c', 'd')
    if 'HT_Id' and 'LT_Id' in steel_prep():
        assert(isinstance(LT_Id, int))
        print("LT_Id must be an integer value")
        assert(isinstance(HT_Id, int))
        print("HT_Id must be an integer value")
        assert(ismember(LT_Id, phaseMap))
        if ismember(LT_Id, phaseMap):
            pass
        else:
            HT_Id = 60
    # Scan the pahse data for mineral names or cell dimensions that make sense
    else:
        [HT_Id, LT_Id] = determine_HT_LT_Steel(
            ebsd.CS_HT(), ebsd.CS_LT())
    
    # Using the HT_Id and LT_Id, fix the phaseIds array
    phaseIds(phaseIds == (100 + HT_Id)) == 1
    phaseIds(phaseIds == (100+LT_Id)) == 2
    phaseIds(phaseIds > 80) == 0
    
    # At this point, we want to rearrange the ebsd phase data to match the
    # standardized format. 
    
    ebsd.phaseId = phaseIds*0;
    ebsd.phaseMap = [0] #ok<NBRAK> 
    
    # Then repopulate
    ebsd.CSList = CSList;
    ebsd.phaseMap = [1,2,3,4,0];
    ebsd.phaseId = phaseIds;

    # at this point, we have identical scans with the following phase IDs:
    # 1 : Untransformed Parent
    # 2 : Transformed Child
    # 3 : Reconstructed Parent (starts empty)
    # 4 : Idealized Variants (starts empty)
    # 0 : Not Indexed (will also include phases like pearlite or cemetite that aren't part of the resonstruction)
   
    return ebsd

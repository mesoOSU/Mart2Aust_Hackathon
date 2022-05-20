# -*- coding: utf-8 -*-
"""
Created on Wed May 18 09:22:30 2022

@author: xucha
"""
# Import of the dictionary

import load_options
import numpy as np

from orix.io import load
from orix import crystal_map
from orix.quaternion.symmetry import get_point_group
from orix.crystal_map import Phase, PhaseList

#===================#
#     Functions     #
#===================#

# Imports the steel dictionary and checks to see what the user input
def dictionary_return(mat_name):
    check = load_options.material_load(mat_name)
    steel_dict = load_options.steel_dictionary(check)
    
    return steel_dict

# Function 3 from the matlab code
# The old_cslist is the raw data that is from ebsd.phases
# We want to write over the stuff so that all the values are correct
def determine_HT_LT_Steel(nested_dictionary):
    
    #Hard code the high temperature
    HT_Id = Phase(nested_dictionary['CS_HT']['mineral'], 
               None,
               nested_dictionary['CS_HT']['crystalSymmetry'].name,
               None, "tab:blue"
               ) 
    
    LT_Id = Phase(nested_dictionary['CS_LT']['mineral'], 
               None,
               nested_dictionary['CS_LT']['crystalSymmetry'].name,
               None, "tab:orange"
               )
    
    return [HT_Id, LT_Id]
    
    
    
#     return [HT_Id, LT_Id]
#==============#
#     Main     #
#==============#

#============================#
#     Function 1 Results     #
#============================#

# Takes input from user
# Breaks my reader so commented out for now 
# material_name = input("Material name: ") 

# Hard coded material name for testing
material_name = ('Steel')
steel_data = dictionary_return(material_name)

#============================#
#     Function 2 Results     #
#============================#

#Retruns a nested dictionary for the seperate sections
steel_data_sorted = load_options.nested_dictionary(steel_data)

#============================#
#     Function 3 Results     #
#============================#

#need to overwrite the wrong headers and phases with the correct one below
HT_LT_Id = determine_HT_LT_Steel(steel_data_sorted)


#============================#
#     Function 4 Results     #
#============================#


#Loads in the crystal map = ebsd data
test_map = load('AF_001.ang')



























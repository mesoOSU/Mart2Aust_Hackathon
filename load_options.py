# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:37:51 2022

@author: xucha
"""
from matplotlib import pyplot as plt


# The load up of the material being chosen
def load_options(preset = 'Steel'):
    
    return preset

# Testing Purposes
#Should be user input
#Crashes my environment when I attempt
material = load_options()

#Check for Steel as the key word
if 'eel' in material:
    # Dictionary for Steel
    options_struct = {
                        # Material System 
                        # (determines parent/child relationship) 
                        'material' : 'Steel', 'specimen_symmetry' : 'triclinic',
                        ## High Temperature Phase (Phase 1, Austenite in Steel) ##
                        # This is the phase ID assigned to Untransformed 
                        # or "retained" areas
                        'High_Temp_phase_name' : 'Austenite', 
                        'High_Temp_phase_color' : plt.plot(color = 'lightgreen'),
                        'High_Temp_phase_symm' : 'm-3m',
                        'High_Temp_lattice_parameters' : [3.65, 3.65, 3.65],
                         ## Low Temperature Phase (Phase 2, Martensite in Steel) ##
                         # This is the assigned to the transformed phase 
                         # in the original EBSD. It is the portion that 
                         # will be reconstructed by this code, and is 
                         # normally the bulk of the EBSD scan.
                         'Low_Temp_phase_name' : 'Martensite', 
                         'Low_Temp_phase_color' : plt.plot(color = 'darkred'),
                         'Low_Temp_phase_symm' : 'm-3m',
                         'Low_Temp_lattice_parameters' : [2.87, 2.87, 2.87],
                         ### Reconstructed phase ###
                         # (Phase 3, Identical CS to Phase 1) 
                         # The calculated orientations of the original 
                         # HT grain structure.
                         # Purely calculated, not in original EBSD.
                         'Reconstructed_phase_name' : 'Reconstructed Austenite', 
                         'Reconstructed_phase_color' : plt.plot(color = 'darkblue'),
                         ### Ideal Variant orientation ###
                         # (Phase 4, Identical CS to Phase 2) 
                         # The calculated ideal orientations of the 
                         # variants prior to deformation.
                         'Variant_phase_name' : 'Reconstructed Variants',
                         'Variant_phase_color' : plt.plot(color = 'violet'),
                          ### Orientation Relationship (OR) Information ###
                          # Allows users to manually set the OR
                          # Leave bank to auto-calculate
                         'OR_ksi' : [0, 0, 0], 'OR_noise' : 0,
                         'OR_sampling_size' : 2000, 'OR_fit_TolFun' : 1e-4,
                         'OR_fit_TolX' : 1e-4, 'OR_plot_PAG_Mart_Guess' : 0,
                         'OR_plot_ODF_of_PAG_OR_guess' : 0,
                         'OR_plot_ksi_spread' : 0,
                         ### Reconstruction Graph Cut Parameters ###
                         'RGC_in_plane_m' : 3, 'RGC_in_plane_b' : 12,
                         'RGC_post_pre_m' : 0.175, 'RGC_post_pre_b' : 0.35,
                         'degree_of_connections_for_neighborhood' : 1,
                         'min_cut_size' : 5, 'max_recon_attempts' : 500,
                         ### Segmentation Parameters ###
                         'Seg_IP_m' : 25, 'Seg_IP_b' : 0, 'Seg_OP_m' : 1, 
                         'Seg_OP_b' : 0, 'Seg_Denoise' : False, 
                         ### Post Recon options ###
                         'calc_grain_metrics' : 1, 'variant_segmentation' : 1,
                         'calc_variant_metrics' : 1, 'plot_packets' : 1,
                         'plot_blocks' : 1, 'plot_variants' : 1,
                         'output_text_file' : 1 
                             
                    }
#Check for Titanium as the key word    
elif 'ium' in material:
    print("There is no current work done on Titanium")
    
else:
    print("Please select either Steel or Titanium as the material")

        

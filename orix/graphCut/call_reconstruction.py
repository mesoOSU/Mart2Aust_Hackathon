# -*- coding: utf-8 -*-
"""
Created on Monday May 16 13:39:19 2022

@author: simon

main call function run on LT ebsd and returns HT ebsd reconstruction
secondary functions used in call_recon
"""

###imports
from orix import data, io, plot
from orix.io import load
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Orientation, Rotation, symmetry, Misorientation
from orix.vector import Vector3d

import numpy as np
import random
import maxflow
import networkx as nx
from scipy.io import loadmat
import scipy.sparse as sp
import os
from orix.graphCut.temp_functions import yardley_variants
from orix.graphCut.graph_cut import graph_cut
import matplotlib.pyplot as plt

# def get_raw_ip_op_weights(orig_xmap, ip_connectivity):

#     ## Calculate misorientations for in-plane weights

#     #m = (~o1).outer(o2) # from orix documentation, but slow and has memory problems
#     o1 = orig_xmap.rotations[ip_connectivity[0,:]] # orientation of node u
#     o2 = orig_xmap.rotations[ip_connectivity[1,:]] # orientation of node v
#     raw_ip_weights = Misorientation(o1*o2.conj).to_euler() # misorientations between every u and v

#     allO = orig_xmap.rotations # orientation of node u
#     # o2 = xmap2.rotations[connectivity2[1,:]] # orientation of node v
#     raw_op_weights = Misorientation(allO*allO[4044].conj).to_euler() # misorientations between every u and v

#     return raw_ip_weights, raw_op_weights

def get_ip_weights(orig_xmap, ip_connectivity):
    """_summary_

    Args:
        orig_xmap (_type_): _description_
        ip_connectivity (_type_): _description_

    Returns:
        _type_: _description_
    """
    ## Calculate misorientations for in-plane weights

    #m = (~o1).outer(o2) # from orix documentation, but slow and has memory problems
    o1 = orig_xmap.rotations[ip_connectivity[0,:]] # orientation of node u
    o2 = orig_xmap.rotations[ip_connectivity[1,:]] # orientation of node v
    # mis = o1-o2
    raw_ip_weights = Misorientation(o1*o2.conj).to_euler() # misorientations between every u and v

    ### input raw weights
    ### pass through MDF
    ### return actual ip weights

    data_p = 'Mart2Aust/orix/graphCut/data/ipw_w.mat'
    mat_data = loadmat(data_p)
    ip_weights = mat_data['ipw_w'] # or 'ipw_w'
    ip_weights = ip_weights.T

    return ip_weights

def get_op_weights(active_xmap):
    """_summary_

    Args:
        active_xmap (_type_): _description_

    Returns:
        _type_: _description_
    """

    Guess_ID = random.randint(0, active_xmap.size)
    active_ori = active_xmap.orientations
    HT_guess_ori = active_ori[Guess_ID]
    raw_op_weights = active_ori - HT_guess_ori

    ### old method
    # allO = orig_xmap.rotations # orientation of node u
    # Guess_ID = random.randint(0, allO.size)
    # o2 = xmap2.rotations[connectivity2[1,:]] # orientation of node v
    # raw_op_weights = Misorientation(allO*allO[Guess_ID].conj).to_euler() # misorientations between every u and v

    ### input raw weights
    ### pass through MDF
    ### return actual 0p weights

    data_p = 'Mart2Aust/orix/graphCut/data/opw_w.mat'
    mat_data = loadmat(data_p)
    op_weights = mat_data['opw_w']
    op_weights = op_weights.T

    return op_weights

def get_op_weights_v2(xmap):
    # get yardley variants as orix rotation objects
    ksi = 'ks'
    yv_payton = yardley_variants('ks')
    yv = Rotation(np.vstack([Rotation.from_matrix(np.vstack(yv_payton[i])).data for i in range(yv_payton.shape[0])]))
    # LOAD EBSD
    xmap.phases[1].name = 'Aust'
    xmap.phases[2].name = 'Mart'
    # pick an id randomly that is martensite
    guess_id = xmap['Mart'].id[np.random.randint(xmap['Mart'].id.size)]
    guess_ori = xmap[xmap.id == guess_id].orientations
    inv_guess_ori = Orientation(guess_ori.conj,symmetry.O)
    # rotate everything by that guess_ori
    rotated_mart = xmap['Mart'].orientations*inv_guess_ori
    # Find the distance between those rotated orientations and the variants
    all_misos = rotated_mart.angle_with_outer(Orientation(yv,symmetry.O))
    # Pick the min
    misos = np.min(all_misos, axis = 1)
    # turn that into a likelyhood. This assumes the orientation spread is perfectly
    # symmetric( it isn't), and that this is the correct de la valle Poisson 
    # equation (it might be). It will get you close though.
    hw = 2*np.pi/180
    likelyhood = np.log(0.5**0.5)/np.log(np.cos(misos*hw/2))
    like_map = np.zeros([321*321])
    like_map[xmap['Mart'].id] = likelyhood
    like_map = np.log(like_map.reshape(321,321))
    #scipy_oris = R.from_euler('ZXZ',orix_oris.as_euler())

    return like_map

def first_pass_cut(active_xmap, ip_data, options):
    """_summary_

    Args:
        active_xmap (_type_): _description_
        ip_data (_type_): _description_
        options (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Does a single layer graph cut to pull out an area that likely contains at
    # least one prior austenite grain, plus some surrounding materials

    # do some setup stuff
    ##HT_guess_ori = orientation.byEuler(296.5*degree,231.8*degree,49.4*degree,active_xmap.CS) # erase this later
    ####HT_guess_ori = orientation.byEuler(68*degree,140.5*degree,185.5*degree,active_xmap.CS) # erase this later
    
    # [pruned_ip_connectivity,pruned_ip_weights] = prune_IP_graph_connections(active_xmap.id,ip_connectivity,ip_weights)

    ### add calc op_weights
    # op_weights = get_op_weights(active_xmap)
    op_weights = get_op_weights_v2(active_xmap)
    plt.imshow(op_weights)

    ####likelyhoods y=mx+b

    rough_graph = graph_cut(options['NROWS'], options['NCOLS_ODD'], options['GRID_TYPE'], ip_data, op_weights)

    rough_graph.g.maxflow()

    sgm = rough_graph.g.get_grid_segments(rough_graph.nodeids)
    img3 = np.int_(np.logical_not(sgm))
    from matplotlib import pyplot as ppl
    ppl.figure("first")
    ppl.imshow(img3)
    ppl.show()

    # Perform graph cut
    # [~,~,cs,~]=maxflow(FP_digraph,N+1,N+2)
    # cs[cs > np.len(active_xmap)] = []
    sgmVector = np.reshape(sgm, sgm.shape[0]*sgm.shape[1])
    sgmVector = sgmVector[:active_xmap.size]
    Rough_Guess = active_xmap[sgmVector]
    # print(Rough_Guess.size, active_xmap.size)
    # Rough_Guess.plot(overlay=Rough_Guess.ci)
    # Code for debugging to show the cut out area. NOTE: this is not a grain,
    # Its just a region of the scan that likely has at least one complete grain
    # in it
    ##figure()
    ##plot(Rough_Guess,Rough_Guess.orientations)
    # Here is some extra troubleshooting code for seeing the IP weights, which
    # are what this algorithm is trying to cut along (blue = weak,
    # yellow=strong)
    ##figure()
    #l = pruned_ip_connectivity(:,1)
    #r = pruned_ip_connectivity(:,2)
    #scatter(-active_xmap(l).y -active_xmap(r).y, active_xmap(l).x +active_xmap(r).x,1, pruned_ip_weights)
    return Rough_Guess

def precision_cut(Rough_Guess,ip_connectivity,ip_weights,options):
    """_summary_

    Args:
        Rough_Guess (_type_): _description_
        ip_connectivity (_type_): _description_
        ip_weights (_type_): _description_
        options (_type_): _description_

    Returns:
        _type_: _description_
    """
    # # Starting with a rough prior cut, this cut finds the most common high temp
    # # orientation and cuts out JUST that grain and and twins of it.

    # # do some setup stuff (note this is a more heavily pruned starting list
    # # than the rough cut)

    # [pruned_ip_connectivity,pruned_ip_weights] = prune_IP_graph_connections(Rough_Guess.id,ip_connectivity,ip_weights)
    # N = np.size(Rough_Guess)[0] # number of voxels
    # L = pruned_ip_connectivity[:,] # left side of ip_connectivity connections
    # R = pruned_ip_connectivity[:,2] # right side of ip_connectivity connections

    # OR = Rough_Guess.orientations
    # HT_CS = Rough_Guess.phases
    # [T2R,~] = calc_T2R(OR,Rough_Guess.CSList(3),Rough_Guess.CSList(2))
    # [R2T,~] = calc_R2T(OR,Rough_Guess.CSList(3),Rough_Guess.CSList(2))

    # psi = Rough_Guess.opt.psi
    # oris = Rough_Guess.orientations

    # T2r=ori*yardley
    # r2t=roi*invYardley

    # # find the most likely HT orientation, deduce (if applicable), generate all
    # # LT variants of the parent and twin (120 non-unique for steel), and build
    # # an ODF from them whose kernel spread matches the estimated per-variant
    # # spread of the LT phase (determined in the Auto-OR script). This is how we
    # # will weight the out of plane weights. (STEVE: speed this up if you can, 
    # # huge time sink. approx 10-30# of run total time on the next few lines)
    # Possible_PAG_oris = rotation(symmetrise(oris))*T2R
    # Possible_PAG_oris.CS = HT_CS
    # Parent_odf=calcDensity(Possible_PAG_oris,'kernel',psi)
    # [~,Guess_ori] = max(Parent_odf)
    # Guess_rot = rotation.byEuler(Guess_ori.phi1,Guess_ori.Phi,Guess_ori.phi2,HT_CS)
    # if options.material == "Steel":
    #     twin_rots = rotation.byAxisAngle(vector3d([1,1,1 -1,-1,1 -1,1,1 1,-1,1]'),60*degree,HT_CS)
    # else:
    #     twin_rots = idquaternion

    # PT_rots = [Guess_rottranspose(Guess_rot*twin_rots)]
    # PT_oris = orientation(PT_rots,Rough_Guess.CS)
    # system_variants = rotation(symmetrise(PT_oris))*R2T
    # parent_twin_odf = calcDensity(system_variants,'kernel',psi)
    # parent_twin_odf.CS = HT_CS

    # tester figures
    #figure()
    #plotPDF(oris,Miller(0, 0, 1,HT_CS), 'all')
    #hold on
    #plotPDF(parent_twin_odf,Miller(0, 0, 1,HT_CS), 'all')

    # # NOTE: LAST TIME, we weighted the OP weights by the MISORIENTATION between
    # # what was there and the HT orientation we wanted. Now though, it is the
    # # likelyhood that a given ORIENTATION is part of a HT Parent and/or Twin
    # likelyhoods = eval(parent_twin_odf,oris)##ok<EV2IN> 
    # likelyhoods(likelyhoods <=0) = 0
    # OP_wts = (likelyhoods*options.RGC_post_pre_m) + options.RGC_post_pre_b

    # plotting for help
    #figure()
    #plot(parent_twin_odf)
    ##figure()
    ##plot(Rough_Guess,OP_wts)
    #figure()
    #l = pruned_ip_connectivity(:,1)
    #r = pruned_ip_connectivity(:,2)
    #scatter(-Rough_Guess(L).y -Rough_Guess(R).y, Rough_Guess(L).x +Rough_Guess(R).x,1, pruned_ip_weights)


    # # make a digraph with n+2 nodes (1 per voxel,plus source and sink)
    # # NOTE: source has ID n+1, sink has ID n+2
    # FP_digraph = digraph
    # FP_digraph = addnode(FP_digraph,N+2)

    # # add source-to-voxel weights (equal to likelyhood that a given voxel's
    # # orientation is part of the grain of the suggested orientation)
    # FP_digraph = addedge(FP_digraph, N+1, 1:N, OP_wts)
    # # add voxel-to-sink weights (PRECISION CUT: WE ARE NOW WEIGHTING BY
    # # INVERSE OF SOURCE-TO-VOXEL WEIGHTS)
    # FP_digraph = addedge(FP_digraph, 1:N, N+2, 4./OP_wts)

    # # Add in-plane (voxel to voxel) connections)
    # FP_digraph = addedge(FP_digraph,L,R,pruned_ip_weights)
    # FP_digraph = addedge(FP_digraph,R,L,pruned_ip_weights)


    # # Perform graph cut
    # [~,~,cs,~]=maxflow(FP_digraph,N+1,N+2)
    # cs(cs>length(Rough_Guess)) = []
    # proposed_grain = Rough_Guess(cs)
    # Code for debugging to show the cut out area. NOTE: this is not a grain,
    # Its just a region of the scan that likely has at least one complete grain
    # in it
    ##figure()
    ##plot(proposed_grain,proposed_grain.orientations)
    # Here is some extra troubleshooting code for seeing the IP weights, which
    # are what this algorithm is trying to cut along (blue = weak,
    # yellow=strong)
    #figure()
    #l = pruned_ip_connectivity(:,1)
    #r = pruned_ip_connectivity(:,2)
    #scatter(-active_xmap(l).y -active_xmap(r).y, active_xmap(l).x +active_xmap(r).x,1, ip_weights)

    # return proposed_grain, PT_oris
    return None, None

def reconstruct_HT_grains(orig_xmap, ip_data, options):
    """_summary_

    Args:
        orig_xmap (_type_): _description_
        ip_data (_type_): _description_
        options (_type_): _description_
    """

    active_xmap = orig_xmap[orig_xmap.phases[options['MART_PHASE_ID']].name]
    continue_recon = True
    iterations = 0
    bad_cut_counter = 0

    while continue_recon == 1:
        # checks to break out of while loop
        if iterations > options['max_recon_attempts'] or bad_cut_counter > 10:
            continue_recon = False
            print('\n=========================================')
            print('    reconstruction failed to complete    ')
            print('    #0.0f voxels remain untransformed\n',sum(orig_xmap[options['MART_PHASE_ID']]))
            print('=========================================\n')
            continue
        if sum(active_xmap.phase_id == options['MART_PHASE_ID']) == 0:
            continue_recon = False
            print('\n=========================================')
            print('reconstruction completed successfully!')
            print('=========================================\n')
            continue

        ## ======== Step 1 ======== ##
        # Find a likely high temp grain orientation to attempt to cut out.
        # Guess_ID = random.randint(0, active_xmap.size)
        # active_ori = active_xmap.orientations
        # HT_guess_ori = active_ori[Guess_ID]
        # Mart = ebsd[ebsd.phases.names[2]]
        # Mart.flatten[np.random.randint(Mart.size)]

        Rough_Guess = first_pass_cut(active_xmap, ip_data, options) ##HT_guess_ori
        # is the grain big enough? if not, iterate counters and try again.
        if Rough_Guess.size < options['min_cut_size']:
            bad_cut_counter = bad_cut_counter+1
            iterations = iterations+1
            continue

        ## ======== Step 2 ======== ##
        # Now we want to clean up this guess with a precision cut, where we
        # let the code choose the most likely parent orientation instead of
        # guessing
        [proposed_grain, PT_oris] = precision_cut(Rough_Guess,ip_data,options)
        # is the grain big enough? if not, iterate counters and try again.
        if np.size(proposed_grain)[0] < options['min_cut_size']:
            bad_cut_counter = bad_cut_counter +1
            iterations = iterations+1
            continue

    #     ## ======== Step 3 ======== ##
    #     # if the grain made it this far, reset the counters and do 5 graph
    #     # cuts (1 for the parent, 4 for the twins) to find the parent/twin areas
    #     bad_cut_counter = 0
    #     # [PT_ID_stack] = seperate_twins(proposed_grain, sparse_adjacency, ip_weights, PT_oris,options)
    #     # Use this data to overwrite the orientation and phase data in LT_ebsd
    #     for i in np.arange(np.size(PT_oris)[0]):
    #         mask = PT_ID_stack[i,:]
    #         if sum(mask) < 5:
    #             continue
    #         mask = mask(mask>0)
    #         IDs_to_assign = proposed_grain(mask).id
    #         # LT_ebsd(ismember(LT_ebsd.id, IDs_to_assign)).orientations = PT_oris(i)
    #         # LT_ebsd(ismember(LT_ebsd.id, IDs_to_assign)).phase = 3

    #     # Prune out any orphaned pixels
    # #    orphan_IDs = prune_orphaned_pixels(active_xmap,ip_connectivity)
    # #    if size(orphan_IDs,1) >0
    # #        print('slicing orphans\n' )
    # #        orphan_IDs
    # #        LT_ebsd(ismember(LT_ebsd.id,orphan_IDs) == 1).phase = 1
    # #    end

#       # Now prune the active_xmap to only include untransformed stuff.
        # active_xmap = LT_ebsd(LT_ebsd[MART_PHASE_ID])
        # mart = ebsd[ebsd.phases[2].name]

        # report on reconstruction progress
        # iterations = iterations +1
        # print('\n ------ Iter: #d Pcnt: #0.2f Remainder:#0.0f ------\n',
        #     iterations,
        #     sum(LT_ebsd.phaseId == 2)*100./np.size(LT_ebsd)[0],
        #     sum(LT_ebsd.phaseId == 2))
    # At this point, either the reconstruction is finished, or it failed to
    # complete. either way, send the final result back to Aus_Recon
    LT_ebsd2 = 0

    return LT_ebsd2


def call_reconstruction(orig_xmap, options, LT_MDF=None):
    """_summary_

    Args:
        orig_xmap (_type_): _description_
        options (_type_): _description_
        LT_MDF (_type_, optional): _description_. Defaults to None.
    """

    # If users provide a misorientation distribution function for the low-temp  phase (LT_MDF), overwrite the saved one with that
#     if LT_MDF != None:
#         ###pseudocode until we know how to shove MDF into orix.ebsd object
#         orig_ebsd.odf.mdf = LT_MDF

    connect_graph = graph_cut(options['NROWS'], options['NCOLS_ODD'], options['GRID_TYPE'])
    print(connect_graph)
    print(connect_graph.nxGraph)

    ip_connectivity = connect_graph.get_connectivity()

    ip_weights = get_ip_weights(orig_xmap, ip_connectivity)
    # op_weights = get_op_weights(orig_xmap)

    ### ip data is connections and weights
    ### something is diff from yesterday so i have to slice loaded matlab weights
    ### this shouldnt be needed and wont be when we can python calc MDF
    ip_data = np.hstack((ip_connectivity, ip_weights[:205436,:])) ###TODO lower triangle this data??

    recon_ebsd = reconstruct_HT_grains(orig_xmap, ip_data, options)

    # Now do the actual reconstruction as a seperate function (helps avoid
    # accidental overwrites and removes pre-flight parameters)
    # recon_ebsd = reconstruct_HT_grains(LT_ebsd, sparse_adjacency, ip_weights, options, MART_PHASE_ID)
    # Now copy the original ebsd and replace the old values for the LT phases
    # with the new ones.
#     temp_ebsd = recon_ebsd
#     temp_ebsd.phase = MART_PHASE_ID
#     HT_ebsd = orig_ebsd
#     HT_ebsd(orig_ebsd.phaseId == MART_PHASE_ID).orientations = temp_ebsd.orientations
#     HT_ebsd(orig_ebsd.phaseId == MART_PHASE_ID).phaseId = recon_ebsd.phaseId

#     return HT_ebsd

    return None


# def first_pass_cut(active_xmap,ip_connectivity,ip_weights,HT_guess_ori,options):
    # Does a single layer graph cut to pull out an area that likely contains at
    # least one prior austenite grain, plus some surrounding materials

    # do some setup stuff
    ###HT_guess_ori = orientation.byEuler(296.5*degree,231.8*degree,49.4*degree,active_xmap.CS) # erase this later
    #####HT_guess_ori = orientation.byEuler(68*degree,140.5*degree,185.5*degree,active_xmap.CS) # erase this later
    # [pruned_ip_connectivity,pruned_ip_weights] = prune_IP_graph_connections(active_xmap.id,ip_connectivity,ip_weights)

    # N = np.size(active_xmap)[0] # number of voxels
    # L = pruned_ip_connectivity[:,1] # left side of ip_connectivity connections
    # R = pruned_ip_connectivity[:,2]# right side of ip_connectivity connections
    # oris = active_xmap.orientations
    # # mori = inv(HT_guess_ori)*oris
    # LT_MDF = active_xmap.opt.LT_MDF
    # #clear active_xmap
    # # likelyhoods = alt_eval(LT_MDF,mori)
    # likelyhoods[likelyhoods <= 0] = 0
    # #lll = ((likelyhoods)+2)*0.175

    # # temp plotting stuff
    # ##figure()
    # ##plot(active_xmap,lll)

    # # make a digraph with n+2 nodes (1 per voxel,plus source and sink)
    # # NOTE: source has ID n+1, sink has ID n+2

    # FP_digraph = graph_cut(nrows, ncols_odd, grid)
    # FP_digraph = addnode(FP_digraph,N+2)

    # # add source-to-voxel weights (equal to likelyhood that a given voxel's
    # # orientation came from the suggested prior austenite ODF, as calculated
    # # from the transformed phase's MDF)
    # # NOTE: eval calls take time, so we should make as few as possible. However,
    # # this one changes every loop. Could pregen several lists from the start,
    # # but that would be memory intensive so probably not worth it.
    # #likelyhoods = alt_eval(LT_MDF,mori)
    # #likelyhoods(likelyhoods <=0) = 0
    # OP_wts = (likelyhoods*options.RGC_post_pre_m) + options.RGC_post_pre_b
    # # OP_wts = (likelyhoods +2)*0.175
    # FP_digraph = addedge[FP_digraph, N+1, 1:N, OP_wts]

    # # Add in-plane (voxel to voxel) connections)
    # FP_digraph = addedge[FP_digraph,L,R,pruned_ip_weights]
    # FP_digraph = addedge[FP_digraph,R,L,pruned_ip_weights]

    # # add voxel-to-sink weights (all equal to the mean of the OP weights)
    # FP_digraph = addedge[FP_digraph, 1:N, N+2, np.ones((N,1))*np.mean(OP_wts)]
    # #FP_digraph = addedge(FP_digraph, 1:N, N+2, 4./OP_wts)

    # # Perform graph cut
    # [~,~,cs,~]=maxflow(FP_digraph,N+1,N+2)
    # cs[cs > np.len(active_xmap)] = []
    # Rough_Guess = active_xmap(cs)
    # # Code for debugging to show the cut out area. NOTE: this is not a grain,
    # # Its just a region of the scan that likely has at least one complete grain
    # # in it
    # ##figure()
    # ##plot(Rough_Guess,Rough_Guess.orientations)
    # # Here is some extra troubleshooting code for seeing the IP weights, which
    # # are what this algorithm is trying to cut along (blue = weak,
    # # yellow=strong)
    # ##figure()
    ##l = pruned_ip_connectivity(:,1)
    ##r = pruned_ip_connectivity(:,2)
    ##scatter(-active_xmap(l).y -active_xmap(r).y, active_xmap(l).x +active_xmap(r).x,1, pruned_ip_weights)
    # Rough_Guess=None
    # return Rough_Guess



# def precision_cut(Rough_Guess,ip_connectivity,ip_weights,options):
#     # Starting with a rough prior cut, this cut finds the most common high temp
#     # orientation and cuts out JUST that grain and and twins of it.

#     # # do some setup stuff (note this is a more heavily pruned starting list
#     # # than the rough cut)
#     # [pruned_ip_connectivity,pruned_ip_weights] = prune_IP_graph_connections(Rough_Guess.id,ip_connectivity,ip_weights)
#     # N = np.size(Rough_Guess)[0] # number of voxels
#     # L = pruned_ip_connectivity[:,] # left side of ip_connectivity connections
#     # R = pruned_ip_connectivity[:,2]# right side of ip_connectivity connections
#     # OR = Rough_Guess.opt.OR
#     # HT_CS = Rough_Guess.CS
#     # [T2R,~] = calc_T2R(OR,Rough_Guess.CSList(3),Rough_Guess.CSList(2))
#     # [R2T,~] = calc_R2T(OR,Rough_Guess.CSList(3),Rough_Guess.CSList(2))
#     # psi = Rough_Guess.opt.psi
#     # oris = Rough_Guess.orientations

#     # # find the most likely HT orientation, deduce (if applicable), generate all
#     # # LT variants of the parent and twin (120 non-unique for steel), and build
#     # # an ODF from them whose kernel spread matches the estimated per-variant
#     # # spread of the LT phase (determined in the Auto-OR script). This is how we
#     # # will weight the out of plane weights. (STEVE: speed this up if you can, 
#     # # huge time sink. approx 10-30# of run total time on the next few lines)
#     # Possible_PAG_oris = rotation(symmetrise(oris))*T2R
#     # Possible_PAG_oris.CS = HT_CS
#     # Parent_odf=calcDensity(Possible_PAG_oris,'kernel',psi)
#     # [~,Guess_ori] = max(Parent_odf)
#     # Guess_rot = rotation.byEuler(Guess_ori.phi1,Guess_ori.Phi,Guess_ori.phi2,HT_CS)
#     # if options.material == "Steel":
#     #     twin_rots = rotation.byAxisAngle(vector3d([1,1,1 -1,-1,1 -1,1,1 1,-1,1]'),60*degree,HT_CS)
#     # else:
#     #     twin_rots = idquaternion

#     # PT_rots = [Guess_rottranspose(Guess_rot*twin_rots)]
#     # PT_oris = orientation(PT_rots,Rough_Guess.CS)
#     # system_variants = rotation(symmetrise(PT_oris))*R2T
#     # parent_twin_odf = calcDensity(system_variants,'kernel',psi)
#     # parent_twin_odf.CS = HT_CS

#     # # tester figures
#     # #figure()
#     # #plotPDF(oris,Miller(0, 0, 1,HT_CS), 'all')
#     # #hold on
#     # #plotPDF(parent_twin_odf,Miller(0, 0, 1,HT_CS), 'all')

#     # # NOTE: LAST TIME, we weighted the OP weights by the MISORIENTATION between
#     # # what was there and the HT orientation we wanted. Now though, it is the
#     # # likelyhood that a given ORIENTATION is part of a HT Parent and/or Twin
#     # likelyhoods = eval(parent_twin_odf,oris)##ok<EV2IN> 
#     # likelyhoods(likelyhoods <=0) = 0
#     # OP_wts = (likelyhoods*options.RGC_post_pre_m) + options.RGC_post_pre_b

#     # plotting for help
#     #figure()
#     #plot(parent_twin_odf)
#     ##figure()
#     ##plot(Rough_Guess,OP_wts)
#     #figure()
#     #l = pruned_ip_connectivity(:,1)
#     #r = pruned_ip_connectivity(:,2)
#     #scatter(-Rough_Guess(L).y -Rough_Guess(R).y, Rough_Guess(L).x +Rough_Guess(R).x,1, pruned_ip_weights)


#     # make a digraph with n+2 nodes (1 per voxel,plus source and sink)
#     # # NOTE: source has ID n+1, sink has ID n+2
#     # FP_digraph = digraph
#     # FP_digraph = addnode(FP_digraph,N+2)

#     # # add source-to-voxel weights (equal to likelyhood that a given voxel's
#     # # orientation is part of the grain of the suggested orientation)
#     # FP_digraph = addedge(FP_digraph, N+1, 1:N, OP_wts)
#     # # add voxel-to-sink weights (PRECISION CUT: WE ARE NOW WEIGHTING BY
#     # # INVERSE OF SOURCE-TO-VOXEL WEIGHTS)
#     # FP_digraph = addedge(FP_digraph, 1:N, N+2, 4./OP_wts)

#     # # Add in-plane (voxel to voxel) connections)
#     # FP_digraph = addedge(FP_digraph,L,R,pruned_ip_weights)
#     # FP_digraph = addedge(FP_digraph,R,L,pruned_ip_weights)


#     # # Perform graph cut
#     # [~,~,cs,~]=maxflow(FP_digraph,N+1,N+2)
#     # cs(cs>length(Rough_Guess)) = []
#     # proposed_grain = Rough_Guess(cs)
#     # # Code for debugging to show the cut out area. NOTE: this is not a grain,
#     # # Its just a region of the scan that likely has at least one complete grain
#     # # in it
#     # ##figure()
#     # ##plot(proposed_grain,proposed_grain.orientations)
#     # # Here is some extra troubleshooting code for seeing the IP weights, which
#     # # are what this algorithm is trying to cut along (blue = weak,
#     # # yellow=strong)
#     # #figure()
#     # #l = pruned_ip_connectivity(:,1)
#     # #r = pruned_ip_connectivity(:,2)
#     # #scatter(-active_xmap(l).y -active_xmap(r).y, active_xmap(l).x +active_xmap(r).x,1, ip_weights)

#     return proposed_grain, PT_oris


# def seperate_twins(proposed_grain, ip_connectivity, ip_weights, PT_oris,options)
#     # Given we have a for sure parent grain, check to see if parts would make
#     # more sense as a twin or as part of the parent.

#     # do some setup stuff
#     [pruned_ip_connectivity,pruned_ip_weights] = prune_IP_graph_connections(proposed_grain.id,ip_connectivity,ip_weights)
#     N = np.size(proposed_grain)[0] # number of voxels
#     L = pruned_ip_connectivity[:,1] # left side of ip_connectivity connections
#     R = pruned_ip_connectivity[:,2] # right side of ip_connectivity connections
#     OR = proposed_grain.opt.OR
#     HT_CS = proposed_grain.CS
#     [R2T,~] = calc_R2T(OR,proposed_grain.CSList(3),proposed_grain.CSList(2))
#     psi = proposed_grain.opt.psi
#     oris = proposed_grain.orientations

#     # We already got the most likely HT Parent grain. NOTE: We COULD
#     # recalculate the most likely grain here (as was done in the original
#     # code), but I am fairly sure it is mostly redundant to do so.

#     Parent_and_Twin_IDs = zeros(np.size(PT_oris)[0],np.size(oris)[0])

#     # Find the likelyhood that each pixel is part of the Parent grain. this
#     # will become the background weighting for the 4 twin cuts.
#     Parent_variants = rotation(symmetrise(PT_oris(1)))*R2T
#     Parent_odf = calcDensity(Parent_variants,'kernel',psi)
#     Parent_odf.CS = HT_CS
#     bg_likelyhoods = eval(Parent_odf,oris)##ok<EV2IN> 
#     bg_likelyhoods(bg_likelyhoods <=0) = 0
#     mx = (bg_likelyhoods*options.RGC_post_pre_m)
#     b =  options.RGC_post_pre_b
#     OP_Parent_wts = mx + b
#     clear Parent_variants Parent_odf bg_likelyhoods mx b

#     for i = 2:size(PT_oris,1)

#         # find the likelyhood that each pixel is part of grain i
#         Twin_variants = rotation(symmetrise(PT_oris(i)))*R2T
#         Twin_odf = calcDensity(Twin_variants,'kernel',psi)
#         Twin_odf.CS = HT_CS
#         fg_likelyhoods = eval(Twin_odf,oris)##ok<EV2IN> 
#         fg_likelyhoods(fg_likelyhoods<0) = 0
#         mx = (fg_likelyhoods*options.RGC_post_pre_m)
#         b =  options.RGC_post_pre_b
#         OP_twin_wts = mx + b
#         clear Twin_variants Twin_odf fg_likelyhoods mx b
        
#         # Set the weights of already assigned poitns to 1e-10, making already
#         # assigned grains nearly impossible to cut (should maybe be 0?)
#         assigned =(1:N).*(sum(Parent_and_Twin_IDs,1)>0)
#         OP_twin_wts(ismember((1:N),assigned)>0) = 1e-10

#         # make a digraph with n+2 nodes (1 per voxel, plus source and sink)
#         # NOTE: source has ID n+1, sink has ID n+2
#         FP_digraph = digraph
#         FP_digraph = addnode(FP_digraph,N+2)
#         # add source-to-voxel weights (likelyhood pixel is twin i)
#         FP_digraph = addedge(FP_digraph, N+1, 1:N, OP_twin_wts)
#         # add voxel-to-sink weights (likelyhood pixel is part of the parent)
#         FP_digraph = addedge(FP_digraph, 1:N, N+2, OP_Parent_wts)
#         # Add in-plane (voxel to voxel) connections)
#         FP_digraph = addedge(FP_digraph,L,R,pruned_ip_weights)
#         FP_digraph = addedge(FP_digraph,R,L,pruned_ip_weights)
#         # Perform graph cut
#         [~,~,cs,~]=maxflow(FP_digraph,N+1,N+2)
#         cs(cs>length(proposed_grain)) = []
#         # save out the mask of the cut
#     #    yes_mask = [1:N].*ismember([1:N],cs)
#         Parent_and_Twin_IDs(i,1:end) = ismember((1:N),cs)
#         # find the already assigned orientations, set their in-plane weights to
#         # zero so they won't get pulled out again.
#         assigned =(1:N).*(sum(Parent_and_Twin_IDs,1)>0)
#         neigh_mask = ((ismember(L,assigned) == 0) +(ismember(R,assigned) == 0))>0
#         pruned_ip_weights = pruned_ip_weights.*neigh_mask    

#         # plotting for help
#         #figure()
#         #plot(proposed_grain,OP_wts)
#         #figure()
#         #l = L(L>0)
#         #r = L(R>0)
#         #scatter(-proposed_grain(l).y -proposed_grain(r).y, ...
#         #    proposed_grain(l).x +proposed_grain(r).x,1, pruned_ip_weights)
#         ###figure()
#         ###plot(proposed_grain,Parent_and_Twin_IDs(i,1:end))
#     # get a map of twice-twinned (should never happen) and never twinned pixels
#     overlap_mask = sum(Parent_and_Twin_IDs,1) <= 1
#     parent_mask = sum(Parent_and_Twin_IDs,1) ~= 1
#     # Twice twinned pixels get switched to never twinned
#     Parent_and_Twin_IDs = Parent_and_Twin_IDs.*overlap_mask
#     # assign never twinned pixels to parent
#     Parent_and_Twin_IDs(1,:) = parent_mask
#     # and add the IDs back in (maybe make this step unnecessary in the future?)
#     Parent_and_Twin_IDs = Parent_and_Twin_IDs.*(1:N)

#     return Parent_and_Twin_IDs

# def prune_IP_graph_connections(ids,ip_connectivity,ip_weights):
#     # take the global neighbor list and in plane weights, prune out the
#     # connections that don't connect voxels in the ids list, and renumber them
#     # appropriately so the digraph function doesn't make orphan nodes
#     ids = sort(unique(ids))
#     prune_mask = ismember(ip_connectivity(:,1),ids).*ismember(ip_connectivity(:,2),ids)

#     # prune
#     pruned_ip_weights = ip_weights(prune_mask == 1)
#     pruned_ip_connectivity = ip_connectivity(prune_mask == 1,:)


#     # create translater array
#     translator = zeros(max(ids),1)
#     for i = 1:length(ids)
#         translator(ids(i))=i
    

#     #remap
#     l = translator(pruned_ip_connectivity(:,1))
#     r = translator(pruned_ip_connectivity(:,2))

#     remapped_ip_connectivity = [l r]

#     return remapped_ip_connectivity,pruned_ip_weights

# def get_ip_weights(orig_xmap):#(ip_connectivity,ebsd,options):
#     # use the misorientation between neighbor pairs to determine a "weight" to
#     # give to that connection in terms of network flow capacity.

#     # get misorientation angle between pairs
#     [~,id_Dl] = ismember(ip_connectivity(:,1),ebsd.id)
#     [~,id_Dr] = ismember(ip_connectivity(:,2),ebsd.id)
#     o_Dl = ebsd(id_Dl).orientations
#     o_Dr = ebsd(id_Dr).orientations
#     #o_Dl = ebsd(ip_connectivity(:,1)).orientations
#     #o_Dr = ebsd(ip_connectivity(:,2)).orientations
#     Mori = inv(o_Dl).*(o_Dr)

#     # Find likelyhoods for those misorientation angles to occur in the expected
#     # Martensite structure
#     LT_MDF_vals=alt_eval(ebsd.opt.LT_MDF,Mori)
#     LT_MDF_vals(LT_MDF_vals<0)=0

#     # Alter their weights using a y=mx+b style linear equation. For anyone
#     # reading this and confused, we are altering the "strength" of the
#     # pixel-to-pixel (ie, in-plane) connections. increasing the value of b
#     # increases the cost of making any in-plane cuts (ie, will favor a shorter
#     # overall length of grain boundaries). increasing m will increase the
#     # cost of making a cut through a likely pair compared to an unlikely pair.
#     m = options.RGC_in_plane_m
#     b = options.RGC_in_plane_b
#     ip_weights = LT_MDF_vals*m +b
#     #look up orientation of IDs in neighborhood list, find the angle between
#     # them, look up the likelyhood for that weight in the misorientation
#     # distribution function, and Finall

#     return ip_weights

# def get_out_of_plane_weights(oris,guess_ori,MDF,options):
#     # given a list of orientations, guess ori, and MODF, use eval to find the
#     # likelyhood that each orientation came from the guess orientation, then
#     # weight those values according to the OP weighting and scaling factors

#     # get misorientation angle
#     mori = inv(guess_ori).*oris
#     # query MDF for likelyhood of those misorientation angles
#     likelyhoods = alt_eval(MDF,mori)
#     likelyhoods(likelyhoods <=0) = 0
#     # apply a y = mx+b style rescaling factor. Increasing m makes close
#     # likelyhoods easier to include and unlikely grains harder. increasing b
#     # just makes ALL cuts equally harder (only really matters for precision
#     # cut, where some of the weights are done as 1/OP)
#     OP_wts = (likelyhoods*options.RGC_post_pre_m)+options.RGC_post_pre_b

#     return OP_wts

# def prune_orphaned_pixels(active_xmap,ip_connectivity):
#     # filter out pixels without any neighbors

#     a = ip_connectivity(ismember(ip_connectivity(1:end,1),active_xmap.id),2)
#     b = ip_connectivity(ismember(ip_connectivity(2:end,1),active_xmap.id),1)
#     ab = unique([a b])
#     orphan_IDs = active_xmap.id(ismember(active_xmap.id,ab) == 0)
    
#     # BIG NOTE HERE: FOR SOME REASON, the original code uses not just
#     # neighbors, but neighbors of neighbors in their code (IE,1st through
#     # 3rd 2D Von Neumann neighborhoods) to calculate the adjacency matrix. 
#     # This SEEMS wrong, as it isnt a typical adjacency array A, but the 
#     # equivilant of A +(A*A). It would also slow down the graph cut by a 
#     # factor of 4. HOWEVER, for some reason it works, so I'm keeping as an 
#     # option.To switch to first (IE, only 1st Von Neumann
#     # neighborhood) change options.degree_of_connections_for_neighborhood
#     # from 2 to 1.

#     return orphan_IDs






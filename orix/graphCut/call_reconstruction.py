# -*- coding: utf-8 -*-
"""
Created on Monday May 16 13:39:19 2022

@author: simon

main call function run on LT ebsd and returns HT ebsd reconstruction
"""

###imports
import numpy as np
import orix

def call_reconstruction(orig_ebsd,options,LT_MDF=None):
    '''
    
    Parameters
    ----------
    orig_ebsd : .ang ebsd file of low temperature as-measured form
    options : asdf
    LT_MDF : low temperature misorientation distribution function #possible that this will be calculated, not passed
        
    Returns
    -------
    HT_ebsd : 

    '''

    # If users provide a misorientation distribution function for the low-temp  phase (LT_MDF), overwrite the saved one with that
    if LT_MDF != None:
        orig_ebsd.opt.LT_MDF = LT_MDF

    # make a copy of the ebsd scan that is JUST the phase that needs
    # reconstructing. Everything else is ignored and left unchanged.
    LT_ebsd = orig_ebsd(orig_ebsd.phaseId == 2)
    # renumber

    # get in-plane weights by calculating adjaceny array using graph cut network generation
    # to network x output for sparse array of (node1, node2, edge_weight)
    sparse_adjacency, ip_weights = get_ip_weights(orig_ebsd)

    # Now do the actual reconstruction as a seperate function (helps avoid
    # accidental overwrites and removes pre-flight parameters)
    recon_ebsd = Reconstruct_HT_grains(LT_ebsd, sparse_adjacency, ip_weights, options)

    # Now copy the original ebsd and replace the old values for the LT phases
    # with the new ones.
    temp_ebsd = recon_ebsd
    temp_ebsd.phase = 2
    HT_ebsd = orig_ebsd
    HT_ebsd(orig_ebsd.phaseId == 2).orientations = temp_ebsd.orientations
    HT_ebsd(orig_ebsd.phaseId == 2).phaseId = recon_ebsd.phaseId

    return HT_ebsd


def Reconstruct_HT_grains(LT_ebsd, sparse_adjacency, ip_weights, options):
    #Reconstruct_HT_grains perform the actual reconstruction
    #   performs a 'while ' loop that progressively cuts out possible grains
    #   until eithr there is nothing left to cut, or the algorithm starts
    #   failing too much
    Active_Ebsd = LT_ebsd(LT_ebsd.phaseId == 2)
    continue_recon = True
    iterations = 0
    bad_cut_counter = 0
    #temp plotting stuff
    #close all
    ##figure()
    ##plot(Active_Ebsd,Active_Ebsd.orientations)

    while continue_recon == 1:
        # checks to break out of while loop
        if iterations > options.max_recon_attempts or bad_cut_counter > 10:
            continue_recon = False
            print('\n=========================================')
            print('    reconstruction failed to complete    ')
            print('    #0.0f voxels remain untransformed\n',sum(LT_ebsd.phaseId == 2))
            print('=========================================\n')
            continue
        if sum(LT_ebsd.phaseId == 2) == 0:
            continue_recon = False
            print('\n=========================================')
            print('reconstruction completed successfully!')
            print('=========================================\n')
            continue

        ## ======== Step 1 ======== ##
        # Find a likely high temp grain orientation to attempt to cut out.
        Guess_ID = randsample(1:length(Active_Ebsd),1,true,Active_Ebsd.ci)
        HT_guess_ori = Active_Ebsd(Guess_ID).orientations

        # Using that guess, make a rough cut to find a likely HT grain.If the
        # guess is a Parent orientation, it will get all Twins, but if it is a
        # Twin orientation, it will only grab the parent and not other twins of
        # that parent (this is fine, we fix this in a later step)
        Rough_Guess = First_Pass_Cut(Active_Ebsd,neigh_list,IP_wts,HT_guess_ori,options)
        # is the grain big enough? if not, iterate counters and try again.
        if size(Rough_Guess,1) < options.min_cut_size:
            bad_cut_counter = bad_cut_counter +1
            iterations = iterations+1
            continue

        ## ======== Step 2 ======== ##
        # Now we want to clean up this guess with a precision cut, where we
        # let the code choose the most likely parent orientation instead of
        # guessing
        [proposed_grain, PT_oris] = Precision_cut(Rough_Guess,neigh_list,IP_wts,options)
        # is the grain big enough? if not, iterate counters and try again.
        if size(proposed_grain,1) < options.min_cut_size:
            bad_cut_counter = bad_cut_counter +1
            iterations = iterations+1
            continue

        ## ======== Step 3 ======== ##
        # if the grain made it this far, reset the counters and do 5 graph
        # cuts (1 for the parent, 4 for the twins) to find the parent/twin areas
        bad_cut_counter = 0
        [PT_ID_stack] = Seperate_Twins(proposed_grain, neigh_list, IP_wts, PT_oris,options)
        # Use this data to overwrite the orientation and phase data in LT_ebsd
        for i in 111#1:size(PT_oris,1):
            mask = 111#PT_ID_stack(i,1:end)
            if sum(mask) < 5:
                continue
            mask = mask(mask>0)
            IDs_to_assign = proposed_grain(mask).id
            LT_ebsd(ismember(LT_ebsd.id, IDs_to_assign)).orientations = PT_oris(i)
            LT_ebsd(ismember(LT_ebsd.id, IDs_to_assign)).phase = 3

        # Prune out any orphaned pixels
    #    orphan_IDs = prune_orphaned_pixels(Active_Ebsd,neigh_list)
    #    if size(orphan_IDs,1) >0
    #        print('slicing orphans\n' )
    #        orphan_IDs
    #        LT_ebsd(ismember(LT_ebsd.id,orphan_IDs) == 1).phase = 1
    #    end

        # Now prune the Active_Ebsd to only include untransformed stuff.
        Active_Ebsd = LT_ebsd(LT_ebsd.phaseId == 2)

        # report on reconstruction progress
        iterations = iterations +1
        print('\n ------ Iter: #d Pcnt: #0.2f Remainder:#0.0f ------\n',...
            iterations,...
            sum(LT_ebsd.phaseId == 2)*100./size(LT_ebsd,1),...
            sum(LT_ebsd.phaseId == 2))
    # At this point, either the reconstruction is finished, or it failed to
    # complete. either way, send the final result back to Aus_Recon
    LT_ebsd2 = 0

    return LT_ebsd2


def First_Pass_Cut(Active_ebsd,neigh_list,IP_wts,HT_guess_ori,options):
    # Does a single layer graph cut to pull out an area that likely contains at
    # least one prior austenite grain, plus some surrounding materials

    # do some setup stuff
    ###HT_guess_ori = orientation.byEuler(296.5*degree,231.8*degree,49.4*degree,Active_ebsd.CS) # erase this later
    #####HT_guess_ori = orientation.byEuler(68*degree,140.5*degree,185.5*degree,Active_ebsd.CS) # erase this later
    [pruned_neigh_list,pruned_IP_wts] = prune_IP_graph_connections(Active_ebsd.id,neigh_list,IP_wts)
    N = size(Active_ebsd,1) # number of voxels
    L = pruned_neigh_list(:,1) # left side of neigh_list connections
    R = pruned_neigh_list(:,2)# right side of neigh_list connections
    oris = Active_ebsd.orientations
    mori = inv(HT_guess_ori).*oris
    LT_MDF = Active_ebsd.opt.LT_MDF
    #clear Active_ebsd
    likelyhoods = alt_eval(LT_MDF,mori)
    likelyhoods(likelyhoods <=0) = 0
    #lll = ((likelyhoods)+2)*0.175

    # temp plotting stuff
    ##figure()
    ##plot(Active_ebsd,lll)

    # make a digraph with n+2 nodes (1 per voxel,plus source and sink)
    # NOTE: source has ID n+1, sink has ID n+2
    FP_digraph = digraph
    FP_digraph = addnode(FP_digraph,N+2)

    # add source-to-voxel weights (equal to likelyhood that a given voxel's
    # orientation came from the suggested prior austenite ODF, as calculated
    # from the transformed phase's MDF)
    # NOTE: eval calls take time, so we should make as few as possible. However,
    # this one changes every loop. Could pregen several lists from the start,
    # but that would be memory intensive so probably not worth it.
    #likelyhoods = alt_eval(LT_MDF,mori)
    #likelyhoods(likelyhoods <=0) = 0
    OP_wts = (likelyhoods*options.RGC_post_pre_m) + options.RGC_post_pre_b
    # OP_wts = (likelyhoods +2)*0.175
    FP_digraph = addedge(FP_digraph, N+1, 1:N, OP_wts)

    # Add in-plane (voxel to voxel) connections)
    FP_digraph = addedge(FP_digraph,L,R,pruned_IP_wts)
    FP_digraph = addedge(FP_digraph,R,L,pruned_IP_wts)

    # add voxel-to-sink weights (all equal to the mean of the OP weights)
    FP_digraph = addedge(FP_digraph, 1:N, N+2, ones(N,1)*mean(OP_wts))
    #FP_digraph = addedge(FP_digraph, 1:N, N+2, 4./OP_wts)

    # Perform graph cut
    [~,~,cs,~]=maxflow(FP_digraph,N+1,N+2)
    cs(cs>length(Active_ebsd)) = []
    Rough_Guess = Active_ebsd(cs)
    # Code for debugging to show the cut out area. NOTE: this is not a grain,
    # Its just a region of the scan that likely has at least one complete grain
    # in it
    ##figure()
    ##plot(Rough_Guess,Rough_Guess.orientations)
    # Here is some extra troubleshooting code for seeing the IP weights, which
    # are what this algorithm is trying to cut along (blue = weak,
    # yellow=strong)
    ##figure()
    ##l = pruned_neigh_list(:,1)
    ##r = pruned_neigh_list(:,2)
    ##scatter(-Active_ebsd(l).y -Active_ebsd(r).y, Active_ebsd(l).x +Active_ebsd(r).x,1, pruned_IP_wts)
    
    return Rough_Guess



def Precision_cut(Rough_Guess,neigh_list,IP_wts,options):
    # Starting with a rough prior cut, this cut finds the most common high temp
    # orientation and cuts out JUST that grain and and twins of it.

    # do some setup stuff (note this is a more heavily pruned starting list
    # than the rough cut)
    [pruned_neigh_list,pruned_IP_wts] = prune_IP_graph_connections(Rough_Guess.id,neigh_list,IP_wts)
    N = size(Rough_Guess,1) # number of voxels
    L = pruned_neigh_list(:,1) # left side of neigh_list connections
    R = pruned_neigh_list(:,2)# right side of neigh_list connections
    OR = Rough_Guess.opt.OR
    HT_CS = Rough_Guess.CS
    [T2R,~] = calc_T2R(OR,Rough_Guess.CSList(3),Rough_Guess.CSList(2))
    [R2T,~] = calc_R2T(OR,Rough_Guess.CSList(3),Rough_Guess.CSList(2))
    psi = Rough_Guess.opt.psi
    oris = Rough_Guess.orientations

    # find the most likely HT orientation, deduce (if applicable), generate all
    # LT variants of the parent and twin (120 non-unique for steel), and build
    # an ODF from them whose kernel spread matches the estimated per-variant
    # spread of the LT phase (determined in the Auto-OR script). This is how we
    # will weight the out of plane weights. (STEVE: speed this up if you can, 
    # huge time sink. approx 10-30# of run total time on the next few lines)
    Possible_PAG_oris = rotation(symmetrise(oris))*T2R
    Possible_PAG_oris.CS = HT_CS
    Parent_odf=calcDensity(Possible_PAG_oris,'kernel',psi)
    [~,Guess_ori] = max(Parent_odf)
    Guess_rot = rotation.byEuler(Guess_ori.phi1,Guess_ori.Phi,Guess_ori.phi2,HT_CS)
    if options.material == "Steel":
        twin_rots = rotation.byAxisAngle(vector3d([1,1,1 -1,-1,1 -1,1,1 1,-1,1]'),60*degree,HT_CS)
    else:
        twin_rots = idquaternion

    PT_rots = [Guess_rottranspose(Guess_rot*twin_rots)]
    PT_oris = orientation(PT_rots,Rough_Guess.CS)
    system_variants = rotation(symmetrise(PT_oris))*R2T
    parent_twin_odf = calcDensity(system_variants,'kernel',psi)
    parent_twin_odf.CS = HT_CS

    # tester figures
    #figure()
    #plotPDF(oris,Miller(0, 0, 1,HT_CS), 'all')
    #hold on
    #plotPDF(parent_twin_odf,Miller(0, 0, 1,HT_CS), 'all')

    # NOTE: LAST TIME, we weighted the OP weights by the MISORIENTATION between
    # what was there and the HT orientation we wanted. Now though, it is the
    # likelyhood that a given ORIENTATION is part of a HT Parent and/or Twin
    likelyhoods = eval(parent_twin_odf,oris)##ok<EV2IN> 
    likelyhoods(likelyhoods <=0) = 0
    OP_wts = (likelyhoods*options.RGC_post_pre_m) + options.RGC_post_pre_b

    # plotting for help
    #figure()
    #plot(parent_twin_odf)
    ##figure()
    ##plot(Rough_Guess,OP_wts)
    #figure()
    #l = pruned_neigh_list(:,1)
    #r = pruned_neigh_list(:,2)
    #scatter(-Rough_Guess(L).y -Rough_Guess(R).y, Rough_Guess(L).x +Rough_Guess(R).x,1, pruned_IP_wts)


    # make a digraph with n+2 nodes (1 per voxel,plus source and sink)
    # NOTE: source has ID n+1, sink has ID n+2
    FP_digraph = digraph
    FP_digraph = addnode(FP_digraph,N+2)

    # add source-to-voxel weights (equal to likelyhood that a given voxel's
    # orientation is part of the grain of the suggested orientation)
    FP_digraph = addedge(FP_digraph, N+1, 1:N, OP_wts)
    # add voxel-to-sink weights (PRECISION CUT: WE ARE NOW WEIGHTING BY
    # INVERSE OF SOURCE-TO-VOXEL WEIGHTS)
    FP_digraph = addedge(FP_digraph, 1:N, N+2, 4./OP_wts)

    # Add in-plane (voxel to voxel) connections)
    FP_digraph = addedge(FP_digraph,L,R,pruned_IP_wts)
    FP_digraph = addedge(FP_digraph,R,L,pruned_IP_wts)


    # Perform graph cut
    [~,~,cs,~]=maxflow(FP_digraph,N+1,N+2)
    cs(cs>length(Rough_Guess)) = []
    proposed_grain = Rough_Guess(cs)
    # Code for debugging to show the cut out area. NOTE: this is not a grain,
    # Its just a region of the scan that likely has at least one complete grain
    # in it
    ##figure()
    ##plot(proposed_grain,proposed_grain.orientations)
    # Here is some extra troubleshooting code for seeing the IP weights, which
    # are what this algorithm is trying to cut along (blue = weak,
    # yellow=strong)
    #figure()
    #l = pruned_neigh_list(:,1)
    #r = pruned_neigh_list(:,2)
    #scatter(-Active_ebsd(l).y -Active_ebsd(r).y, Active_ebsd(l).x +Active_ebsd(r).x,1, IP_wts)

    return proposed_grain, PT_oris


def Seperate_Twins(proposed_grain, neigh_list, IP_wts, PT_oris,options)
    # Given we have a for sure parent grain, check to see if parts would make
    # more sense as a twin or as part of the parent.

    # do some setup stuff
    [pruned_neigh_list,pruned_IP_wts] = prune_IP_graph_connections(proposed_grain.id,neigh_list,IP_wts)
    N = size(proposed_grain,1) # number of voxels
    L = pruned_neigh_list(:,1) # left side of neigh_list connections
    R = pruned_neigh_list(:,2)# right side of neigh_list connections
    OR = proposed_grain.opt.OR
    HT_CS = proposed_grain.CS
    [R2T,~] = calc_R2T(OR,proposed_grain.CSList(3),proposed_grain.CSList(2))
    psi = proposed_grain.opt.psi
    oris = proposed_grain.orientations

    # We already got the most likely HT Parent grain. NOTE: We COULD
    # recalculate the most likely grain here (as was done in the original
    # code), but I am fairly sure it is mostly redundant to do so.

    Parent_and_Twin_IDs = zeros(size(PT_oris,1),size(oris,1))

    # Find the likelyhood that each pixel is part of the Parent grain. this
    # will become the background weighting for the 4 twin cuts.
    Parent_variants = rotation(symmetrise(PT_oris(1)))*R2T
    Parent_odf = calcDensity(Parent_variants,'kernel',psi)
    Parent_odf.CS = HT_CS
    bg_likelyhoods = eval(Parent_odf,oris)##ok<EV2IN> 
    bg_likelyhoods(bg_likelyhoods <=0) = 0
    mx = (bg_likelyhoods*options.RGC_post_pre_m)
    b =  options.RGC_post_pre_b
    OP_Parent_wts = mx + b
    clear Parent_variants Parent_odf bg_likelyhoods mx b

    for i = 2:size(PT_oris,1)

        # find the likelyhood that each pixel is part of grain i
        Twin_variants = rotation(symmetrise(PT_oris(i)))*R2T
        Twin_odf = calcDensity(Twin_variants,'kernel',psi)
        Twin_odf.CS = HT_CS
        fg_likelyhoods = eval(Twin_odf,oris)##ok<EV2IN> 
        fg_likelyhoods(fg_likelyhoods<0) = 0
        mx = (fg_likelyhoods*options.RGC_post_pre_m)
        b =  options.RGC_post_pre_b
        OP_twin_wts = mx + b
        clear Twin_variants Twin_odf fg_likelyhoods mx b
        
        # Set the weights of already assigned poitns to 1e-10, making already
        # assigned grains nearly impossible to cut (should maybe be 0?)
        assigned =(1:N).*(sum(Parent_and_Twin_IDs,1)>0)
        OP_twin_wts(ismember((1:N),assigned)>0) = 1e-10

        # make a digraph with n+2 nodes (1 per voxel, plus source and sink)
        # NOTE: source has ID n+1, sink has ID n+2
        FP_digraph = digraph
        FP_digraph = addnode(FP_digraph,N+2)
        # add source-to-voxel weights (likelyhood pixel is twin i)
        FP_digraph = addedge(FP_digraph, N+1, 1:N, OP_twin_wts)
        # add voxel-to-sink weights (likelyhood pixel is part of the parent)
        FP_digraph = addedge(FP_digraph, 1:N, N+2, OP_Parent_wts)
        # Add in-plane (voxel to voxel) connections)
        FP_digraph = addedge(FP_digraph,L,R,pruned_IP_wts)
        FP_digraph = addedge(FP_digraph,R,L,pruned_IP_wts)
        # Perform graph cut
        [~,~,cs,~]=maxflow(FP_digraph,N+1,N+2)
        cs(cs>length(proposed_grain)) = []
        # save out the mask of the cut
    #    yes_mask = [1:N].*ismember([1:N],cs)
        Parent_and_Twin_IDs(i,1:end) = ismember((1:N),cs)
        # find the already assigned orientations, set their in-plane weights to
        # zero so they won't get pulled out again.
        assigned =(1:N).*(sum(Parent_and_Twin_IDs,1)>0)
        neigh_mask = ((ismember(L,assigned) == 0) +(ismember(R,assigned) == 0))>0
        pruned_IP_wts = pruned_IP_wts.*neigh_mask    

        # plotting for help
        #figure()
        #plot(proposed_grain,OP_wts)
        #figure()
        #l = L(L>0)
        #r = L(R>0)
        #scatter(-proposed_grain(l).y -proposed_grain(r).y, ...
        #    proposed_grain(l).x +proposed_grain(r).x,1, pruned_IP_wts)
        ###figure()
        ###plot(proposed_grain,Parent_and_Twin_IDs(i,1:end))
    # get a map of twice-twinned (should never happen) and never twinned pixels
    overlap_mask = sum(Parent_and_Twin_IDs,1) <= 1
    parent_mask = sum(Parent_and_Twin_IDs,1) ~= 1
    # Twice twinned pixels get switched to never twinned
    Parent_and_Twin_IDs = Parent_and_Twin_IDs.*overlap_mask
    # assign never twinned pixels to parent
    Parent_and_Twin_IDs(1,:) = parent_mask
    # and add the IDs back in (maybe make this step unnecessary in the future?)
    Parent_and_Twin_IDs = Parent_and_Twin_IDs.*(1:N)

    return Parent_and_Twin_IDs

def prune_IP_graph_connections(ids,neigh_list,IP_wts)
    # take the global neighbor list and in plane weights, prune out the
    # connections that don't connect voxels in the ids list, and renumber them
    # appropriately so the digraph function doesn't make orphan nodes
    ids = sort(unique(ids))
    prune_mask = ismember(neigh_list(:,1),ids).*ismember(neigh_list(:,2),ids)

    # prune
    pruned_IP_wts = IP_wts(prune_mask == 1)
    pruned_neigh_list = neigh_list(prune_mask == 1,:)


    # create translater array
    translator = zeros(max(ids),1)
    for i = 1:length(ids)
        translator(ids(i))=i
    

    #remap
    l = translator(pruned_neigh_list(:,1))
    r = translator(pruned_neigh_list(:,2))

    remapped_neigh_list = [l r]

    return remapped_neigh_list,pruned_IP_wts



def get_ip_weights(orig_ebsd):#(neigh_list,ebsd,options):
    # use the misorientation between neighbor pairs to determine a "weight" to
    # give to that connection in terms of network flow capacity.

    # get misorientation angle between pairs
    [~,id_Dl] = ismember(neigh_list(:,1),ebsd.id)
    [~,id_Dr] = ismember(neigh_list(:,2),ebsd.id)
    o_Dl = ebsd(id_Dl).orientations
    o_Dr = ebsd(id_Dr).orientations
    #o_Dl = ebsd(neigh_list(:,1)).orientations
    #o_Dr = ebsd(neigh_list(:,2)).orientations
    Mori = inv(o_Dl).*(o_Dr)

    # Find likelyhoods for those misorientation angles to occur in the expected
    # Martensite structure
    LT_MDF_vals=alt_eval(ebsd.opt.LT_MDF,Mori)
    LT_MDF_vals(LT_MDF_vals<0)=0

    # Alter their weights using a y=mx+b style linear equation. For anyone
    # reading this and confused, we are altering the "strength" of the
    # pixel-to-pixel (ie, in-plane) connections. increasing the value of b
    # increases the cost of making any in-plane cuts (ie, will favor a shorter
    # overall length of grain boundaries). increasing m will increase the
    # cost of making a cut through a likely pair compared to an unlikely pair.
    m = options.RGC_in_plane_m
    b = options.RGC_in_plane_b
    ip_weights = LT_MDF_vals*m +b
    #look up orientation of IDs in neighborhood list, find the angle between
    # them, look up the likelyhood for that weight in the misorientation
    # distribution function, and Finall

    return ip_weights

def get_Out_of_plane_weights(oris,guess_ori,MDF,options):
    # given a list of orientations, guess ori, and MODF, use eval to find the
    # likelyhood that each orientation came from the guess orientation, then
    # weight those values according to the OP weighting and scaling factors

    # get misorientation angle
    mori = inv(guess_ori).*oris
    # query MDF for likelyhood of those misorientation angles
    likelyhoods = alt_eval(MDF,mori)
    likelyhoods(likelyhoods <=0) = 0
    # apply a y = mx+b style rescaling factor. Increasing m makes close
    # likelyhoods easier to include and unlikely grains harder. increasing b
    # just makes ALL cuts equally harder (only really matters for precision
    # cut, where some of the weights are done as 1/OP)
    OP_wts = (likelyhoods*options.RGC_post_pre_m)+options.RGC_post_pre_b

    return OP_wts

def prune_orphaned_pixels(Active_Ebsd,neigh_list):
    # filter out pixels without any neighbors

    a = neigh_list(ismember(neigh_list(1:end,1),Active_Ebsd.id),2)
    b = neigh_list(ismember(neigh_list(2:end,1),Active_Ebsd.id),1)
    ab = unique([a b])
    orphan_IDs = Active_Ebsd.id(ismember(Active_Ebsd.id,ab) == 0)
    
    # BIG NOTE HERE: FOR SOME REASON, the original code uses not just
    # neighbors, but neighbors of neighbors in their code (IE,1st through
    # 3rd 2D Von Neumann neighborhoods) to calculate the adjacency matrix. 
    # This SEEMS wrong, as it isnt a typical adjacency array A, but the 
    # equivilant of A +(A*A). It would also slow down the graph cut by a 
    # factor of 4. HOWEVER, for some reason it works, so I'm keeping as an 
    # option.To switch to first (IE, only 1st Von Neumann
    # neighborhood) change options.degree_of_connections_for_neighborhood
    # from 2 to 1.

    return orphan_IDs






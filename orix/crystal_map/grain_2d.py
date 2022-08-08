# -*- coding: utf-8 -*-

"""
Author: Tyler Martin
Github: martint98
Date: 5/16/2022
"""
import pickle
import time
from tqdm import tqdm
import numpy as np
from scipy import sparse
from scipy.io import loadmat
import networkx as nx
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import orix.io
from orix.crystal_map import CrystalMap, utilities


class grain2d:
    def __init__(self, ebsd, V, F, I_DG, I_FD, A_Db):
        self.ebsd = ebsd        # EBSD data set
        self.scanunit = ebsd.scan_unit
        self.V = V              # list of vertices
        self.F = F              # list of edges
        self.I_DG = I_DG        # incidence matrix - ebsd cells x grains
        self.I_FD = I_FD        # incidence matrix - edges x ebsd cells
        self.A_Db = A_Db        # adjacency matrix of cells

    @property
    def phaseID(self):
        return self.phaseID

    @phaseID.setter
    def phaseID(self, ebsd, I_DG):
        # TODO: Check difference between xmap.phase_id and ebsd.phaseID
        self.phaseID = np.full(max(np.atleast_2d(I_DG).T.conj() * sparse.spdiags(ebsd.phase_id, 0, len(ebsd.phase_id), len(ebsd.phase_id)), [], 2))

    @phaseID.deleter
    def phaseID(self):
        del self.phaseID

    @property
    def CSlist(self):
        return self.CSlist

    @CSlist.setter
    def CSlist(self, ebsd):
        self.CSlist = ebsd.phases.space_groups

    @property
    def id(self):
        id = []      # id of each grain
        return self.id

    @id.setter
    def id(self):
        self.id = np.atleast_2d(np.arange(start=0, stop=len(self.phaseID))).T

    @property
    def poly(self):
        poly = {}    # cell list of polygons forming the grains
        inclusionId = []  # number of elements in poly that model inclusions
        return self.poly

    @property
    def V(self, grains):
        V = grains.boundary.V               # vertices with x,y coordinates
        x = grains.boundary.x               # x coordinates of the vertices of the grains
        y = grains.boundary.y               # y coordinates of the vertices of the grains
        # idV              # active vertices
        # isCell = cellfun('isclass',grains.poly,'cell');
        # polygons = grains.poly;
        # polygons(isCell) = cellfun(@(x) [x{:}] ,grains.poly(isCell),'UniformOutput',false);
        # idV = unique([polygons{:}]);
        qAdded = 0       # this is only for returning to calcGrains to avoid multiple outputs
        return self.V

    @property
    def boundary(self):
        boundary = grainBoundary  # boundary of the grains
        return self.boundary

    @boundary.setter
    def boundary(self):
        self.boundary = grainBoundary(V, F, I_FDext, ebsd, self.phase_id)

    @property
    def innerBoundary(self):
        innerBoundary = grainBoundary  # inner grain boundary
        return self.innerBoundary

    @innerBoundary.setter
    def innerBoundary(self):
        self.innerBoundary = grainBoundary(V, F, I_FDint, ebsd, self.phase_id)

    @property
    def triplePoints(self):
        triplePoints = grains.boundary.triplePoints     # triple points
        return self.triplePoints

    @triplePoints.setter
    def triplePoints(self):
        grains.boundary.triplePoints = tP

    @property
    def grainSize(self):
        grainSize = []  # number of measurements per grain
        return self.grainSize

    @grainSize.setter
    def grainSize(self, I_DG):
        self.grainSize = np.atleast_2d(np.full(sum(I_DG, 1))).T

    @property
    def GOS(self):
        GOS              # intragranular average misorientation angle
        # function gos = get.GOS(grains)
#       gos = grains.prop.GOS;
        return self.GOS

    @property
    def meanOrientation(self):
        meanOrientation  # mean orientation
         # function ori = get.meanOrientation(grains)
        #       if isempty(grains)
        #         ori = orientation;
        #       else
        #         ori = orientation(grains.prop.meanRotation,grains.CS);
        #
        #         % set not indexed orientations to nan
        #         if ~all(grains.isIndexed), ori(~grains.isIndexed) = NaN;
        return self.meanOrientation

    # @meanOrientation.setter
    # def meanOrientation(self):
        # function grains = set.meanOrientation(grains,ori)
        #       if ~isempty(grains)
        #
        #         if isnumeric(ori) && all(isnan(ori(:)))
        #           grains.prop.meanRotation = rotation.nan(size(grains.prop.meanRotation));
        #         else
        #           % update rotation
        #           grains.prop.meanRotation = rotation(ori);
        #
        #           % update phase
        #           grains.CS = ori.CS;

        # @classmethod
        # def update(self, grains):
        #     self.grains.boundary = grains.boundary.update(grains)
        #     #     function grains = update(grains)
        #     #       grains.boundary = grains.boundary.update(grains);
        #     #       grains.innerBoundary = grains.innerBoundary.update(grains);
        #     #       grains.triplePoints = grains.triplePoints.update(grains);
        #     #
        #     #     end


def main():
    X = np.loadtxt(
        'C:\\git\\Mart2Aust_Hackathon\\spatial_decomposition_test_folder\\spatialDecomposition_input_X.csv',
        delimiter=',', dtype=float)
    uc = np.loadtxt(
        'C:\\git\\Mart2Aust_Hackathon\\spatial_decomposition_test_folder\\calcUnitCell_output_unitCell.csv',
        delimiter=',', dtype=float)
    # X = np.loadtxt(
    #     'C:\\PyRepo\\Hackathon\\Mart2Aust_Hackathon\\spatial_decomposition_test_folder\\'
    #     'spatialDecomposition_input_X.csv', delimiter=',', dtype=float)
    # uc = np.loadtxt(
    #     'C:\\PyRepo\\Hackathon\\Mart2Aust_Hackathon\\spatial_decomposition_test_folder\\'
    #     'calcUnitCell_output_unitCell.csv', delimiter=',', dtype=float)
    # ebsd_path = "C:\\PyRepo\\measureGrainSize\\GS_Meas\\myEBSD_high_res_1.mat"
    # ebsd = loadmat(ebsd_path)
    # ebsd_x_path = "C:\\PyRepo\\measureGrainSize\\GS_Meas\\myEBSD_high_res_1_x.mat"
    # ebsd_y_path = "C:\\PyRepo\\measureGrainSize\\GS_Meas\\myEBSD_high_res_1_y.mat"
    # unitCell = loadmat("C:\\PyRepo\\measureGrainSize\\GS_Meas\\myEBSD_high_res_1_unit_cell.mat")
    # unitCell = unitCell['myUnitCell']
    # ebsd_x = loadmat(ebsd_x_path)  # Import raw ebsd.x from MTEX as .mat file
    # ebsd_y = loadmat(ebsd_y_path)  # Import raw ebsd.y from MTEX as .mat file
    V, F, I_FD = utilities.spatial_decomposition(X, unit_cell=uc)

    print(f"V = {V}")
    # print(type(V))
    print(f"shape V = {np.shape(V)}")
    print(f"F = {F}")
    # print(type(F))
    print(f"shape F = {np.shape(F)}")
    print(f"I_FD = {I_FD}")
    # print(type(I_FD))
    print(f"shape I_FD = {np.shape(I_FD)}")
    face_one = F[0]
    print(f"face_one = {face_one}")
    face_one_vert_one_ind = face_one[0]
    print(f"face_one_vert_one_ind = {face_one_vert_one_ind}")
    face_one_vert_one_coords = V[face_one[0]]
    print(f"face_one_vert_one_coords = {face_one_vert_one_coords}")
    face_one_vert_two_ind = face_one[1]
    print(f"face_one_vert_two_ind = {face_one_vert_two_ind}")
    face_one_vert_two_coords = V[face_one[1]]
    print(f"face_one_vert_two_coords = {face_one_vert_two_coords}")

    face_1_coords = V[F[0]]
    print(f"face_1_coords = {face_1_coords}")
    faces = V[F[:]]
    print(f"faces = {faces}")
    print(f"shape faces = {np.shape(faces)}")
    faces_x_coords = faces[:, :, 0]
    print(f"faces_x_coords = {faces_x_coords}")
    print(f"shape faces_x_coords = {np.shape(faces_x_coords)}")
    faces_y_coords = faces[:, :, 1]
    print(f"faces_y_coords = {faces_y_coords}")
    print(f"shape faces_y_coords = {np.shape(faces_y_coords)}")

    print(f"face_1_x_coords = {faces_x_coords[0]}")

    plt.style.use('dark_background')
    plt.rc('axes', axisbelow=True)

    plt.figure()
    start_time = time.time()
    plt.plot(faces_x_coords[:100000], faces_y_coords[:100000])
    # plt.scatter(face_x_coords, face_y_coords)
    end_time = time.time()
    time_to_plot = end_time - start_time
    plt.title(f"Line Plot of Face Endpoints ({round(time_to_plot, 3)}s)")
    plt.savefig(f"C:/git/Mart2Aust_Hackathon/orix/tests/crystal_map/Line Plot Test.tiff", dpi=4000)
    plt.show()

    # start_time = time.time()
    # plt.figure()
    # plt.scatter(V[:, 0], V[:, 1], c='red', s=0.0001)
    # plt.ylabel(r'microns?, $[/micron/m]$', fontsize=18)
    # plt.xlabel('microns', fontsize=18)
    # plt.ylim([0,100])
    # plt.yticks(fontsize=16, rotation=0)
    # plt.xticks(fontsize=16, rotation=0)
    # plt.grid()
    # plt.box(on=False)
    # plt.tight_layout()
    # plt.savefig(f"C:/git/Mart2Aust_Hackathon/orix/tests/crystal_map/test1.tiff", dpi=4000)
    # end_time = time.time()
    # plt.show()
    # print(f"Time to plot vertices = {end_time - start_time}")
    #
    #
    # scaled_F = F/1000
    # start_time = time.time()
    # plt.figure()
    # plt.scatter(scaled_F[:, 0], scaled_F[:, 1], c="red", s=0.0001)
    # plt.ylabel(r'microns?, $[/micron/m]$', fontsize=18)
    # plt.xlabel('microns', fontsize=18)
    # # plt.ylim([0,100])
    # # plt.yticks(fontsize=16, rotation=0)
    # # plt.xticks(fontsize=16, rotation=0)
    # # plt.grid()
    # # plt.box(on=False)
    # # plt.tight_layout()
    # # plt.savefig(f"C:/git/Mart2Aust_Hackathon/orix/tests/crystal_map/test2.tiff", dpi=4000)
    # end_time = time.time()
    # plt.show()
    # # print(f"Time to plot faces = {end_time - start_time}")

    # # Testing
    # G = nx.Graph()  # Create an empty graph with no nodes and no edges.
    # # node_number = np.arange(start=0, stop=(len(V)))
    # node_coords =
    # # node = np.arange(start=0, stop=(len(V))/100)
    # node = [1, 2, 3]
    # F = [[1, 2], [2, 3], [3, 1]]
    # G.add_nodes_from(node)
    # nx.set_node_attributes(G, verts, 'coord')
    # G.add_edges_from(F)
    # # print(f"g.number_of_nodes() = {G.number_of_nodes()}")
    # # print(f"g.number_of_edges() = {G.number_of_edges()}")
    # nx.draw(G, node_size=10, node_color='red')
    # plt.show()

    # # Protype
    # node_number = np.arange(start=0, stop=len(V))
    # G = nx.Graph()  # Create an empty graph with no nodes and no edges.
    # end_ind = int(len(V)/8)
    # for node in tqdm(node_number[0:end_ind], desc="Plotting nodes"):
    #     G.add_node(node)
    # # G.add_edges_from(F)
    # # incidence_cell_face = I_FD[0]
    # # print(len(incidence_cell_face[0]))
    # # for node in V[0:1000]:
    # #     G.add_nodes_from(node, c='red', s=0.001)
    # # G.add_edges_from(F[0:1000], c='blue', s=0.001)
    # # print(f"g.number_of_nodes() = {G.number_of_nodes()}")
    # # print(f"g.number_of_edges() = {G.number_of_edges()}")
    # print(f"Added {end_ind} nodes!")
    # nx.draw(G)
    # print(f"Drawing done")
    # plt.savefig(f"C:/git/Mart2Aust_Hackathon/orix/tests/crystal_map/test4.tiff", dpi=1000)
    # print(f"Saving done")
    # plt.show()

    # node_number = np.arange(start=0, stop=len(V))
    # node_vertices = V
    #
    # start_time = time.time()
    # plt.figure()
    # fig, ax = plt.subplots()
    # # ax.scatter(V[:, 0], V[:, 1], color="deepskyblue", label="V'")
    # # ax.scatter(scaled_F[:, 0], scaled_F[:, 1], color="darkorange", s=0.01, label="F")
    # plt.xlabel('microns', fontsize=18)
    # plt.ylabel('microns [um]', fontsize=18)
    # # verts = mpatches.Patch(color='deepskyblue', label='Vertices')
    # # faces = mpatches.Patch(color='darkorange', label='Faces')
    # # legend = ax.legend(handles=[verts, faces])
    # # frame = legend.get_frame()
    # # frame.set_linewidth(0)
    # # frame.set_alpha(1)
    # plt.yticks(fontsize=16, rotation=0)
    # plt.xticks(fontsize=16, rotation=0)
    # plt.grid()
    # plt.box(on=False)
    # plt.tight_layout()
    # plt.savefig(f"C:/git/Mart2Aust_Hackathon/orix/tests/crystal_map/test3.tiff", dpi=3000)
    # end_time = time.time()
    # plt.show()
    # print(f"Time to plot verts and faces = {end_time - start_time}")

    # # V, F, I_FD = utilities.spatial_decomposition(np.array([ebsd_x, ebsd_y]), unitCell)
    # # print(f"V = {V}\nF = {F}\nI_FD = {I_FD}")
    # mySteel = orix.io.loadang('C:\\git\\Mart2Aust_Hackathon\\Data\\steel_ebsd.ang')
    # # mySteel = orix.io.loadang('C:\\PyRepo\\Hackathon\\Mart2Aust_Hackathon\\Data\\steel_ebsd.ang')
    # ebsd = CrystalMap(mySteel)
    # # myGrain = grain2d(ebsd, V, F, I_DG, I_FD, A_Db)


if __name__ == '__main__':
    main()

###### MTEX grain2d structure code below
# classdef grain2d < phaseList & dynProp
#   % class representing two dimensional grains
#   %
#   % Syntax
#   %   grains = grain2d(ebsd,V,F,I_DG,I_FD,A_Db)
#   %
#   % Input
#   %   ebsd - EBSD data set
#   %   V    - list of vertices
#   %   F    - list of edges
#   %   I_DG - incidence matrix - ebsd cells x grains
#   %   I_FD - incidence matrix - edges x ebsd cells
#   %   A_Db - adjacense matrix of cells
#   %
#   % Class Properties
#   %  phaseId - phase identifier of each grain
#   %  id            - id of each grain
#   %  poly          - cell list of the vertex ids of each grain (index to V)
#   %  V             - list of verticies (x,y coordinates)
#   %  boundary      - @grainBoundary
#   %  innerBoundary - @grainBoundary
#   %  triplePoints  - @triplePoints
#   %  grainSize     - number if pixels belonging to the grain
#   %  GOS           - grain orientation spread
#   %  meanOrientation - average grain orientation (<GrainOrientationParameters.html only single phase>)
#   %
#   % See also
#   % GrainReconstruction GrainSpatialPlots SelectingGrains ShapeParameter
#
#   % properties with as many rows as data
#   properties
#     poly={}    % cell list of polygons forming the grains
#     id=[]      % id of each grain
#     grainSize = [] % number of measurements per grain
#   end
#
#   properties (Hidden = true)
#     inclusionId = []; % number of elements in poly that model inclusions
#     qAdded      = 0;  % this is only for returning to calcGrains to avoid multiple outputs
#   end
#
#   % general properties
#   properties
#     boundary = grainBoundary % boundary of the grains
#     innerBoundary = grainBoundary % inner grain boundary
#   end
#
#   properties (Dependent = true)
#     meanOrientation  % mean orientation
#     V                % vertices with x,y coordinates
#     scanUnit         % unit of the vertice coordinates
#     GOS              % intragranular average misorientation angle
#     x                % x coordinates of the vertices of the grains
#     y                % y coordinates of the vertices of the grains
#     triplePoints     % triple points
#   end
#
#   properties (Dependent = true, Access = protected)
#     idV % active vertices
#   end
#
#   methods
#     function grains = grain2d(ebsd,V,F,I_DG,I_FD,A_Db,varargin)
#       % constructor
#
#       if nargin == 0, return;end

#       % compute phaseId's
#       grains.phaseId = full(max(I_DG' * ...
#         spdiags(ebsd.phaseId,0,numel(ebsd.phaseId),numel(ebsd.phaseId)),[],2));
#       grains.phaseId(grains.phaseId==0) = 1;
#       grains.CSList = ebsd.CSList;
#       grains.phaseMap = ebsd.phaseMap;

        # TODO: Do something with this once Orix has equivalent
#       % split face x cell incidence matrix into
#       % I_FDext - faces x cells external grain boundaries
#       % I_FDint - faces x cells internal grain boundaries
#       [I_FDext,I_FDint] = calcBoundary;

        # TODO: May not be necessary
#       % remove empty lines from I_FD, F, and V
#       isBoundary = full(any(I_FDext,2) | any(I_FDint,2));
#       F = F(isBoundary,:);
#       I_FDext = I_FDext.'; I_FDext = I_FDext(:,isBoundary).';
#       I_FDint = I_FDint.'; I_FDint = I_FDint(:,isBoundary).';
#
        # TODO: May not be necessary
#       % remove vertices that are not needed anymore
#       [inUse,~,F] = unique(F);
#       V = V(inUse,:);
#       F = reshape(F,[],2);
#
        # TODO: May not be necessary
#       % detect quadruple points
#       if check_option(varargin,'removeQuadruplePoints')
#         quadPoints = find(accumarray(reshape(F(full(any(I_FDext,2)),:),[],1),1) == 4);
#       else
#         quadPoints = [];
#       end
#
        # TODO: If block may not be necessary
#       if ~isempty(quadPoints)
#
#         % find the 4 edges connected to the quadpoints
#         I_FV = sparse(repmat((1:size(F,1)).',1,2),F,ones(size(F)));
#
#         quadPoints = find(sum(I_FV) == 4).';
#         [iqF,~] = find(I_FV(:,quadPoints));
#
#         % this is a length(quadPoints x 4 list of edges
#         iqF = reshape(iqF,4,length(quadPoints)).';
#
#         % find the 4 vertices adfacent to each quadruple point
#         qV = [F(iqF.',1).';F(iqF.',2).'];
#         qV = qV(qV ~= reshape(repmat(quadPoints.',8,1),2,[]));
#         qV = reshape(qV,4,[]).';
#
#         % compute angle with respect to quadruple point
#         qOmega = reshape(atan2(V(qV,1) - V(repmat(quadPoints,1,4),1),...
#           V(qV,2) - V(repmat(quadPoints,1,4),2)),[],4);
#
#         % sort the angles
#         [~,qOrder] = sort(qOmega,2);
#
#         % find common pixels for pairs of edges - first we try 1/4 and 2/3
#         s = size(iqF);
#         orderSub = @(i) sub2ind(s,(1:s(1)).',qOrder(:,i));
#
#         iqD = I_FDext(iqF(orderSub(1)),:) .* I_FDext(iqF(orderSub(4)),:) + ...
#           I_FDext(iqF(orderSub(2)),:) .* I_FDext(iqF(orderSub(3)),:);
#
#         % if not both have one common pixel
#         switchOrder = full(sum(iqD,2))~= 2;
#
#         % switch to 3/4 and 1/2
#         qOrder(switchOrder,:) = qOrder(switchOrder,[4 1 2 3]);
#         orderSub = @(i) sub2ind(s,(1:s(1)).',qOrder(:,i));
#
#         iqD = I_FDext(iqF(orderSub(1)),:) .* I_FDext(iqF(orderSub(4)),:) + ...
#           I_FDext(iqF(orderSub(2)),:) .* I_FDext(iqF(orderSub(3)),:);
#
#         % some we will not be able to remove
#         ignore = full(sum(iqD,2)) ~= 2;
#         iqD(ignore,:) = [];
#         quadPoints(ignore) = [];
#         iqF(ignore,:) = [];
#         qV(ignore,:) = [];
#         qOrder(ignore,:) = [];
#         s = size(iqF);
#         orderSub = @(i) sub2ind(s,(1:s(1)).',qOrder(:,i));
#
#         % add an additional vertex (with the same coordinates) for each quad point
#         newVid = (size(V,1) + (1:length(quadPoints))).';
#         V = [V;V(quadPoints,:)];
#
#         % include new vertex into face list, i.e. replace quadpoint -> newVid
#         Ftmp = F(iqF(orderSub(1)),:).';
#         Ftmp(Ftmp == quadPoints.') = newVid;
#         F(iqF(orderSub(1)),:) = Ftmp.';
#
#         Ftmp = F(iqF(orderSub(2)),:).';
#         Ftmp(Ftmp == quadPoints.') = newVid;
#         F(iqF(orderSub(2)),:) = Ftmp.';
#
#         %F(iqF(orderSub(1)),:) = [qV(orderSub(1)),newVid];
#         %F(iqF(orderSub(2)),:) = [newVid,qV(orderSub(2))];
#         sw = F(:,1) > F(:,2);
#         F(sw,:) = fliplr(F(sw,:));
#
#         [iqD,~] = find(iqD.'); iqD = reshape(iqD,2,[]).';
#
#         % if we have different grains - we need a new boundary
#         newBd = full(sum(I_DG(iqD(:,1),:) .* I_DG(iqD(:,2),:),2)) == 0;
#
#         % add new edges
#         F = [F; [quadPoints(newBd),newVid(newBd)]];
#         grains.qAdded = sum(newBd);
#
#         % new rows to I_FDext
#         I_FDext = [I_FDext; ...
#           sparse(repmat((1:grains.qAdded).',1,2), iqD(newBd,:), 1, ...
#           grains.qAdded,size(I_FDext,2))];
#
#         % new empty rows to I_FDint
#         I_FDint = [I_FDint; sparse(grains.qAdded,size(I_FDint,2))];
#
#       end
#

#       grains.id = (1:numel(grains.phaseId)).';
#       grains.grainSize = full(sum(I_DG,1)).';

#       grains.boundary = grainBoundary(V,F,I_FDext,ebsd,grains.phaseId);
#       grains.boundary.scanUnit = ebsd.scanUnit;
#       grains.innerBoundary = grainBoundary(V,F,I_FDint,ebsd,grains.phaseId);
#
        # TODO: May not be necessary
#       [grains.poly, grains.inclusionId]  = calcPolygons(I_FDext * I_DG,F,V);
#
#
        # TODO: May not be necessary
#       function [I_FDext,I_FDint] = calcBoundary
#         % distinguish between interior and exterior grain boundaries
#
#         % cells that have a subgrain boundary, i.e. a boundary with a cell
#         % belonging to the same grain
#         sub = ((A_Db * I_DG) & I_DG)';                 % grains x cell
#         [i,j] = find( diag(any(sub,1))*double(A_Db) ); % all adjacence to those
#         sub = any(sub(:,i) & sub(:,j),1);              % pairs in a grain
#
#         % split grain boundaries A_Db into interior and exterior
#         A_Db_int = sparse(i(sub),j(sub),1,size(I_DG,1),size(I_DG,1));
#         A_Db_ext = A_Db - A_Db_int;                    % adjacent over grain boundray
#
#         % create incidence graphs
#         I_FDbg = diag( sum(I_FD,2)==1 ) * I_FD;
#         D_Fbg  = diag(any(I_FDbg,2));
#
#         [ix,iy] = find(A_Db_ext);
#         D_Fext  = diag(sum(abs(I_FD(:,ix)) & abs(I_FD(:,iy)),2)>0);
#
#         I_FDext = (D_Fext| D_Fbg)*I_FD;
#
#         [ix,iy] = find(A_Db_int);
#         D_Fsub  = diag(sum(abs(I_FD(:,ix)) & abs(I_FD(:,iy)),2)>0);
#         I_FDint = D_Fsub*I_FD;
#
#       end
#     end
#

#     function V = get.V(grains)
#       V = grains.boundary.V;
#     end

#     function x = get.x(grains)
#       x = grains.boundary.x;
#     end

#     function y = get.y(grains)
#       y = grains.boundary.y;
#     end

#     function grains = set.V(grains,V)
#
#       grains.boundary.V = V;
#       grains.innerBoundary.V = V;
#
#     end
#
#     function idV = get.idV(grains)
#
#       isCell = cellfun('isclass',grains.poly,'cell');
#       polygons = grains.poly;
#       polygons(isCell) = cellfun(@(x) [x{:}] ,grains.poly(isCell),'UniformOutput',false);
#       idV = unique([polygons{:}]);
#
#     end
        # TODO: May not be necessary

#     function varargout = size(grains,varargin)
#       [varargout{1:nargout}] = size(grains.id,varargin{:});
#     end

#     function ori = get.meanOrientation(grains)
#       if isempty(grains)
#         ori = orientation;
#       else
#         ori = orientation(grains.prop.meanRotation,grains.CS);
#
#         % set not indexed orientations to nan
#         if ~all(grains.isIndexed), ori(~grains.isIndexed) = NaN; end
#       end
#     end

#     function grains = set.meanOrientation(grains,ori)
#
#       if ~isempty(grains)
#
#         if isnumeric(ori) && all(isnan(ori(:)))
#           grains.prop.meanRotation = rotation.nan(size(grains.prop.meanRotation));
#         else
#           % update rotation
#           grains.prop.meanRotation = rotation(ori);
#
#           % update phase
#           grains.CS = ori.CS;
#         end
#       end
#
#     end

#     function gos = get.GOS(grains)
#       gos = grains.prop.GOS;
#     end

#     function unit = get.scanUnit(grains)
#       unit = grains.boundary.scanUnit;
#     end

#     function tP = get.triplePoints(grains)
#       tP = grains.boundary.triplePoints;
#     end

#     function grains = set.triplePoints(grains,tP)
#       grains.boundary.triplePoints = tP;
#     end
########   TODO: Moving note as bookmark to translation progress (below needs translated)

#     function grains = update(grains)
#
#       grains.boundary = grains.boundary.update(grains);
#       grains.innerBoundary = grains.innerBoundary.update(grains);
#       grains.triplePoints = grains.triplePoints.update(grains);
#
#     end
#
#   end
#
# end
#

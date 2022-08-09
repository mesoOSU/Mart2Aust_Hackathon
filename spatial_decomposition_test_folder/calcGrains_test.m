clear all; close all; clc;

dataPath = '/Users/paytone/Desktop/'
% fileName = 'Slice_1.ang'
% 
% ebsd = EBSD.load([dataPath filesep fileName],'convertEuler2SpatialReferenceFrame')

ebsd = EBSD.load([dataPath filesep 'twins.ctf'])
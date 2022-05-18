load('spatialDecomp_test.mat')
X = X(:,1:2);
%function dummyCoordinates = calcBoundary(X,unitCell,varargin)
% dummy coordinates so that the voronoi-cells of X are finite

dummyCoordinates = [];

% specify a bounding polyogn
method = get_option(varargin,'boundary','hull',{'char','double'});

if ischar(method)
  
  switch lower(method)
    case 'tight'
      x = X(:,1);  y = X(:,2);      

      % the value 0.95 adjusts the boundary a little bit towards the convex hull
      delta = get_option(varargin,'tight',0.95,'double');
      k = boundary(x,y,delta);
      
      % erase all linear dependend points
      angle = atan2( x(k(1:end-1))-x(k(2:end)),...
        y(k(1:end-1))-y(k(2:end)) );      
      k = k([true; abs(diff(angle))>eps; true]);
      
      boundingX = X(k,:);

    case {'hull','convexhull'}
      x = X(:,1);  y = X(:,2);
      
      assignin('base', 'X_chull_check', [x,y])
      k = convhull(x,y, 'simplify', true);
      
      % erase all linear dependend points
      angle = atan2( x(k(1:end-1))-x(k(2:end)),...
        y(k(1:end-1))-y(k(2:end)) );      
      k = k([true; abs(diff(angle))>eps; true]);
            
      boundingX = X(k,:);
      
    case 'cube'
      
      % set up a rectangular box
      envelopeX = [min(X); max(X)];
      boundingX = [ ...
        envelopeX(1),envelopeX(3);
        envelopeX(2),envelopeX(3);
        envelopeX(2),envelopeX(4);
        envelopeX(1),envelopeX(4);
        envelopeX(1),envelopeX(3) ];
      
    otherwise     
      
      error('uknown boundary type. Available options are ''convexhull'' ''tight '' and ''cube''.');   
      
  end
  
elseif isa(method,'double')
  
  boundingX = method;
  
end


radius = mean(sqrt(sum(unitCell.^2,2)));
edgeLength = sqrt(sum(diff(boundingX).^2,2));

% fill each line segment with nodes every 20 points (in average)
nto = fix((edgeLength>0)*4); fix(edgeLength*(2*radius));

assignin('base','nto_test',nto)
cs = cumsum([1; 1 + nto]);
assignin('base','cs_test',cs)
assignin('base','boundX_ini',boundingX)
boundingX(cs,:) = boundingX;
assignin('base','boundX_fin',boundingX)

% interpolation
for k=1:numel(nto)
  for dim=1:2
    boundingX(cs(k):cs(k+1),dim) = ...
      linspace(boundingX(cs(k),dim), boundingX(cs(k+1), dim),nto(k)+2);
  end  
end


% homogeneous coordinates
X(:,3) = 1;
boundingX(:,3) = 1;

% householder matrix
H = @(v) eye(3) - 2./(v(:)'*v(:))*(v(:)*v(:)') ;
% translation matrix
T  = @(s) [ 1 0 s(1); 0 1 s(2); 0 0 1];

% direction of the edge
edgeDirection = diff(boundingX);
assignin('base', 'edgeDir_test',edgeDirection)
edgeAngle = atan2(edgeDirection(:,2),edgeDirection(:,1));
edgeLength = sqrt(sum(edgeDirection.^2,2));

% shift the starting vertex
bX = squeeze(double(axis2quat(zvector,edgeAngle)* ...
  vector3d([0; radius; 1])));
assignin('base', 'bX_test', bX)
offsetX = bX - boundingX(1:end-1,:);

for k=1:size(boundingX,1)-1
  
  % mirror the point set X on each edge
  pX = X * -(T(offsetX(k,:)) * H(edgeDirection(k,:)) * T(offsetX(k,:)))';
  
  % distance between original and mirrowed point
  dist = sqrt(sum((X(:,1:2)-pX(:,1:2)).^2,2));
 
  intendX = 2*radius*sign(edgeDirection(k,1:2));
  
  % now try to delete unnecessary points
  m = 2;
  while true
    
    tmpX = pX(dist < m*radius,1:2);
    
    right = (bsxfun(@minus, tmpX, boundingX(k,1:2)  - intendX ) * edgeDirection(k,1:2)') < 0;
    left  = (bsxfun(@minus, tmpX, boundingX(k+1,1:2)+ intendX ) * edgeDirection(k,1:2)') > 0;
   
    tmpX = tmpX( ~(right | left) ,:);
     
    if edgeLength(k)/size(tmpX,1) < radius/3
      break;
    elseif m < 2^7
      m = m*2;
    elseif m < 2^7+100
      m = m+10; 
    else
      break;
    end
    
  end
  
  dummyCoordinates = [dummyCoordinates; tmpX];
  
end
  
dummyCoordinates = unique(dummyCoordinates,'first','rows');

% remove those points which are inside the b

id = inpolygon(dummyCoordinates(:,1),dummyCoordinates(:,2),boundingX(:,1),boundingX(:,2));

dummyCoordinates(id,:) = [];
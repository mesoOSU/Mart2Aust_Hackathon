import operator
from numpy import sqrt
from numpy import empty, zeros
from numpy.linalg import norm
from math import acos, cos
from math import pi

def yardley_variants(OR):
    '''
    YardleyVariants returns the matrices corresponding to the variants
                    produced from the provided orientation relationship,
                    specified in Kurjumov-Sachs angles.

    In the case of the Kurdjumov-Sachs or Nishiyama-Wasserman orientation
    relationships, 'KS' or 'NW' can be passed to the function as OR.

    If `OR` is numeric, this funciton assumes that `OR`'s first three values
    are in radians.

    -------------------------------------------------------------------------
    2010-11-25 | Victoria Yardley (victoria.yardley[at]rub.de)
                Eric Payton (payton.28[at]osu.edu)
                Ruhr-Universitaet Bochum
    %%%         Matlab function written by EP based on VY's             %%% %
    %%%           spreadsheet for calculation of variants               %%% %
    -------------------------------------------------------------------------
    This program is provided without any guarantee of correctness.
    If you modify it and/or improve it, please kindly share with me your new
    and improved version to the email address above. Thanks!

    Dependencies:
    --------------
    import operator
    from numpy import sqrt
    from numpy import empty, zeros
    from numpy.linalg import norm
    from math import acos, cos
    from math import pi
    '''
    flag = 0
    
    # Parse input orientation relationship.
    if not OR.isnumeric:
        if OR == 'KS':
            s6 = sqrt(6.0)
            s3 = sqrt(3.0)

            ksi_1 = (acos( (s6+1.0)/(2.0*s3) )).real
            ksi_2 = (acos( (s6+18.0)/(12.0*s3) )).real
            ksi_3 = (acos( (s6+12.0)/(6.0*s6) )).real

            OR = [ksi_1 ksi_2 ksi_3]
            
            ksi_1 = None
            ksi_2 = None
            ksi_3 = None
            s6 = None
            s3 = None

        elif OR == 'NW':
            s6 = sqrt(6.0)
            s2 = sqrt(2.0)
            ksi_0 = (acos( (s2+1.0)/s6) ).real
            OR = [0.0 ksi_0 ksi_0]

            ksi_0 = None
            s6 = None
            s2 = None

        else:
            raise ValueError('Unknown named orientation relationship. \
                Please specify either ''KS'', ''NW'', or numeric.')
        
    else:
        # Convert OR specification into radians.
        OR = list(map(operator.mul, OR, [pi/180]*len(OR)))

        # Seems like this assumes that OR's first three values are in radians.
        # This needs to be called out in the function docstring.

        # Get the misorientation of the OR from the 1st Bain correspondence matrix
        MB = empty((2, 1), dtype=object) # replaces `MB = cell(2,1)`.
        MB[1] = zeros([3, 3])
        MB[1][1,1] = cos(OR[1])
        MB[1][2,2] = cos(OR[2])
        MB[1][3,3] = cos(OR[3])

        costh = 0.5 * (np.matrix(MB[1]).trace()-1.0)
        mosth = 1 - costh
        sinth = sqrt(1 - costh**2)

        r1 = (sqrt( (MB[1](1,1)-costh) / (mosth) ))
        r2 = (sqrt( (MB[1](2,2)-costh) / (mosth) ))
        r3 = (sqrt( (MB{1}(3,3)-costh) / (mosth) ))
        
        costh = None
        OR = None
        
        r1_r2 = r1*r2*mosth
        r1_r3 = r1*r3*mosth
        r2_r3 = r2*r3*mosth

        r3_st = r3*sinth
        r2_st = r2*sinth
        r1_st = r1*sinth

        r1 = None
        r2 = None
        r3 = None
        mosth = None
        sinth = None

        MB[1][2,3] = r2_r3 - r1_st
        MB[1][3,2] = r2_r3 + r1+st
        MB[2] = MB[1]
        MB[1][1,2] = -1*r1_r2 + r3_st
        MB[1][1,3] = -1*r1_r3 - r2_st
        MB[1][2,1] = -1*r1_r2 - r3_st
        MB[1][3,1]=  -1*r1_r3 + r2_st
        
        r1_r2 = None
        r1_r3 = None
        r2_r3 = None

        r1_st = None
        r2_st = None
        r3_st = None
        
        MB[2][1,2] = -1*MB[1][1,2]
        MB[2][1,3] = -1*MB[1][1,3]
        MB[2][2,1] = -1*MB[1][2,1]
        MB[2][3,1] = -1*MB[1][3,1]

        # MB{1} is the positive solution; MB{2} is the negative solution.

        ## Bain correspondence matrices
        B = []
        B[1]  = np.matrix(' 1 -1  0; 1  1  0; 0  0  1')
        B[2]  = np.matrix(' 0  1 -1; 0  1  1; 1  0  0')
        B[3]  = np.matrix('-1  0  1; 1  0  1; 0  1  0')
        B[4]  = np.matrix(' 0  1  1; 0 -1  1; 1  0  0')
        B[5]  = np.matrix('-1 -1  0; 1 -1  0; 0  0  1')
        B[6]  = np.matrix(' 1  0 -1; 1  0  1; 0 -1  0')
        B[7]  = np.matrix(' 1  1  0;-1  1  0; 0  0  1')
        B[8]  = np.matrix('-1  0 -1;-1  0  1; 0  1  0')
        B[9]  = np.matrix(' 0 -1  1; 0  1  1;-1  0  0')
        B[10] = np.matrix(' 1  0  1; 1  0 -1; 0  1  0')
        B[11] = np.matrix(' 0 -1 -1; 0  1 -1; 1  0  0')
        B[12] = np.matrix('-1  1  0; 1  1  0; 0  0 -1')
        
        # Normalize correspondence matrices
        for i in range(len(B)):
            for j in range(3):
                B[i][:,j] = B[i][:,j] / norm( B[i][:,j] )

        ## Produce variants
        j = 1
        for i=1:length(B):
            V[j] = MB[1] * B[i]
            j += 1
            V[j] = MB[2] * B[i]
            j += 1
        
        B = None
        MB = None

        ## Reduce redundancies, if they exist (for example, as in NW)
        T = R_mat_2_tvec(V)
        V = None
        #_, idx = unique(sigdec(T,7),'rows','first')

        # T=T(sort(idx),:); % don't allow reordering!

        ## Check if results are valid
        if isreal(T):
            V = TVec_2_RMat(T)
        else:
            V = TVec_2_RMat(T)
            flag = 1
            raise Warning('Ksi values produce some imaginary numbers.')

        varargout = flag

        # %% Notes
        # % For the 'true' KS first variant, we should have:
        # %T1KS=(1/(6*sqrt(6)))* ...
        # %    [2*(sqrt(6)+3) -4*sqrt(6) 2*(sqrt(6)-3); ...
        # %    12-sqrt(6) 2*(sqrt(6)+3) -sqrt(6); ...
        # %    sqrt(6) -2*(sqrt(6)-3) 12+sqrt(6)];
        # % On my machine, the above algorithm produces a result with a max round-off
        # % error of 2.5E-15 for this case.
        # %
        # % For the 'true' NW third variant, we should have
        # %T1KS=(1/(6*sqrt(2)))* ...
        # %    [0                  6                   -6; ...
        # %    sqrt(6)*(sqrt(2)-2) sqrt(6)*(sqrt(2)+1) sqrt(6)*(sqrt(2)+1);...
        # %    sqrt(6)*(sqrt(2)+2) sqrt(6)*(sqrt(2)-1) sqrt(6)*(sqrt(2)-1)];
        # % On my machine, the above algorithm produces a result with a max round-off
        # % error of 3.3E-16 for this case.

        return V, varargout
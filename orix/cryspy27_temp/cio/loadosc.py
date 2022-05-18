# -*- coding: utf-8 -*-


#filename = '8816-1_01_center.osc'
#fnom = '8816-1.osc'
#fnom = '8816-1_0_prevScan.osc'
#fnom = '8816-1_02_nearEdge.osc'


from cryspy.ebsd import ebsd
from cryspy.rot import bunge, quat
from cryspy.xtal import orientation, interpret_point_group_name
import struct
import numpy as np

def loadosc(filename, create_ebsd_object = True, ang_output = False, \
             ang_output_filename = None):
    """ loads osc files and converts to ang format, if desired
    
    This Python code performs the OSC to ANG conversion approximately 25% faster
    than the previous script in Matlab [#TODO: add reference to matlab], with 
    some corrections in output and a potential improvement in memory usage
    for opening very large OSC files. 
    
    NOTES
    -----
    
    There are at least two Matlab versions of this floating around. One is from 
    Adam Shiveley (AFRL/RXCM) and the other is in the MTEX Matlab Toolbox. The
    MTEX version is a partially rewritten version of Shiveley's by Florian
    Bachmann. I was unable to get it to work in my testing. One problem is was that
    the buffer size was set too small.
    
    Adam Shiveley's version from 9 Feb 2012 for Matlab is still in use at AFRL as
    of 21 June 2016. My translation includes all of his original code as comments.
    All Matlab code will be indicated by a % sign at the beginning of the comment
    line.
    
    NOTES SPECIFIC TO THIS PYTHON VERSION:
    
    1. HKL_COL_4 for first phase is wrong
    2. TEM_PX_PER_MICRON not yet correctly located in data file. We put in a dummy 
             when we write to ang.
    3. Not yet tested for more than 2 phases.
    4. We don't locate the elastic constants or "categories" in the osc file. 
          ANG reading works without the elastic constants, and we put in a dummy 
          for the categories.
    5. Some items that appear as zeroes to 5 or so decimal places appear as very 
          small floats ~1E-38 when I import them.
    
    PERFORMANCE TESTS:
    A. 8234 kB OSC file: 3.75 s Python version; 5.05 s Matlab version 
        (~25% improvement)
    B. 277635 kB OSC file: 98.9 s Python version; 132.8 s Matlab version 
        (~25% improvement)
    
    #------------------------------------------------------------------------------
    ADAM SHIVELEY'S ORIGINAL HEADER:
        %function TSL_version = Decode_OSC_Header(OscFile, fileWriteID)   
        %  Decode_Osc_Header(OscFile, fileWriteID)
        % 
        %  Inputs: OscFile is a string containing the path and file name of the OSC
        %          file to decode
        %          fileWriteID is an integer resulting from fileWriteID = 
                                                       fopen('output.ang','w');
        % 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % coded by: Adam Shiveley   9 Feb 12, AFRL/RXCM
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        %The OSC file is basically structured like this:  
        %Phase 1 name
        %Phase 1 symmetry
        %Phase 1 LatticeConstants
        %Phase 1 Number of Families
        %Phase 1 hklFamilies (read in every third one)
        %Phase 1 Formula name
        %This repeats for each phase in the scan
        %sounds simple, right? Wrong the values are stored in decimal
        %because of the way I choose to read the file in.
        %This means data(10) = 70 which is actually the letter
        %F as explained in this example:
        %This is in decimal!!!!
        %Example: data(10) = 70
        %char(data(10)) = F         Look at ASCII table, Dec 70 = F
        %This is where the fun begins.  Most of the header information
        %can be extracted directly using the decimal values and running
        %the char command, however, the Euler angles must be extracted
        %and converted into columns of length 4 then typecasted to a single
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    def _unpack_from_mmap(mmap_id, location_start):
        
        mmap_id.seek(location_start)
        check = mmap_id.read_byte()
        mmap_id.seek(location_start) # go back
        counter = 0
        
        if check != '\x00':
            
            temp_variable = '\x01'
            
            while temp_variable != '\x00':
                
                temp_variable = mmap_id.read_byte()
                counter += 1  
            
            counter -= 1
            
            mmap_id.seek(location_start)
            result = mmap_id.read(counter)
    
        else:
            
            result = ""
        
        return result    
        
    
    #------------------------------------------------------------------------------   

    # So we can measure the time it takes to read the data
    import time
    t = time.clock()                
    
    """ 
    %     %Now let's get the handle to open the scan file
    %     fid=fopen(OscFile);
    % 
    %     %This is in decimal!!!!
    %     %Example: data(10) = 70
    %     %char(data(10)) = F         Look at ASCII table, Dec 70 = F
    %     data = transpose( fread( fid, '*bit8' ) );
    
    
    ------------------------------------------
    
    We have a couple options for opening the file in Python, e.g.:
    # with open(fnom, mode='rb') as file: # b is important -> binary
    #    bindat = file.read()
    
    I have chosen to proceed with memory mapping rather than reading the whole file
    into memory because it's supposed to offer benefits for large files.
    
    My translation of Adam Shiveley's original code takes advantage of this for
    some functions but memory management could still use future improvement.
    
    Shiveley reads in the entire binary file as 8 bit int and then casts the 
    integers into other data types. I am generally trying to selectively read the
    portions we need as binary and unpack them as the correct data type.
    """
    
    import mmap
    with open(filename, "r+b") as f:
        # memory-map the file, size 0 means whole file
        mm = mmap.mmap(f.fileno(), 0)
    
    """ The expected locations of header information in the data array"""
    info_location_start = 72
    operator_location_start = 1095
    sample_ID_location_start =  1350
    scan_ID_location_start = 1605
    calibration_location_start = 1860
    
    """ 
    %     %Let's locate if there is any info the user typed in
    %     %There might be an issue here depending on how long the user types the
    %     %comments.  Will need further testing to see if it breaks the hard-coded
    %     %index locations of the SampleID, Operator, and ScanID
    %     if(data(73) ~= 0)
    % 
    %        info_temp_variable = 1; 
    %        info_counter = 73; 
    % 
    %        while info_temp_variable ~= 0
    % 
    %             info_temp_variable = data(info_counter);
    %             info_counter = info_counter +1;
    % 
    %        end
    % 
    %         %Convert the data into characters
    %         info = char(data(73:info_counter-1));
    % 
    %         %Clear the vareiable to free up some memory
    %         clear info_temp
    % 
    %     else
    % 
    %         info = '';
    % 
    %     end
    % 
    %     %Now let's parse out the operator name
    %     if(data(1096) ~= 0)
    % 
    %        operator_temp_variable = 1; 
    %        operator_counter = 1096; 
    % 
    %        while operator_temp_variable ~= 0
    % 
    %             operator_temp_variable = data(operator_counter);
    %             operator_counter = operator_counter +1;
    % 
    %        end
    % 
    %         %Convert the data into characters
    %         operator = char(data(1096:operator_counter-1));
    % 
    %         %Clear the vareiable to free up some memory
    %         clear operator_temp
    %        
    %     else
    % 
    %         operator = ' ';
    % 
    %     end
    % 
    % 
    %     %Now let's parse out the Sample ID.  The index location could change if
    %     %a long info string is typed in.  Seems to not change on the test I have
    %     %ran
    %     if(data(1351) ~= 0)
    % 
    %        sample_ID_temp_variable = 1; 
    %        sample_ID_counter = 1351; 
    % 
    %        while sample_ID_temp_variable ~= 0
    % 
    %             sample_ID_temp_variable = data(sample_ID_counter);
    %             sample_ID_counter = sample_ID_counter +1;
    % 
    %        end
    % 
    %         %Convert the data into characters
    %         Sample_ID = char(data(1351:sample_ID_counter-1));
    % 
    %         %Clear the vareiable to free up some memory
    %         clear sample_ID_temp
    % 
    %        
    %     else
    % 
    %         Sample_ID = ' ';
    % 
    %     end
    % 
    %     %Now let's parse out the Scan ID.  Again, the index could change if the
    %     %info is really long
    %     if(data(1606) ~= 0)
    % 
    %        scan_ID_temp_variable = 1; 
    %        scan_ID_counter = 1606; 
    % 
    %        while scan_ID_temp_variable ~= 0
    % 
    %             scan_ID_temp_variable = data(scan_ID_counter);
    %             scan_ID_counter = scan_ID_counter +1;
    % 
    %        end
    % 
    %         %Convert the data into characters
    %         Scan_ID = char(data(1606:scan_ID_counter-1));
    % 
    %         %Clear the vareiable to free up some memory
    %         clear scan_ID_temp
    %       
    %     else
    % 
    %         Scan_ID =' ';
    % 
    %     end
    
    """
    info = _unpack_from_mmap(mm, info_location_start) # FIXME: write this info data somewhere to be used for writing ang files
    operator_name = _unpack_from_mmap(mm, operator_location_start)
    sample_ID = _unpack_from_mmap(mm, sample_ID_location_start)
    scan_ID = _unpack_from_mmap(mm, scan_ID_location_start)
    
    """
    % Now let's check the file for the TEM_PIXperUm, x-star, y-star, z-star
    % and working distance
    
    % Not sure where the TEM_PIXperUM is located because I don't have the TEM
    % version, so I'll just hard code it for now
    """
    TEM_PIXperUM = 1
    
    """
    %     %This extracts the calibration info
    %     calibration_encoded = data(1861:1876);
    % 
    %     %Now we need to reshape the calibration info into groups of 4
    %     calibration_reshaped = reshape(calibration_encoded,
                                            [4 length(calibration_encoded)/4]);
    % 
    %     %Let's typecast the data into a single representation
    %     calibration_decoded = typecast(calibration_reshaped(:), 'single');
    % 
    %     %Here's the final calibration info
    %     x_star = calibration_decoded(1);
    % 
    %     y_star = calibration_decoded(2);
    % 
    %     z_star = calibration_decoded(3);
    % 
    %     working_distance = calibration_decoded(4);
    """
    mm.seek(calibration_location_start)
    calibration_encoded = mm.read(16)
    calibration_decoded = struct.unpack('<4f', calibration_encoded)
    del calibration_encoded
    
    x_star = calibration_decoded[0]
    y_star = calibration_decoded[1]
    z_star = calibration_decoded[2]
    working_distance = calibration_decoded[3]
    del calibration_decoded
    
    """
    %     %Find the starting location for phase 1.  This never seems to change
    %     %so far it stays constant, but I would like to find a more secure way
    %     %of finding the 1st phase.  Seems like the command:
    %     %strfind(data, [-71 11 -17 -1 1]) would work...
    %     %maybe something like:
    %     start_phase_location = strfind(data, [-71 11 -17 -1 1 0 0 0]);
    %     phase_one_temp = data(start_phase_location:end);
    %     begin_phase_one = strfind(phase_one_temp, [0 0 0]);
    %     start_phase_1 = begin_phase_one(1) + 3 + start_phase_location(1); 
                                     % EJP! same as start_phase_location + 8???
    %     phase_1_data = data(start_phase_1(1):end); 
                                    % EJP! this and the previous several lines are
                                      a complicated way of getting to the end of 
                                      the -71 11 ... 0 0 0 match string!
    %     %something like that might work, but it needs further testing to verify
    %     
    %     %%%%%%%%%%%%%%%%%%%%%%%%%%
    %     %Added 3 Feb 2012
    %     %This will locate the begining of the actual phase
    %     
    %     %Alright, so it now appears that the number after -128 43 is the number
    %     %of phases in the sample. 
    %     begin_phase_1_check = strfind(phase_1_data, [-128 63]);
    %     
    %     %get the number of phases from the scan
    %     number_of_phases = phase_1_data(begin_phase_1_check(1)+2);
    %     
    %     %This will extract the begining of the acutal phase to the end
    %     phase_1_data = phase_1_data(begin_phase_1_check(1)+3:end);
    %     
    %     %So, now we need to ignore all the zeros until we hit the actual phase
    %     %name
    %     phase_1_zero_check = 0;
    %     phase_1_zero_check_counter = 1;
    %     
    %     %locate the begining of the actual phase 1 name
    %     %The way this loop is structured, you have to subtract 2 from the final
    %     %number in phase_1_zero_check_counter.  Since the loop starts at 1 and
    %     %increments after it gets the current position, this will add 2 extra
    %     %iteration counts, 1 because it starts at 1 and 1 because it will
    %     %increment after the non-zero number is found
    %     while phase_1_zero_check == 0
    %         
    %         phase_1_zero_check = phase_1_data(phase_1_zero_check_counter);
    %         phase_1_zero_check_counter = phase_1_zero_check_counter + 1;
    %         
    %     end
    %     %%%%%%%%%%%%%%%%%%%%%%%%%%%
    %     
    %    
    %     %Find the ending location for phase 1 search reigon
    %     %This is used until I write a code to begin searching
    %     %at a given offset.  Then you can just say:
    %     %end_phase_1 = strfind(data(1945), 0);
    %     %This would mean begin searching data from position 1945
    %     %until you find a "0"
    %     end_phase_1 = strfind(phase_1_data, [110 0 0]);
    % %     end_phase_1 = strfind(phase_1_data(begin_phase_1_check+3+ ...
    % %                           phase_1_zero_check_counter), 0);
    % 
    %     
    %     if length(end_phase_1) > 1
    % 
    %         end_phase_1 = end_phase_1(1); 
    % 
    %     end
    % 
    %     %Extract phase 1 from the file using the 1989 index plus the ending index
    %     %since we searched for the ending_phase in an array starting at 1945 from
    %     %the original data set.  The -1 here is because since we start at 1989
    %     %we calculated everything else according to starting at 0.
    %     phase_1_location = data(start_phase_1-1+begin_phase_1_check(1) + 3 + ...
    %                      phase_1_zero_check_counter-2:start_phase_1+end_phase_1);
    % 
    %     %new let's clear the phase_1_data array
    %     %clear phase_1_data
    % 
    %     %Find the end location of phase 1
    %     end_phase_1_temp = strfind(phase_1_location, 0);
    % 
    %     %Convert the decimal values into characters using ASCII
    %     final_phase_1 = char(phase_1_location(1:end_phase_1_temp-1));
    """
    
    # First Phase -- There must be at least one! 
    # It appears that each value is padded by a byte.
    start_phase_location = mm.find(struct.pack(\
                                             '<8B', 185, 11, 239, 255, 1, 0, 0, 0))
    
    tmpini = start_phase_location + 16
    end_phase_1 = mm[tmpini:].find(struct.pack('<3b', 110, 0, 0)) + 4
    
    tmpini = start_phase_location+16
    tmpfin = start_phase_location+16+end_phase_1-3
    temp = mm[tmpini:tmpfin].split('\x00')
    number_of_phases = struct.unpack('<b', temp[0])[0]
    final_phase_1 = temp[-2] # in case the third entry is reserved for something
                             # that is currently set to zero in my test data set
    del temp
    
    """
    %     %check the string for the sysmetry number by searching beyond
    %     %the location of the end of the phase name.  Agian we have to use the
    %     %staring index of 1989 since end_phase was found inside a subset of data
    %     %begining at 1989
    %     symmetry_temp = 0; 
    %     symmetry_counter = start_phase_1 + end_phase_1 + 1 + begin_phase_1_check
                             + 3 + phase_1_zero_check_counter-2; 
    % 
    %      while symmetry_temp == 0
    %         symmetry_counter(1)
    %         symmetry_temp = data(symmetry_counter(1));
    %         symmetry_counter = symmetry_counter +1;
    % 
    %      end
    % 
    %      %Phase symmetry in decimal form
    %      phase_1_symmetry = symmetry_temp;
    """
    # find next location that isn't a zero -- this is the start of symmetry data
    symmetry_temp = 0
    symmetry_counter = start_phase_location+16+end_phase_1-1
    while symmetry_temp == 0:
        symmetry_temp = struct.unpack('<b', mm[symmetry_counter])[0]
        symmetry_counter += 1
    
    # the last location is the symmetry in geo convention
    phase_1_symmetry = symmetry_temp
    
    """
    %     %This is used to extract the begining of the hkl data.  The first values
    %     %are the Lattice Constants followed by the number of families followed by
    %     %the actual hkl values
    %     temp_hkl_location = data(symmetry_counter+1:end);
    %     temp_hkl_location(1:10)
    %     
    %     %locate the begining of the LatticeConstants
    %     begin_lattice_constants_location = strfind(temp_hkl_location, [0 0 0]);
    % 
    %     %This is used to extract the lattice constants and the number of families
    %     %from the temp_hkl_location_array
    %     temp_lattice_constants = temp_hkl_location
                                         (1:begin_lattice_constants_location(1));
    % 
    %     %This is used to get the Number of HKL families in decimal form
    %     Number_hkl_families = temp_lattice_constants(end-1);
    % 
    %     %This is used to extract the lattice constants encoded.
    %     lattice_constants_encoded = temp_lattice_constants(3:end-2);
    % 
    %     %reshpae the lattice constants into an array 4x6 because there are 6
    %     %lattice constants
    %     reshaped_lattice_constants = reshape(lattice_constants_encoded,
                                          [4,length(lattice_constants_encoded)/4]);
    % 
    %     %final lattice constants in decimal form
    %     lattice_constants = typecast(reshaped_lattice_constants(:), 'single');
    """
    # get the beginning of the lattice constants
    begin_lattice_constants_location = mm[symmetry_counter+1:].find( \
                                                     struct.pack('<3b', 0, 0, 0))
    
    temp_lattice_constants = mm[symmetry_counter+1: \
                               symmetry_counter+begin_lattice_constants_location+1]
                               
    number_hkl_families = struct.unpack('<b',temp_lattice_constants[-1])[0]
    
    lattice_constants_encoded = temp_lattice_constants[2:-1]
    lattice_constants_phase_1 = np.array( \
                                    struct.unpack('<6f',lattice_constants_encoded))
    del temp_lattice_constants, lattice_constants_encoded
    
    ''' 
    %     %This is used to determine how many values to read from the extracted
    %     %data. The 3 and the 4 are because there are 3 hkl values and each value
    %     %is seperated by three digits before the next hkl is read in.  It might
    %     %be possible to change the way the file is read into Matlab by shifting
    %     %the bits, but since this works, I never have tried reading the file a
    %     %different way
    %     Total_number_hkl_families = double(Number_hkl_families) * 3 * 4;
    % 
    %     %This extracts the hkl families
    %     temp_hkl_families = temp_hkl_location(
                                   length(temp_lattice_constants)+3:
                                   length(temp_lattice_constants)+
                                   Total_number_hkl_families-1)
    % 
    %     %Pull every 4th element from the array
    %     temp_hkl_families = temp_hkl_families(1:4:end);
    % 
    %     %rehsape to match the *.ang output
    %     final_hkl_families = reshape(temp_hkl_families,
                                           [3 (length(temp_hkl_families)/3)])';
    
    '''
    total_number_hkl_families = number_hkl_families * 3 * 4
    startloc = symmetry_counter + begin_lattice_constants_location + 4
    temp_number_hkl_families = struct.unpack(
                                   "<{0:d}b".format(total_number_hkl_families), 
                                   mm[startloc:startloc+total_number_hkl_families])
                                   
    th = np.asanyarray(temp_number_hkl_families)
    final_hkl_families = np.reshape(np.reshape(th,
                                    [th.shape[0]/4, 4])[:,0], [th.shape[0]/4/3, 3])
    del temp_number_hkl_families, th, startloc
    
    """ 
    %     %Extract the formula name region, the 3 is used because of the
    %     %trailing 0 0 0 after the hkl families
    %     temp_formula_name = temp_hkl_location(length(temp_lattice_constants)
                                                +Total_number_hkl_families+3:end);
    % 
    %     %Search for the formula name
    %     formula_name_location = strfind(temp_formula_name, 0);
    % 
    %     %Use the first index to locate the end of the forumla name
    %     formula_name = char(temp_formula_name(1:formula_name_location(1)));
    """
    startloc = symmetry_counter + begin_lattice_constants_location + \
                          4 + total_number_hkl_families
    temp_formula_loc = mm[startloc:].find(struct.pack('<b', 0))
    formula_name = mm[startloc:startloc + temp_formula_loc]
    
    """
    %     %now let's find the last column of the hkl information
    %     %The string here is the locating string for the begining of the last col
    %     %info.  If there is more than one phase, it will return the location of
    %     %all the last column info for each phase
    %     last_hkl_col_location = strfind(temp_hkl_location, 
                          [2 0 0 0 -1 -1 -1 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] );
    %     last_hkl_col_location(1)
    """
    temp_hkl_location = mm[symmetry_counter+1:]
    last_hkl_col_location = mm[symmetry_counter+1:].find( \
                               struct.pack('<24b', 2,  0, 0, 0, -1, -1,
                                                  -1, -1, 0, 0,  0,  0,
                                                   0,  0, 0, 0,  0,  0, 
                                                   0,  0, 0, 0,  0,  0))
    
    """
    %     %Now extract the first phase last hkl col information.  This is a
    %     %little bit confusing.  The 25 is because the strfind locates the string
    %     %above, which is 16 characters long.  There are exactly 8 characters of
    %     %zeros before you reach the first index.  This totals to be 24.  
    %     %Now, the 20 is in here because you have to include the 25 chracters
    %     %described eariler, but you need to subtract 4 because you are starting
    %     %at the first hkl row so you can't count that.  This totals to be 20.
    %     temp_hkl_last_col = temp_hkl_location(
                           last_hkl_col_location(1)+24:
                           last_hkl_col_location(1)+20+
                                       (4*double(Number_hkl_families)));
    % 
    %     %Extract every 4th element and transpose the array to match the *.ang
    %     %format
    %     hkl_last_col = temp_hkl_last_col(1:4:end)';
    
    -------------------------------------------------------------------------------
        
    # The total should be 24. Otherwise we can't reshape correctly.
    """
    tmpini = symmetry_counter+1+last_hkl_col_location+24 
    tmpfin = symmetry_counter+1+last_hkl_col_location+24+4*number_hkl_families
    temp_hkl_last_col = np.asanyarray(struct.unpack( \
                               '<{0:d}b'.format(tmpfin-tmpini), mm[tmpini:tmpfin]))
    
    # Extract every 4th element
    hkl_last_col = np.reshape(temp_hkl_last_col, \
                                            [temp_hkl_last_col.shape[0]/4, 4])[:,0]
    del temp_hkl_last_col
    
    """    
    %     %Now let's find the 5th column by locating the last column
    %     %by reverse looking for a string not 1 or 0
    %     hkl_column_temp = 0; 
    %     current_hkl_index = last_hkl_col_location(1)-1
    % 
    %      while (hkl_column_temp == 0) || (hkl_column_temp == 1)
    % 
    %         hkl_column_temp = temp_hkl_location(current_hkl_index);
    %         current_hkl_index = current_hkl_index - 1;
    % 
    %      end
    %
    %     %Extract the 5th column values.  The reason for the + 2 is because
    %     %current_hkl_index has already decremented and we need to account for the
    %     %1 in locating the starting position.  The reason for the + 1 is because 
    %     %the while loop is designed to decrement current_hkl_index after it
    %     %searchs the string.  This means once the location has been found, the 
    %     %while loopwill still decrement current_hkl_index.  While works well,
    %     %there is an issue with the way Matlab is truncating the values. 
    %     %Anything below 1 seems to become 0.
    %     hkl_5_col = temp_hkl_location(current_hkl_index - 
                            (4*double(Number_hkl_families))+2:current_hkl_index+1);
    %     
    %     %reshpae the 5th hkl column
    %     hkl_5_col_unreshaped = reshape(hkl_5_col,[4,length(hkl_5_col)/4]);
    % 
    %     %The 5th column in it's decimal form
    %     hkl_5_final = typecast(hkl_5_col_unreshaped(:), 'single');
    %
    %     %Now we need to remove the Nan from the HKL families due to some
    %     %conversion errors!!
    %     Nan_index_first_phase_hkl = isnan(hkl_5_final);
    %     hkl_5_final(Nan_index_first_phase_hkl) = 0;  
    -------------------------------------------------------------------------------
    I fixed the location of the indexing and we no longer get NaN errors in my
    test datasets. The results of this column are sometimes zeros to several
    decimals in ANG files but can be very small floats in my testing.
    """
    hkl_column_temp = 0
    current_hkl_index = last_hkl_col_location - 1
    while hkl_column_temp == 0 or hkl_column_temp == 1:
    
        hkl_column_temp = struct.unpack('<b', \
                                         temp_hkl_location[current_hkl_index])[0]
        current_hkl_index -= 1
    
    tmpini = current_hkl_index - 4*number_hkl_families +3
    tmpfin = current_hkl_index +3
    hkl_5_col = np.asanyarray(struct.unpack('<{0:d}f'.format((tmpfin-tmpini)/4),  \
                                            temp_hkl_location[tmpini:tmpfin]))
    del hkl_column_temp
    
    """
    %     %Now find the 4th column of data from the hkl information
    %     hkl_4_col_temp = temp_hkl_location(current_hkl_index -  
                                             4*double(Number_hkl_families) - 
                                             (4*double(Number_hkl_families)) +
                                             2:current_hkl_index -  
                                             4*double(Number_hkl_families) - 2);
    % 
    %     %Extract every 4th element and transpose the array to match the *.ang
    %     %format
    %     hkl_4_col = hkl_4_col_temp(1:4:end)';
    % 
    %     %This is just used to show all the hkl information in column format
    % %     final_first_phase = cat(2,final_hkl_families, hkl_4_col, hkl_5_final,
                                      hkl_last_col);
    """
    tmpini = current_hkl_index - 4*number_hkl_families - 4*number_hkl_families + 2
    tmpfin = current_hkl_index - 4*number_hkl_families + 2
    hkl_4_col_tmp = np.asanyarray(struct.unpack('<{0:d}b'.format(tmpfin - tmpini),\
                                                 temp_hkl_location[tmpini:tmpfin]))
    hkl_4_col = np.reshape(hkl_4_col_tmp, [hkl_4_col_tmp.shape[0]/4, 4])[:,0]
    del hkl_4_col_tmp
    
    """
    %     %the above lines of code will extract the entire header
    %     %information for only 1 phase, so now we need to check to see if any more
    %     %phases exist. This is going to look like just a repeat of the code above
    %     %but I think it's easier to always look for 1 phase, then check if more
    %     %exist and if they do enter a function to parse then out then it would be
    %     %to enter the function first.  This idea my change later once I get the
    %     %code working reliably.
    % 
    % 
    %     %Multiple phase checking begins here.  The 2 is used as a counter to be
    %     %able to output the phase the number to the final *.ang file
    %     for phase_count = 2:number_of_phases
    %     %for phase_count = 2:length(last_hkl_col_location)
    %         while (new_phase == -128) || (new_phase == -65) || (new_phase == 0)
    % 
    %            new_phase = new_phase_all(new_phase_counter);
    %            new_phase_counter = new_phase_counter +1;
    % 
    %         end
    % 
    %         %Extract the begining location for the new phase
    %         new_formula_name = new_phase_all(new_phase_counter-1:end);
    % 
    %         %Find the ending location of the new phase 
    %         new_phase_name_location = strfind(new_formula_name, 0);
    % 
    %         %Extract the phase name from the file
    %         new_phase_name_decimal = new_formula_name(
                                                     1:new_phase_name_location(1));
    % 
    %         %Convert the phase name into a character string
    %         new_phase_name = char(new_phase_name_decimal);
    % 
    %         %check the string for the sysmetry number by searching beyond
    %         %The location of the end of the phase name
    %         new_symmetry_temp = 0; 
    % 
    %         %I don't like this, but it works.  Sometimes, TSL writes the second
    %         %phase name and uses a temp holder like 
                                                 "m Files TexSEM SpaceGroupsV2.bin"
    %         %which makes no sense at all. So the 100 here is offset the search to
    %         %begin past the temp holder.
    %         new_symmetry_counter = new_phase_name_location(1)+ 
                                                     100+new_phase_counter-1; 
    % 
    %         
    %          while (new_symmetry_temp == -128) || (new_symmetry_temp == -65) ...
    %                 || (new_symmetry_temp == 0)
    % 
    %             new_symmetry_temp = new_phase_all(new_symmetry_counter);
    %             new_symmetry_counter = new_symmetry_counter +1;
    % 
    %          end
    % 
    %          %Phase symmetry in decimal form
    %          new_phase_symmetry = new_symmetry_temp;
    % 
    %         %This is used to extract the begining of the hkl data. The first 
    %         %values are the Lattice Constants followed by the number of families 
    %         %followed by the actual hkl values
    %         new_temp_hkl_location = new_phase_all(new_symmetry_counter+1:end);
    % 
    %         %locate the begining of the LatticeConstants
    %         new_begin_lattice_constants_location = strfind(new_temp_hkl_location,
                                                               [0 0 0]);
    % 
    %         %This is used to extract the lattice constants and the number of
    %         %families from the temp_hkl_location_array
    %         new_temp_lattice_constants = new_temp_hkl_location(
                                       1:new_begin_lattice_constants_location(1));
    % 
    %         %This is used to get the Number of HKL families in decimal form
    %         new_Number_hkl_families = new_temp_lattice_constants(end-1);
    % 
    %         %This is used for testing to check next iteration, if there is one
    %         Number_hkl_families = new_Number_hkl_families;
    % 
    %         %This is used to extract the lattice constants encoded.
    %         new_lattice_constants_encoded = new_temp_lattice_constants(3:end-2);
    % 
    %         %reshpae the lattice constants into an array 4x6
    %         new_reshaped_lattice_constants = reshape(
                                    new_lattice_constants_encoded,
                                    [4,length(new_lattice_constants_encoded)/4]);
    % 
    %         %final lattice constants in decimal form
    %         new_lattice_constants = typecast(new_reshaped_lattice_constants(:),
                                                     'single');
    % 
    %         %This is a counter used to keep track of the current phase durin
    %         %the writting
    %         next_phase_check = 0;
    %  
    %         %This is used to determine how many values to read from the
                           extracted data
    %         %The 3 and the 4 are because there are 3 hkl values and each value is
    %         %seperated by three zeros before the next hkl is read in
    %         new_Total_number_hkl_families = 
                                         double(new_Number_hkl_families) * 3 * 4;
    % 
    %         %This extracts the hkl families
    %         new_temp_hkl_families = new_temp_hkl_location(
                                     length(new_temp_lattice_constants)+3:
                                     length(new_temp_lattice_constants)+
                                     new_Total_number_hkl_families-1);
    % 
    %         %Pull every 4th element from the array
    %         new_temp_hkl_families = new_temp_hkl_families(1:4:end);
    % 
    %         %rehsape to match the *.ang output
    %         new_final_hkl_families = reshape(new_temp_hkl_families,
                                         [3 (length(new_temp_hkl_families)/3)])';
    % 
    %         %Extract the formula name region, the 3 is used because of the
    %         %trailing 0 0 0 after the hkl families
    %         new_temp_formula_name = new_temp_hkl_location(
                                  length(new_temp_lattice_constants)+
                                  new_Total_number_hkl_families+3:end);
    % 
    %         %Search for the formula name
    %         new_formula_name_location = strfind(new_temp_formula_name, 0);
    % 
    %         %Use the first index to locate the end of the forumla name
    %         new_formula_name = char(
                            new_temp_formula_name(1:new_formula_name_location(1)));
    % 
    %         %Now let's find the last column of the hkl information
    %         %The string here is the locating string for the begining of the last
    %         %col info.  If there is more than one phase, it will return the 
    %         %location of all the last column info for each phase
    %         new_last_hkl_col_location = strfind(new_temp_hkl_location,
                          [2 0 0 0 -1 -1 -1 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] );
    % 
    %         %Now extract the first phase last hkl col information.  Okay, so this
    %         %is a little bit confusing.  The 25 is because the strfind locates
    %         %the string above, which is 16 characters long.  There are exactly 8 
    %         %characters of zeros before you reach the first index.  This totals  
    %         %to be 24. Now, the 20 is in here because you have to include the 25
    %         %chracters described eariler, but you need to subtract 4 because you
    %         % are starting at the first hkl row so you can't count that.  This 
                totals to be 20.
    %         new_temp_hkl_last_col = new_temp_hkl_location(
                                        new_last_hkl_col_location(1)+24:
                                        new_last_hkl_col_location(1)+20+
                                        (4*double(new_Number_hkl_families)));
    % 
    %         %Extract every 4th element from the array and transpose it to match
    %         %the output of the *.ang file format
    %         new_hkl_last_col = new_temp_hkl_last_col(1:4:end)';
    % 
    %         %Now let's find the 5th column by locating the last column
    %         %nad reverse looking for a string not 1 or 0
    %         new_hkl_column_temp = 0; 
    %         new_current_hkl_index = new_last_hkl_col_location(1)-1; 
    % 
    %          while (new_hkl_column_temp == 0) || (new_hkl_column_temp == 1)
    % 
    %             new_hkl_column_temp = new_temp_hkl_location(
                                                           new_current_hkl_index);
    %             new_current_hkl_index = new_current_hkl_index - 1;
    % 
    %          end
    % 
    %         %Extract the 5th column values.  The reason for the + 2 is because
    %         %current_hkl_index has already decremented and we need to account 
    %         %for the 1 in locating the starting position.  The reason for the + 1
    %         %is because the while loop is designed to decrement current_hkl_index
    %         %after it searchs the string.  This means once the location has been
    %         %found, the while loop will still decrement current_hkl_index.  
               This has some issues because
    %         %of Matlab truncating the decimal, so anything less than 1 becomes 0.
    %         new_hkl_5_col = new_temp_hkl_location(new_current_hkl_index - 
                                        (4*double(new_Number_hkl_families))+2:
                                        new_current_hkl_index+1);
    % 
    %         %reshpae the 5th hkl column
    %         new_hkl_5_col_unreshaped = reshape(new_hkl_5_col,
                                                 [4,length(new_hkl_5_col)/4]);
    % 
    %         %The 5th column in it's decimal form
    %         new_hkl_5_final = typecast(new_hkl_5_col_unreshaped(:), 'single');
    %        
    %         %Now we need to remove the Nan from the HKL families due to some
    %         %conversion errors!!
    %         Nan_index_hkl = isnan(new_hkl_5_final);
    %         new_hkl_5_final(Nan_index_hkl) = 0;  
    %         
    %         
    %         %Now find the 4th column of data from the hkl information
    %         new_hkl_4_col_temp = new_temp_hkl_location(
                                      new_current_hkl_index -  
                                      4*double(new_Number_hkl_families) - 
                                      (4*double(new_Number_hkl_families))+2:
                                      new_current_hkl_index -  
                                      4*double(new_Number_hkl_families)-2);
    % 
    %         %Extract every 4th element and transpose the data to match the *.ang
    %         %file format
    %         new_hkl_4_col = new_hkl_4_col_temp(1:4:end)';
    % 
    %         inds2 = find(isnan(new_hkl_4_col));
    %         
    %         new_hkl_4_col(inds2) = 0;  
    %         
    %         
    %         
    %         %This is just used to show all the hkl information in column format
    % %         final_new_phase = cat(2,new_final_hkl_families, 
                                         new_hkl_4_col, new_hkl_5_final, 
                                         new_hkl_last_col); 
    % 
    %         %Used here only for debugging
    %         %pause
    % 
    % 
    %        %Now let's write the data for the first phase to the output file
    %         new_final_Phase_out(phase_count) = strcat({'# Phase '}, 
                                         num2str(phase_count));
    %         new_final_MaterialName_out(phase_count) = strcat(
                                    {'# MaterialName  	'}, new_phase_name);
    %         new_final_Formula_out(phase_count) = strcat({
                                    '# Formula     	'},  new_formula_name);
    %         new_final_info_out(phase_count) = strcat({'# Info 		'});
    %         new_final_symmetry_out(phase_count) = strcat(
                       {'# Symmetry              '}, num2str(new_phase_symmetry));
    %         new_final_LatticeConstants_out(phase_count) = strcat(
                                         {'# LatticeConstants      '}, 
                                         {num2str(new_lattice_constants(1),3)}, 
                                         {' '}, ....
    %                                     num2str(new_lattice_constants(2),3), 
                                               {' '}, 
                                               num2str(new_lattice_constants(3),3),
                                               {' '}, ...
    %                                          num2str(new_lattice_constants(4),3),
                                               {' '}, 
                                               num2str(new_lattice_constants(5),3),
                                               {' '},...
    %                                          num2str(new_lattice_constants(6),3)
                                                                      );
    % 
    %         new_final_Number_hkl_families(phase_count) = strcat(
                                               {'# NumberFamilies        '}, 
                                                num2str(new_Number_hkl_families));
    % 
    %         Total_phases_for_output(phase_count) = new_Number_hkl_families;                                
    % 
    %         %build the hkl array of strings that will be printed to the
    %         %file
    %         for new_hkl_counter=1:new_Number_hkl_families
    %         %for hkl=1:size(Number_hkl_families)
    % 
    %           new_final_hkl_string(phase_count, double(new_hkl_counter)) =
                       strcat({'# hklFamilies            '}, 
                       num2str(new_final_hkl_families(new_hkl_counter)), {' '}, ...
    %                  num2str(new_final_hkl_families(
                     double(new_Number_hkl_families)+double(new_hkl_counter))), ...
    %                    {' '}, num2str(new_final_hkl_families(
              (2 * double(new_Number_hkl_families))+ double(new_hkl_counter))), ...
    %                    {' '}, num2str(new_hkl_4_col(new_hkl_counter)),
                         {' '}, num2str(new_hkl_5_final(new_hkl_counter), 6), ...
    %                     {' '}, num2str(new_hkl_last_col(new_hkl_counter))  );      
    % 
    %         end
    % 
    % 
    %     end
    -------------------------------------------------------------------------------
    Full original code above. Key Shiveley comments have %s in translation below.
    """
    phase_out = [final_phase_1]
    formula_out = [formula_name]
    phase_symmetry_out = [phase_1_symmetry]
    hkl_families_out = [final_hkl_families]
    hkl_last_col_out = [hkl_last_col]
    hkl_5_col_out = [hkl_5_col]
    hkl_4_col_out = [hkl_4_col]
    lattice_constants_out = [lattice_constants_phase_1]
    num_hkl_families_out = [number_hkl_families]
    del final_phase_1, formula_name, phase_1_symmetry, final_hkl_families
    del hkl_last_col, hkl_5_col, hkl_4_col
    
    for phase_count in np.arange(2, number_of_phases+1):
        #%This extracts all the information after the previous phase 
        tmpini = last_hkl_col_location+24+4*number_hkl_families+2
    
        #% check the string for the sysmetry number by searching beyond
        #% The location of the end of the phase name
        new_phase = 0
        seek_loc = symmetry_counter+1+last_hkl_col_location+24+\
                                              4*number_hkl_families+2
        mm.seek(seek_loc)
        while new_phase == -128 or new_phase == -65 or new_phase == 0:
           new_phase = struct.unpack('<b', mm.read_byte())[0]
           seek_loc += 1
    
        # Find the ending location of the new phase 
        new_phase_name_location = mm[seek_loc-1:].find(struct.pack('<b', 0))
        new_phase_name = mm[seek_loc-1:seek_loc+new_phase_name_location-1]
    
        #%check the string for the sysmetry number by searching beyond
        #%The location of the end of the phase name
        new_symmetry_temp = 0; 
    
        #%I don't like this, but it works.  Sometimes, TSL writes the second
        #%phase name and uses a temp holder like "m Files TexSEM SpaceGroupsV2.bin"
        #%which makes no sense at all.  So the 100 here is offset the search to
        #%begin past the temp holder.
        seek_loc += new_phase_name_location+100
        mm.seek(seek_loc)
        while new_symmetry_temp == -128 \
           or new_symmetry_temp == -65 \
           or new_symmetry_temp == 0:
            new_symmetry_temp = struct.unpack('<b', mm.read_byte())[0]
            seek_loc += 1
        new_phase_symmetry = new_symmetry_temp
    
        #    %This is used to extract the begining of the hkl data.  The first 
        #    %values are the Lattice Constants followed by the number of families 
        #    %followed by the actual hkl values
        new_begin_lattice_constants_location = mm[seek_loc+1:].find(\
                                                      struct.pack('<3b', 0, 0, 0))
        tmpfin = seek_loc+1+new_begin_lattice_constants_location
        new_temp_lattice_constants = mm[seek_loc+1:tmpfin]
        number_hkl_families = struct.unpack('<b',new_temp_lattice_constants[-1])[0]
    
        new_number_hkl_families = struct.unpack('<B',\
                                                 new_temp_lattice_constants[-1])[0]
        new_lattice_constants_encoded = new_temp_lattice_constants[2:-1]
        new_lattice_constants = np.array(struct.unpack('<6f',\
                                                    new_lattice_constants_encoded))
        
        new_total_number_hkl_families = new_number_hkl_families * 3 * 4
        length_of_new_temp_lattice_constants = np.asanyarray(\
                   [struct.unpack('<b', x)[0] for x in new_temp_lattice_constants]\
                                                             ).shape[0]
        
        tmpini = seek_loc+1+length_of_new_temp_lattice_constants+3
        tmpfin = seek_loc+1+length_of_new_temp_lattice_constants+3+\
                                                      new_total_number_hkl_families
        new_temp_hkl_families = struct.unpack(\
               '<{0}b'.format(new_total_number_hkl_families),mm[tmpini:tmpfin]) 
        th = np.asanyarray(new_temp_hkl_families)
        new_final_hkl_families = np.reshape(\
                                   np.reshape(th,\
                                              [th.shape[0]/4, 4])[:,0],\
                                              [th.shape[0]/4/3, 3])
        
        #%Extract the formula name region, the 3 is used because of the
        #%trailing 0 0 0 after the hkl families
        tmpini = seek_loc + 1 + length_of_new_temp_lattice_constants +\
                  new_total_number_hkl_families + 3
        new_temp_formula_name = mm[tmpini:]
        new_formula_name_location = new_temp_formula_name.find(struct.pack('<b',0))
        new_formula_name = "".join(struct.unpack(\
                              '<{0}c'.format(new_formula_name_location), \
                               new_temp_formula_name[0:new_formula_name_location]))
        
        # NEW TEMP HKL LOCATION IN PYTHON IS = mm[seek_loc+1:]    
        new_last_hkl_col_location = mm[seek_loc+1:].find(struct.pack('<24b', \
                                                          2,  0, 0, 0, -1, -1,\
                                                         -1, -1, 0, 0,  0,  0,\
                                                          0,  0, 0, 0,  0,  0,
                                                          0,  0, 0, 0,  0,  0))
        tmpini = seek_loc+1+new_last_hkl_col_location + 24
        tmpend = seek_loc+1+new_last_hkl_col_location+24+4*new_number_hkl_families
        new_temp_hkl_last_col = np.asanyarray(\
                 struct.unpack('<{0:d}b'.format(tmpend-tmpini), mm[tmpini:tmpend]))
        new_hkl_last_col = np.reshape(new_temp_hkl_last_col,\
                                        [new_temp_hkl_last_col.shape[0]/4, 4])[:,0]
    
        #%Now let's find the 5th column by locating the last column
        #%nad reverse looking for a string not 1 or 0
        new_hkl_column_temp = 0
        new_current_hkl_index = new_last_hkl_col_location - 1
    
        while new_hkl_column_temp == 0 or new_hkl_column_temp == 1:
    
            new_hkl_column_temp = struct.unpack('<b', \
                                          mm[seek_loc+1+new_current_hkl_index])[0]
            new_current_hkl_index -= 1
    
        tmpini = seek_loc + new_current_hkl_index - 4*new_number_hkl_families + 3
        tmpfin = seek_loc + new_current_hkl_index + 3
        new_hkl_5_col_tmp = mm[tmpini:tmpfin]
        new_hkl_5_final = np.asanyarray(struct.unpack(\
                           '<{0:d}f'.format((tmpfin-tmpini)/4), new_hkl_5_col_tmp))
    
        tmpini = seek_loc + 3 + new_current_hkl_index -  \
                        4*new_number_hkl_families - 4*new_number_hkl_families
        tmpfin = seek_loc + 3 + new_current_hkl_index -  \
                        4*new_number_hkl_families
        new_hkl_4_col_tmp = mm[tmpini:tmpfin]
        new_hkl_4_col_tmp2 = np.asanyarray(struct.unpack(\
                               '<{0:d}b'.format(tmpfin-tmpini), new_hkl_4_col_tmp))
        new_hkl_4_final = np.reshape(new_hkl_4_col_tmp2, \
                                           [new_hkl_4_col_tmp2.shape[0]/4, 4])[:,0]
    
        
        phase_out.append(new_phase_name)
        formula_out.append(new_formula_name)
        phase_symmetry_out.append(new_phase_symmetry)
        hkl_families_out.append(new_final_hkl_families)
        hkl_last_col_out.append(new_hkl_last_col)
        hkl_5_col_out.append(new_hkl_5_final)
        hkl_4_col_out.append(new_hkl_4_final)
        lattice_constants_out.append(new_lattice_constants)
        num_hkl_families_out.append(number_hkl_families)
    
    del new_hkl_4_col_tmp, new_hkl_4_col_tmp2, tmpini, tmpfin
    del new_hkl_5_col_tmp, new_temp_hkl_last_col
    
    """
    % function [ RelevantData,Xstep,Ystep ] = Osc2Ang_TSL7( OscFile )
    % % [RelevantData,Xstep,Ystep] = Osc2Ang_TSL7( OscFile )
    % % This function is used to decode the .osc format 
    % % 
    % % Input: 'OscFile' is a string containing the path and directory of the osc
                                                    file to decode (from version 7) 
    % % Outputs:
    % %          RelevantData is a [N x 10] array containing the information in a 
    % %          standard .ANG file. Xstep and Ystep are the step sizes, in
                 microns, in each direction
    % % 
    % % Function coded by Adam Shiveley, USAF AFRL/RXCM
    % % 
    %             
    %             fid = fopen(OscFile,'rb');
    %             data = transpose( fread( fid, '*bit8' ) );
    %             start_euler_indices = strfind(data,[-71 11 -17 -1 2]);
    %             end_euler_indices = strfind(data, [-71 11 -17 -1]);
    %             end_euler = end_euler_indices(find(
                                           end_euler_indices == start_euler_indices
                                                                              )+1);
    %             euler_angles_ascii = data(start_euler_indices:end_euler-1);
    % 
    %             final_euler_angles = reshape(euler_angles_ascii,
                                                 [4,length(euler_angles_ascii)/4]);
    %           
    %         
    %             RelevantData = typecast(final_euler_angles(:), 'single');
    %             
    %             
    % % RelevantData = typecast(final_euler_angles(:), 'double')  ;          
    % % RelevantData = typecast(final_euler_angles(:), 'uint16')            
    % % RelevantData = typecast(final_euler_angles(:), 'single')
    % 
    %             Xstep = RelevantData(4);
    %             Ystep = RelevantData(5);
    % 
    %             RelevantData(1:5) = [];
    % 
    %             RelevantData = reshape(RelevantData, 
                                                    [14,length(RelevantData)/14])';
    %             % convert to stitch code friendly format...
    %             RelevantData(:,11:end)= [];
    %   
    %             fclose(fid);
    % 
    % end
    -------------------------------------------------------------------------------
    % function [ RelevantData,Xstep,Ystep ] = Osc2Ang_TSL6( OscFile )
    % % [RelevantData,Xstep,Ystep] = Osc2Ang_TSL6( OscFile )
    % % This function is used to decode the .osc format 
    % % 
    % % Input: 'OscFile' is a string containing the path and directory of the osc 
                                                    file to decode (from version 6) 
    % % Outputs:
    % %          RelevantData is a [N x 10] array containing the information in a 
    % %          standard .ANG file. Xstep and Ystep are the step sizes,
                 in microns, in each direction
    % % 
    % % Function coded by Adam Shiveley, USAF AFRL/RXCM
    % % 
    %             
    %             fid = fopen(OscFile,'rb');
    %             data = transpose( fread( fid, '*bit8' ) );
    %             start_euler_indices = strfind(data,[-71 11 -17 -1 2]);
    %             end_euler_indices = strfind(data, [-71 11 -17 -1]);
    %             end_euler = end_euler_indices(find(
                                          end_euler_indices == start_euler_indices
                                                                              )+1);
    %             euler_angles_ascii = data(start_euler_indices:end_euler-1);
    % 
    %             final_euler_angles = reshape(euler_angles_ascii,
                                                 [4,length(euler_angles_ascii)/4]);
    %             RelevantData = typecast(final_euler_angles(:), 'single');
    % 
    %             Xstep = RelevantData(4);
    %             Ystep = RelevantData(5);
    % 
    %             RelevantData(1:5) = [];
    % 
    %             RelevantData = reshape(RelevantData, 
                                                    [10,length(RelevantData)/10])';
    %             
    %             fclose(fid);
    % 
    % end
    ------------------------------------------------------------------------------
    % function [ RelevantData,Xstep,Ystep ] = Osc2Ang_TSL531( OscFile )
    % % [RelevantData,Xstep,Ystep] = Osc2Ang_TSL531( OscFile )
    % % This function is used to decode the .osc format 
    % % 
    % % Input: 'OscFile' is a string containing the path and directory of the osc
                                                file to decode (from version 5.3.1)
    % % Outputs:
    % %          RelevantData is a [N x 10] array containing the information in a 
    % %          standard .ANG file. Xstep and Ystep are the step sizes, 
                 in microns, in each direction
    % % 
    % % Function coded by Adam Shiveley, USAF AFRL/RXCM
    % % 
    % %% code:             
    %             fid = fopen(OscFile,'rb');
    %             data = transpose( fread( fid, '*bit8' ) );
    %             start_euler_indices = strfind(data,[-71 11 -17 -1 2]);
    %             end_euler_indices = strfind(data, [-71 11 -17 -1]);
    %             end_euler = end_euler_indices(end);
    %             euler_angles_ascii = data(start_euler_indices:end_euler-1);
    % 
    %             final_euler_angles = reshape(euler_angles_ascii,
                                                 [4,length(euler_angles_ascii)/4]);
    %             RelevantData = typecast(final_euler_angles(:), 'single');
    % 
    %             Xstep = RelevantData(3);
    %             Ystep = RelevantData(4);
    % 
    %             RelevantData(1:4) = [];
    % 
    %             RelevantData = reshape(RelevantData, 
                                                    [10,length(RelevantData)/10])';
    %             
    %             fclose(fid);
    % 
    % end
    """
    start_euler_indices = mm.find(struct.pack('<5b', -71, 11, -17, -1, 2))
    
    end_euler_indices = mm[start_euler_indices+2:].find(\
                                              struct.pack('<4b', -71, 11, -17, -1))
                                              
    euler_angles_ascii = mm[start_euler_indices:\
                                         start_euler_indices + end_euler_indices+2]
                                         
    relevant_data = np.asanyarray(\
                     struct.unpack(\
                      '<{0:d}f'.format((end_euler_indices+2)/4), \
                                                               euler_angles_ascii))
    
    """
    % 
    %     %There are changes between TSL 5.31 and TSL 6.  The biggest
    %     %changes I can determine is where the begining of the first phase is
    %     %located.   In TSL 5.31, it is located at 1929 and in TSL 6 it is
    %     %located at 1989.  Using these two numbers, we can determine which
    %     %version of Osc2Ang to call.
    %     if( (start_phase_location == 1989) || (start_phase_location == 1993) )        
    %         
    %         %So now we should make a call to Osc2Ang to extract the needed info
    %         %Attempt an Osc2Ang to extract information using TSL7, otherwise
    %         %catch the error and send it to TSL6
    %         try    
    %             [CurrentOsc,Xstep,Ystep] = Osc2Ang_TSL7(OscFile);
    %             TSL_version = 7;
    %         catch
    %             [CurrentOsc,Xstep,Ystep] = Osc2Ang_TSL6(OscFile);
    %             TSL_version = 6;   
    %         end
    %     else
    %         
    %         %So now we should make a call to Osc2Ang to extract the needed info
    %         [CurrentOsc,Xstep,Ystep] = Osc2Ang_TSL531(OscFile);
    %         TSL_version = 5;
    %   
    %     end
    ------------------------------------------------------------------------------
    Based on above comments, it seems like an if-elseif approach might be better.
    The data contained in the final four columns of the v7 file are floats that
    appear to round off to the results contained in the ANG file.
    """
    if start_phase_location == 1992: # we have TSL OIM v7'''
        xstep = relevant_data[3]
        ystep = relevant_data[4]
        data_length = relevant_data[5:].shape[0]/14
        out_data = np.reshape(relevant_data[5:], [data_length, 14])
    
    elif start_phase_location == 1988: # We have TSL OIM v6
        xstep = relevant_data[3]
        ystep = relevant_data[4]
        data_length = relevant_data[5:].shape[0]/10
        out_data = np.reshape(relevant_data[5:], [data_length, 10])
        
    elif start_phase_location == 1928: # We have TSL OIM v5.31
        xstep = relevant_data[2]
        ystep = relevant_data[3]
        data_length = relevant_data[4:].shape[0]/10
        out_data = np.reshape(relevant_data[4:], [data_length, 10])
    
    """
    %     %Now let's check the end x and y position to see if the scan grid was
    %     %either square or hex
    %     if(mod(Xstep, Ystep) == 0); 
    %        grid = 'SqrGrid';
    %     else
    %         grid = 'HexGrid';
    %     end
    """
    if np.mod(xstep, ystep) == 0: 
       grid = 'SqrGrid'
    else:
       grid = 'HexGrid'
    
    
    """
    %     %now we need to determine the Xstep, Ystep, NCOLS_ODD, NCOLS_EVEN,
    %     %NROWS and GRID.  From what I can tell, these values are not directly
    %     %sotred in the .osc file.
    ---------------------------------------------------------------------
    %     %Now let's calculate the Ncols odd and even along with the Nrows
    % 
    %     %NCOLS_ODD ending x position / ystep
    %     Ncols_Odd = floor(CurrentOsc(end,4) / Ystep) + 1;
    % 
    % 
    %     %NCOLS_EVEN is a little tricky to figure out.  We need to determine the
    %     %remainder. If the remainder is 0, we need to add 1 to the number
    %     if mod(CurrentOsc(end,4),Xstep) == 0
    %         Ncols_Even = round(CurrentOsc(end,4) / Xstep) + 1;
    %     else
    %         Ncols_Even = round(CurrentOsc(end,4) / Xstep);
    %     end
    % 
    %     %This is used to determine the NRows in the scan.
    %     NRows = (CurrentOsc(end,5) / Ystep) + 1;
    -----------------------------------------------------------------------------
    The approach above didn't seem to always output the correct results due to
    round-off errors. Floor/ceiling seems to be a better approach. For xstep and
    ystep, see above.
    """
    nrows = np.round(out_data[-1,4]/ystep) + 1
    ncols_even = np.floor(out_data[-1, 3] / xstep) + 1
    ncols_odd  = np.ceil( out_data[-1, 3] / xstep) + 1
    
    """
    %     %Now let's write the data for the first phase to the output file
    %     final_Phase_out = strcat({'# Phase '}, '1');
    %     final_MaterialName_out = strcat({'# MaterialName  	'}, final_phase_1);
    %     final_Formula_out = strcat({'# Formula     	'}, formula_name);
    %     final_info_out = strcat({'# Info 		'});
    %     final_symmetry_out = strcat({'# Symmetry              '}, 
                                num2str(phase_1_symmetry));
    %     final_LatticeConstants_out = strcat({'# LatticeConstants      '}, 
    %                                 {num2str(lattice_constants(1),3)}, {' '}, ...
    %                                  num2str(lattice_constants(2),3), {' '}, ...
    %                                  num2str(lattice_constants(3),3), {' '}, ...
    %                                  num2str(lattice_constants(4),3), {' '}, ...
    %                                  num2str(lattice_constants(5),3), {' '},...
    %                                  num2str(lattice_constants(6),3) );
    % 
    %     final_Number_hkl_families = strcat({'# NumberFamilies        '},
                                       num2str(Number_hkl_families));
    % 
    %     %Let's store a variable for the number of hkl families for phase 1 which
    %     %will be used later on for the writing
    %     Number_hkl_families_phase_1_write = Number_hkl_families;                                    
    % 
    % 
    %     %build the hkl array of strings that will be printed to the file
    %     for hkl=1:Number_hkl_families
    %     %for hkl=1:size(Number_hkl_families)
    % 
    %         final_hkl_string(hkl) = strcat({'# hklFamilies           '}, 
                                      num2str(final_hkl_families(hkl)), {' '}, ...
    %                                 num2str(final_hkl_families(
                                                     Number_hkl_families+hkl)), ...
    %                                 {' '}, 
                                      num2str(final_hkl_families(
                                              (Number_hkl_families*2)+hkl)), {' '},
                                              num2str(hkl_4_col(1)), ...
    %                                 {' '}, num2str(hkl_5_final(hkl), 6), {' '},
                                       num2str(hkl_last_col(hkl))  );      
    % 
    %     end
    --------------------------------------------------------------------------
    % 
    %     %Alright, now let's start writing the data
    % 
    %     %Write the calibration data
    %     fprintf(fileWriteID,'# TEM_PIXperUM          %s\r\n', 
                                                         num2str(TEM_PIXperUM, 7));
    %     fprintf(fileWriteID,'# x-star                %s\r\n', 
                                                         num2str(x_star, 7));
    %     fprintf(fileWriteID,'# y-star                %s\r\n', 
                                                         num2str(y_star, 7));
    %     fprintf(fileWriteID,'# z-star                %s\r\n', 
                                                         num2str(z_star, 7));
    %     fprintf(fileWriteID,'# WorkingDistance       %s\r\n', 
                                                    num2str(working_distance, 7));
    %     fprintf(fileWriteID,'#\r\n');
    % 
    %  
    %     %Okay, so now we have to write the information for each phase, but
    %     %begining at the ending phase and moving toward the first phase
    %     for phase_write=2:phase_count
    % 
    % 
    %        fprintf(fileWriteID, '%s\r\n', 
                          char(new_final_Phase_out(phase_count-next_phase_check)));
    % 
    %        fprintf(fileWriteID, '%s\r\n', 
                   char(new_final_MaterialName_out(phase_count-next_phase_check))); 
    % 
    %        fprintf(fileWriteID, '%s\r\n', 
                        char(new_final_Formula_out(phase_count-next_phase_check))); 
    % 
    %        fprintf(fileWriteID, '%s\r\n', 
                           char(new_final_info_out(phase_count-next_phase_check))); 
    % 
    %        fprintf(fileWriteID, '%s\r\n', 
                       char(new_final_symmetry_out(phase_count-next_phase_check))); 
    % 
    %        fprintf(fileWriteID, '%s\r\n',
               char(new_final_LatticeConstants_out(phase_count-next_phase_check))); 
    % 
    %        fprintf(fileWriteID, '%s\r\n', 
                char(new_final_Number_hkl_families(phase_count-next_phase_check)));
    % 
    %        %Now let's loop through the current phases hkl families & output them
    %        %to the file
    %        for write_hkl=1:Total_phases_for_output(phase_count-next_phase_check)   
    % 
    %             fprintf(fileWriteID, '%s\r\n', 
              char(new_final_hkl_string(phase_count-next_phase_check, write_hkl)));
    % 
    %        end
    %        
    %        %So TSL version 6 writes a variable called ElasticConstants.  I
    %        %believe this may be a temp holder right now. I can not seem to locate
    %        %this information in the osc file.  I am sure it is there, but every
    %        %test scan I run has the value -1, so I can't search for differences
    %        %between two files.
    %        if(start_phase_location == 1989)
    %        
    %         fprintf(fileWriteID, '# ElasticConstants       
                -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 \r\n');
    %         fprintf(fileWriteID, '# ElasticConstants       
                -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 \r\n');
    %         fprintf(fileWriteID, '# ElasticConstants       
                -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 \r\n');
    %         fprintf(fileWriteID, '# ElasticConstants       
                -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 \r\n');
    %         fprintf(fileWriteID, '# ElasticConstants       
                -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 \r\n');
    %         fprintf(fileWriteID, '# ElasticConstants       
                -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 \r\n');
    %            
    %        end
    %        
    % 
    %         %Okay, so I'm not sure what this means, so it it hard coded for now.
    %         %It never seems to change, though so I have now way of locating it
    %         %inside the file
    %         fprintf(fileWriteID,'# Categories0 0 0 0 0 \r\n');
    %         fprintf(fileWriteID,'#\r\n');
    % 
    %         next_phase_check = next_phase_check + 1;
    % 
    %     end
    % 
    %     % write the 1st phase to the file now
    %     fprintf(fileWriteID, '%s\r\n', char(final_Phase_out));
    % 
    %     fprintf(fileWriteID, '%s\r\n', char(final_MaterialName_out)); 
    % 
    %     fprintf(fileWriteID, '%s\r\n', char(final_Formula_out)); 
    % 
    %     fprintf(fileWriteID, '%s\r\n', char(final_info_out)); 
    % 
    %     fprintf(fileWriteID, '%s\r\n', char(final_symmetry_out)); 
    % 
    %     fprintf(fileWriteID, '%s\r\n', char(final_LatticeConstants_out)); 
    % 
    %     fprintf(fileWriteID, '%s\r\n', char(final_Number_hkl_families));
    % 
    %     %Now let's loop through the current phases hkl families and output them
    %     %to the file
    %     for write_hkl_2=1:Number_hkl_families_phase_1_write 
    % 
    %          fprintf(fileWriteID, '%s\r\n', char(final_hkl_string(write_hkl_2)));
    % 
    %     end
    % 
    % 
    %     %So now we need to write the ElasticConstants for the first phase
    %     if(start_phase_location == 1989)
    %        
    %       fprintf(fileWriteID, '# ElasticConstants       
                -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 \r\n');
    %       fprintf(fileWriteID, '# ElasticConstants      
                -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 \r\n');
    %       fprintf(fileWriteID, '# ElasticConstants      
                -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 \r\n');
    %       fprintf(fileWriteID, '# ElasticConstants       
                -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 \r\n');
    %       fprintf(fileWriteID, '# ElasticConstants      
                -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 \r\n');
    %       fprintf(fileWriteID, '# ElasticConstants      
                -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 \r\n');
    %            
    %     end
    %        
    %     %Okay, so I'm not sure what this means, so it it hard coded for now.
    %     %It never seems to change, though so I have now way of locating it
    %     %inside the file
    %     fprintf(fileWriteID,'# Categories0 0 0 0 0 \r\n');
    %     fprintf(fileWriteID,'#\r\n');
    % 
    % 
    %     %Now let's write the remaining information to the file
    %     fprintf(fileWriteID,'# GRID: %s\r\n', grid);
    %     fprintf(fileWriteID,'# XSTEP: %s\r\n', num2str(Xstep, 6));
    %     fprintf(fileWriteID,'# YSTEP: %s\r\n', num2str(Ystep, 6));
    %     fprintf(fileWriteID,'# NCOLS_ODD: %d\r\n', Ncols_Odd);
    %     fprintf(fileWriteID,'# NCOLS_EVEN: %d\r\n', Ncols_Even);
    %     fprintf(fileWriteID,'# NROWS: %d\r\n', int8(NRows));
    %     fprintf(fileWriteID,'#\r\n');
    % 
    % 
    %     %Now we need to check to see if any info was added to scan by the user
    %     if(~isempty(info))
    % 
    %         fprintf(fileWriteID,'# INFO: %s\r\n', info);
    %         fprintf(fileWriteID,'#\r\n');
    % 
    %     end  
    % 
    %     %Now let's right the remaing header information
    %     fprintf(fileWriteID,'# OPERATOR: 	%s\r\n', operator);
    %     fprintf(fileWriteID,'#\r\n');
    %     fprintf(fileWriteID,'# SAMPLEID:    %s\r\n', Sample_ID);
    %     fprintf(fileWriteID,'#\r\n');
    %     fprintf(fileWriteID,'# SCANID: 	%s\r\n', Scan_ID);
    %     fprintf(fileWriteID,'#\r\n');
    %    
    % %end
    ------------------------------------------------------------------------------
    Shiveley's format for the initial lines has 7 decimals rather than the 6 that
    is output for an ANG by v7.
    """
    
    if ang_output == True:
    
        # Allow user to specify their own *.ANG output filename
        if ang_output_filename == None:
            # may not be perfectly robust solution, but should work 99+% of times
            ang_output_filename = filename.split('.osc')[0] + '.ang'
    
        import io
        with io.open(ang_output_filename, 'w', newline='\r\n') as fo:
        
            fo.write(u'# TEM_PIXperUM          {0:.6f}\n'.format(TEM_PIXperUM))
            fo.write(u'# x-star                {0:.6f}\n'.format(x_star))
            fo.write(u'# y-star                {0:.6f}\n'.format(y_star))
            fo.write(u'# z-star                {0:.6f}\n'.format(z_star))
            fo.write(u'# WorkingDistance       {0:.6f}\n'.format(working_distance))
            fo.write(u'#\n')
            
            for phase_num in np.arange(number_of_phases, 0, -1):
                fo.write(u'# Phase {0:d}\n'.format(phase_num))
                fo.write(u'# MaterialName  \t{0}\n'.format(phase_out[phase_num-1]))
                fo.write(u'# Formula     \t{0}\n'.format(formula_out[phase_num-1]))
                fo.write(u'# Info \t\t\n')
                fo.write(u'# Symmetry              {0:d}\n'.format(\
                                                  phase_symmetry_out[phase_num-1]))
                fo.write(u'# LatticeConstants      \
                   {0:5.3f} {1:5.3f} {2:5.3f}  {3:7.3f}  {4:7.3f}  {5:7.3f}\n'.format(\
                            lattice_constants_out[phase_num-1][0], \
                            lattice_constants_out[phase_num-1][1], \
                            lattice_constants_out[phase_num-1][2], \
                            lattice_constants_out[phase_num-1][3], \
                            lattice_constants_out[phase_num-1][4], \
                            lattice_constants_out[phase_num-1][5]))
                fo.write(u'# NumberFamilies        {0}\n'.format(\
                                                num_hkl_families_out[phase_num-1]))
                
                hklfam = hkl_families_out[phase_num-1]
                hkl4 = hkl_4_col_out[phase_num-1]
                hkl5 = hkl_5_col_out[phase_num-1]
                hklfin = hkl_last_col_out[phase_num-1]
                tmpstr = u'# hklFamilies   \t{0: >2d} {1: >2d} {2: >2d} '+\
                            '{3:d} {4:.6f} {5:d}\n'
                for i in np.arange(0, num_hkl_families_out[phase_num-1]):
                    fo.write(tmpstr.format(hklfam[i,0], hklfam[i,1], hklfam[i,2],\
                                        hkl4[i], hkl5[i], hklfin[i]))
        
                # Shiveley puts this only for v6, but it seems to also be output
                # for v7. We will write dummy numbers for now.
                if start_phase_location == 1988 or start_phase_location == 1992:
                    for i in np.arange(0, 6):
                        fo.write(u'# ElasticConstants \t-1.000000 -1.000000 \
                                   -1.000000 -1.000000 -1.000000 -1.000000\n')
                
                # Writing dummy numbers...
                fo.write(u'# Categories0 0 0 0 0 \n')
                fo.write(u'#\n')
            
            fo.write(u'# GRID: {0}\n'.format(grid))
            fo.write(u'# XSTEP: {0:.6f}\n'.format(xstep))
            fo.write(u'# YSTEP: {0:.6f}\n'.format(ystep))
            fo.write(u'# NCOLS_ODD: {0:d}\n'.format(ncols_odd.astype(np.int)))
            fo.write(u'# NCOLS_EVEN: {0:d}\n'.format(ncols_even.astype(np.int)))
            fo.write(u'# NROWS: {0:d}\n'.format(nrows.astype(np.int)))
            fo.write(u'#\n')
            fo.write(u'# OPERATOR: \t{0}\n'.format(operator_name))
            fo.write(u'#\n')
            fo.write(u'# SAMPLEID: \t{0}\n'.format(sample_ID))
            fo.write(u'#\n')
            fo.write(u'# SCANID: \t{0}\n'.format(scan_ID))
            fo.write(u'#\n')
            
            if start_phase_location == 1992: # TSL OIM v7
                for i in np.arange(0, data_length):
                    fmt_str =u'{0: >9.5f} {1: >9.5f} {2: >9.5f} {3: >12.5f} '+\
                             u'{4: >12.5f} {5: >6.1f} {6: >6.3f} {7: >2d} '+ \
                             u'{8: >6d} {9: >6.3f} {10: >9.6f} {11: >9.6f} '+\
                             u'{12: >9.6f} {13: >9.6f} \n'    
                    p = out_data[i, :]          
                    fo.write(fmt_str.format(p[0], p[1], p[2], p[3], p[4], p[5], \
                                            p[6], p[7].astype(np.int), \
                                            p[8].astype(np.int), p[9], p[10], \
                                            p[11], p[12], p[13]))
            elif start_phase_location == 1988 or start_phase_location == 1928: # TSL OIM v5.31 thru 6
                for i in np.arange(0, data_length):
                    fmt_str = u'{0: > 9.5f} {1: > 9.5f} {2: > 9.5f} {3: > 12.5f} \
                                {4: > 12.5f} {5: > 8.1f} {6: > 6.3f} {7: > 2d} \
                                {8: > 6d} {9: > 6.3f} \n'       
                    p = out_data[i, :]              
                    fo.write(fmt_str.format(p[0], p[1], p[2], p[3], p[4], p[5], \
                                       p[6], p[7].astype(np.int), \
                                       p[8].astype(np.int), p[9]))       
      
    # NOW, create an ebsd object if the user wants one
    if create_ebsd_object == True:
        
        phase = out_data[:,7].astype(np.int)
        phaseidnums = np.arange(np.shape(phase_out)[0], 0, -1) - 1
        
        s = np.zeros(np.shape(phase))
        index = np.amax(phaseidnums)
        for item in phaseidnums:
            loc = phase == item
            s[loc] = interpret_point_group_name(np.str(phase_symmetry_out[index]),'tsl')
            index -= 1
            
        # Start working with what we've interpreted now
        o = orientation(quaternions = quat.from_bunge(bunge(out_data[:,0], out_data[:,1], out_data[:,2])),\
                        pointgroupnumbers = np.squeeze(s.astype(np.int)))
    
        ebsd_data = ebsd(orientations=o, x=out_data[:,3], y=out_data[:,4], phaseid=phase) # FIXME: need eventually to fix confusing variable naming here

        ebsd_data.calc_scanstepsize()
        ebsd_data.prepare_for_plotting()
        
        # Add other fields. Note that some fields are TSL-specific, and that two of
        # the fields (fit and ...) have not been added yet.
        ebsd_data.iq = out_data[:,5]
        ebsd_data.ci = out_data[:,6]
        
        # Write header stuff
        hdr = []
        hdr.append(u'# TEM_PIXperUM          {0:.6f}\n'.format(TEM_PIXperUM))
        hdr.append(u'# x-star                {0:.6f}\n'.format(x_star))
        hdr.append(u'# y-star                {0:.6f}\n'.format(y_star))
        hdr.append(u'# z-star                {0:.6f}\n'.format(z_star))
        hdr.append(u'# WorkingDistance       {0:.6f}\n'.format(working_distance))
        hdr.append(u'#\n')
        
        for phase_num in np.arange(number_of_phases, 0, -1):
            hdr.append(u'# Phase {0:d}\n'.format(phase_num))
            hdr.append(u'# MaterialName  \t{0}\n'.format(phase_out[phase_num-1]))
            hdr.append(u'# Formula     \t{0}\n'.format(formula_out[phase_num-1]))
            hdr.append(u'# Info \t\t\n')
            hdr.append(u'# Symmetry              {0:d}\n'.format(\
                                              phase_symmetry_out[phase_num-1]))
            hdr.append(u'# LatticeConstants      \
               {0:5.3f} {1:5.3f} {2:5.3f}  {3:7.3f}  {4:7.3f}  {5:7.3f}\n'.format(\
                        lattice_constants_out[phase_num-1][0], \
                        lattice_constants_out[phase_num-1][1], \
                        lattice_constants_out[phase_num-1][2], \
                        lattice_constants_out[phase_num-1][3], \
                        lattice_constants_out[phase_num-1][4], \
                        lattice_constants_out[phase_num-1][5]))
            hdr.append(u'# NumberFamilies        {0}\n'.format(\
                                            num_hkl_families_out[phase_num-1]))
            
            hklfam = hkl_families_out[phase_num-1]
            hkl4 = hkl_4_col_out[phase_num-1]
            hkl5 = hkl_5_col_out[phase_num-1]
            hklfin = hkl_last_col_out[phase_num-1]
            tmpstr = u'# hklFamilies   \t{0: >2d} {1: >2d} {2: >2d} '+\
                        '{3:d} {4:.6f} {5:d}\n'
            for i in np.arange(0, num_hkl_families_out[phase_num-1]):
                hdr.append(tmpstr.format(hklfam[i,0], hklfam[i,1], hklfam[i,2],\
                                    hkl4[i], hkl5[i], hklfin[i]))
    
            # Shiveley puts this only for v6, but it seems to also be output
            # for v7. We will write dummy numbers for now.
            if start_phase_location == 1988 or start_phase_location == 1992:
                for i in np.arange(0, 6):
                    hdr.append(u'# ElasticConstants \t-1.000000 -1.000000 \
                               -1.000000 -1.000000 -1.000000 -1.000000\n')
            
            # Writing dummy numbers...
            hdr.append(u'# Categories0 0 0 0 0 \n')
            hdr.append(u'#\n')
        
        hdr.append(u'# GRID: {0}\n'.format(grid))
        hdr.append(u'# XSTEP: {0:.6f}\n'.format(xstep))
        hdr.append(u'# YSTEP: {0:.6f}\n'.format(ystep))
        hdr.append(u'# NCOLS_ODD: {0:d}\n'.format(ncols_odd.astype(np.int)))
        hdr.append(u'# NCOLS_EVEN: {0:d}\n'.format(ncols_even.astype(np.int)))
        hdr.append(u'# NROWS: {0:d}\n'.format(nrows.astype(np.int)))
        hdr.append(u'#\n')
        hdr.append(u'# OPERATOR: \t{0}\n'.format(operator_name))
        hdr.append(u'#\n')
        hdr.append(u'# SAMPLEID: \t{0}\n'.format(sample_ID))
        hdr.append(u'#\n')
        hdr.append(u'# SCANID: \t{0}\n'.format(scan_ID))
        hdr.append(u'#\n')
        ebsd_data._anghdr = "".join(hdr)
        
        return ebsd_data
        
    # This is it.
    print 'Elapsed time = {0}'.format(time.clock() -t)
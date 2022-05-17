path = 'C:\Users\ashle\Downloads\opDict.mat'
opw = load(onp_path).op_w;
new_ori = orientation.byEuler(opw,MDF.CS)
qq=inv(new_ori(1))*(new_ori(2));
isMisorientation(qq)
II=orientation(idquaternion, MDF.CS);
miso=inv(II)*(new_ori);
opw_w = eval(MDF, miso);
addpath('C:\noy\academics\autonomous driving\devkit_raw_data\devkit\matlab')

fid = fopen('C:\noy\academics\autonomous driving\proj1\2011_09_26_drive_0106_sync\2011_09_26\2011_09_26_drive_0106_sync\velodyne_points\data\0000000000.bin','rb');
PCtmp = fread(fid,'float');
PC=reshape( PCtmp,4,length( PCtmp)/4)';
function [ landmarks ] = readLandmarks( filename )
%READLANDMARKS Summary of this function goes here
%   Detailed explanation goes here

fileID = fopen( filename );
%C = textscan(fileID,'%f %f','headerLines', 1);
C = textscan(fileID,'%f','CommentStyle','%');
fclose(fileID);
landmarks = cell2mat(C);

end


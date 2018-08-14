clc;
clear;
close all;
fprintf('Bonus 1 for Historical and Modern image is running...\n\n')
datafolder = 'A:\COURSE MATERIAL\Winter-2018\computer vision\Assignment-2\data'
dataDir = fullfile(datafolder);
BonusImg = imageSet(dataDir);
montage(BonusImg.ImageLocation);
Historical = read(BonusImg,1); %assign the first image to variable Historical;
Modern = read(BonusImg,2); %assign the 2nd image to variable Modern;
I1 = rgb2gray(Historical);
B1 = rgb2gray(Modern);
points1 = detectHarrisFeatures(I1); %finds the corners
points2 = detectHarrisFeatures(B1);
[features1, valid_points1] = extractFeatures(I1,points1);
[features2 valid_points2] = extractFeatures(B1,points2);
indexPairs = matchFeatures(features1,features2);
matchedPoints1 = valid_points1(indexPairs(:,1),:); 
matchedPoints2 = valid_points2(indexPairs(:,2),:);
figure; showMatchedFeatures(I1,B1,matchedPoints1,matchedPoints2);
title("combined image");
blender = vision.AlphaBlender('Operation', 'Binary mask', ...
  'MaskSource', 'Input port');
 
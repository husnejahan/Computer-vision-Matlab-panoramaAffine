clc;
clear;
close all;
fprintf('Bonus 1 for day light chnged image is running...\n\n')
datafolder = 'A:\COURSE MATERIAL\Winter-2018\computer vision\Assignment-2\Bonus'
dataDir = fullfile(datafolder);
BonusImg = imageSet(dataDir);
montage(BonusImg.ImageLocation);
daylight1 = read(BonusImg,1); 
daylight2 = read(BonusImg,2); 
I1 = im2single(daylight1);
B1 = im2single(daylight2);
figure; 
title("combined image");
halphablend = vision.AlphaBlender;
J = step(halphablend,I1,B1);imshow(J)
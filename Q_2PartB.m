clc;
clear;
fprintf('\nwait few sec Q-2 Part-B is running...\n');
Image1 = imread('Ryerson-left.jpg');
Image2 = imread('Ryerson-right.jpg');
panorama_homography(Image1,Image2);
function panoramaH=panorama_homography(I1,I2)
[h1, w1, ~] = size(I1);
[h2, w2, ~] = size(I2);

gray1 = double(rgb2gray(I1));
gray2 = double(rgb2gray(I2));

sigma = 5;
threshold = 2000;
radius = 3;
disp = 0;
match_count = 200;
neighbor_rad = 20; 

[left_cim,r1,c1] = harriscorner(gray1,sigma,threshold,radius,disp);
[right_cim,r2,c2] = harriscorner(gray2,sigma,threshold,radius,disp);

descriptor_img1 = neighborhd(gray1, neighbor_rad, r1, c1);
descriptor_img2 = neighborhd(gray2, neighbor_rad, r2, c2);

distances = sqrt_dist(descriptor_img1,descriptor_img2);
[~,distance_indices] = sort(distances(:),'ascend');
matches = distance_indices(1:match_count);
[dist_r, dist_c] = ind2sub(size(distances), matches);
match_indices_1 = dist_r;
match_indices_2 = dist_c;

match_r1 = r1(match_indices_1);
match_c1 = c1(match_indices_1);
match_r2 = r2(match_indices_2);
match_c2 = c2(match_indices_2);

img1_hom = [match_c1, match_r1, ones(match_count,1)];
img2_hom = [match_c2, match_r2, ones(match_count,1)];
[H, inliers] = homography(img1_hom,img2_hom);

match_c1 = match_c1(inliers);
match_c2 = match_c2(inliers);
match_r1 = match_r1(inliers);
match_r2 = match_r2(inliers);

tform = maketform('projective', H);
img1_result = imtransform(I1, tform);

stitched_image= stitch(I1, I2, H);
imshow(stitched_image);title('Final image')

end
function [cm, row, col] = harriscorner(im, sigma, thresh, radius, disp)
    
    error(nargchk(2,5,nargin));
    
    dx = [-1 0 1; -1 0 1; -1 0 1]; % Derivative masks
    dy = dx';
    
    Ix = conv2(im, dx, 'same');    % Image derivatives
    Iy = conv2(im, dy, 'same');    
    g = fspecial('gaussian',max(1,fix(6*sigma)), sigma);
    
    Ix2 = conv2(Ix.^2, g, 'same'); % Smoothed squared image derivatives
    Iy2 = conv2(Iy.^2, g, 'same');
    Ixy = conv2(Ix.*Iy, g, 'same');
    
    cm = (Ix2.*Iy2 - Ixy.^2)./(Ix2 + Iy2 + eps); % Harris corner measure

    if nargin > 2   % perform nonmaximal suppression and threshold
	sze = 2*radius+1;                   % Size of mask.
	mx = ordfilt2(cm,sze^2,ones(sze)); % Grey-scale dilate.
	cm = (cm==mx)&(cm>thresh);       % Find maxima.
	
	[row,col] = find(cm);                  % Find row,col coords.
	
	if nargin==5 & disp      % overlay corners on original image
	    figure, imagesc(im), axis image, colormap(gray), hold on
	    plot(col,row,'ys'), title('corners detected');
	end
    
    else 
	row = []; col = [];
    end
end
function [descriptor] = neighborhd(img,radius,r,c)
    feature_count = length(r);
    descriptor = zeros(feature_count, (2 * radius + 1)^2);
    pad = zeros(2 * radius + 1); 
    pad(radius + 1, radius + 1) = 1;
    padded_img = imfilter(img, pad, 'replicate', 'full');
    for i = 1 : feature_count
        rows = r(i) : r(i) + 2 * radius;
        cols = c(i) : c(i) + 2 * radius;
        neighbor = padded_img(rows, cols);
        vect_feat = neighbor(:);
        descriptor(i,:) = vect_feat;
    end
    descriptor = zscore(descriptor')';
end
function dist_value = sqrt_dist(x, c)
% Calculates squared distance between two sets of points.
    [ndata, dimx] = size(x);
    [ncentres, dimc] = size(c);
    if dimx ~= dimc
        error('Data dimension does not match dimension of centres')
    end

    dist_value = (ones(ncentres, 1) * sum((x.^2)', 1))' + ...
      ones(ndata, 1) * sum((c.^2)',1) - ...
      2.*(x*(c'));

    % Rounding errors occasionally cause negative entries in n2
    if any(any(dist_value<0))
      dist_value(dist_value<0) = 0;
    end
end
function [ H, inliers ] = homography( p1, p2 )
    epochs = 200;
    point_count = 4;
    thresh_dist = 10;
    thresh_inlier = .2;
    
    [match_count, ~] = size(p1);
    inlier_count = zeros(epochs,1);
    H_store = {};
    
    for i = 1 : epochs
        subsetIndices = randsample(match_count, point_count);
        p1_sel = p1(subsetIndices, :);
        p2_sel = p2(subsetIndices, :);
        model = fit_hmy(p1_sel, p2_sel);
        projections = projection(model, p1, p2);
        inliers = find(projections < thresh_dist);      
        inlier_count(i) = length(inliers);
        inlier_ratio = inlier_count(i)/match_count;
        if inlier_ratio >=  thresh_inlier
            x_inliers = p1(inliers, :);
            y_inliers = p2(inliers, :);
            H_store{i} = fit_hmy(x_inliers, y_inliers);
        end
    end
    iter_opt = find(inlier_count == max(inlier_count));
    iter_opt = iter_opt(1);
    H_opt = H_store{iter_opt};
    projections = projection(H_opt, p1, p2);
    inliers = find(projections < thresh_dist);
    H = H_opt;
end
function HF = fit_hmy(p1_hom, p2_hom)
    [match_count, ~] = size(p1_hom);
    A = [];
    for i = 1:match_count
        p1 = p1_hom(i,:);
        p2 = p2_hom(i,:);
        A_i = [ zeros(1,3),-p1,p2(2)*p1;p1,zeros(1,3),-p2(1)*p1];
        A = [A; A_i];        
    end
    [u,s,v] = svd(A);
    h = v(:,9);
    HF = reshape(h, 3, 3);
    HF = HF ./ HF(3,3);
    end
function proj = projection(H, p1_hom, p2_hom)
    projected = p1_hom * H;
    lambda_t =  projected(:,3);
    lambda_2 = p2_hom(:,3);
    p1_dist = projected(:,1) ./ lambda_t - p2_hom(:,1) ./ lambda_2;
    p2_dist = projected(:,2) ./ lambda_t - p2_hom(:,2) ./ lambda_2;
    proj = p1_dist.^2 + p2_dist.^2;
end
function [stitched_image] = stitch(im1, im2, H)

    [h1, w1, color1] = size(im1);
    [h2, w2, color2] = size(im2);
    
    vertices = [ 1 1 1;
                w1 1 1;
                w1 h1 1;
                1 h1 1];
    
    homoCoord = vertices * H;
    dimension = size(homoCoord) - 1;
    normCoord = bsxfun(@rdivide,homoCoord,homoCoord(:,end));
    new_vertices = normCoord(:,1:dimension);        

    minX = min( min(new_vertices(:,1)), 1);
    maxX = max( max(new_vertices(:,1)), w2);
    minY = min( min(new_vertices(:,2)), 1);
    maxY = max( max(new_vertices(:,2)), h2);

    xResRange = minX : maxX;
    yResRange = minY : maxY;

    [x,y] = meshgrid(xResRange,yResRange) ;
    Hinv = inv(H);

    warpedHomoScaleFactor = Hinv(1,3) * x + Hinv(2,3) * y + Hinv(3,3);
    warpX = (Hinv(1,1) * x + Hinv(2,1) * y + Hinv(3,1)) ./ warpedHomoScaleFactor ;
    warpY = (Hinv(1,2) * x + Hinv(2,2) * y + Hinv(3,2)) ./ warpedHomoScaleFactor ;


    if color1 == 1
        bLH = interp2( im2double(im1), warpX, warpY, 'cubic') ;
        bRH = interp2( im2double(im2), x, y, 'cubic') ;
    else
        bLH = zeros(length(yResRange), length(xResRange), 3);
        bRH = zeros(length(yResRange), length(xResRange), 3);
        for i = 1:3
            bLH(:,:,i) = interp2( im2double( im1(:,:,i)), warpX, warpY, 'cubic');
            bRH(:,:,i) = interp2( im2double( im2(:,:,i)), x, y, 'cubic');
        end
    end
    bWeight = ~isnan(bLH) + ~isnan(bRH) ;
    bLH(isnan(bLH)) = 0 ;
    bRH(isnan(bRH)) = 0 ;
    stitched_image = (bLH + bRH) ./ bWeight ;
end

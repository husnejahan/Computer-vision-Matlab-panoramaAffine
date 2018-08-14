clc;
clear;
%%2.1
fprintf('\nwait few sec Q-2 partA is running...\n');
Image1 = imread('parliament-left.jpg');
Image2 = imread('parliament-right.jpg');
Panorama_Affine(Image1,Image2);

function panoramaAffine=Panorama_Affine(im_left,im_right)
left = im2single(rgb2gray(im_left));
right = im2single(rgb2gray(im_right));

%%%2.2
% f(1:2) -> centre x,y
[f1,d1] = vl_sift(left);
[f2,d2] = vl_sift(right);
%%%2.3
% matches f1, f2 and scores.
[matches, scores] = vl_ubcmatch(d1, d2);
%%2.4
filtered_match = zeros(size(matches,2), 2);
index = 1;
for i = 1:size(matches,2)
    if(scores(i) < 100) % Euclidean distances between SIFT descriptors are less than 100.
        filtered_match(index,1) = matches(1,i);
        filtered_match(index,2) = matches(2,i);
        index = index + 1;
    end
end
filtered_match = filtered_match(1:index-1,:);
%%2.5
best_fit_est = [0 0 0 0]; 
selected_inliners = [];

for N = 1:200 % Let's repeat 200 times.
    ransac_pts = randperm(size(filtered_match,1),3); %pick three unique random numbers.
    % Get the coordinates.
    p1f1 = f1(1:2, filtered_match(ransac_pts(1),1));
    p1f2 = f2(1:2, filtered_match(ransac_pts(1),2));
    p2f1 = f1(1:2, filtered_match(ransac_pts(2),1));
    p2f2 = f2(1:2, filtered_match(ransac_pts(2),2));
    p3f1 = f1(1:2, filtered_match(ransac_pts(3),1));
    p3f2 = f2(1:2, filtered_match(ransac_pts(3),2));
    % Estimate the affine transformation
    A = [p1f1(2),p1f1(1),0,0,1,0; ...
        0,0,p1f1(2),p1f1(1),0,1; ...
        p2f1(2),p2f1(1),0,0,1,0; ...
        0,0,p2f1(2),p2f1(1),0,1; ...
        p3f1(2),p3f1(1),0,0,1,0; ...
        0,0,p3f1(2),p3f1(1),0,1];
    b = [p1f2(2); p1f2(1); p2f2(2); p2f2(1); p3f2(2); p3f2(1)];
    x = A\b;
    % Now, we have our unknowns.
    T = [x(1) x(2); x(3) x(4)];
    c = [x(5); x(6)];
    
   
    % threshold p.
    p = 0.03;
    inliners = [];
    for i = 1 : size(filtered_match,1)
        if (ismember(i,ransac_pts) == 0) 
            tmp_pix = f1(1:2, filtered_match(i,1));
            pix = [tmp_pix(2); tmp_pix(1)];
            pix_map = T*pix+c; 
            
            tmp_orig = f2(1:2, filtered_match(i,2)); 
            
            % Calculate Euclidean Distance between two points.
            D = pdist([pix_map(2) pix_map(1); tmp_orig(1) tmp_orig(2)],'euclidean');
            
            % If distance is less than p then, it is inliner.
            if (D < p)
                inliners = [inliners; i];
            end
        end
    end
 
    % It is the best estimate, if it has the most number of inliners.
    if (best_fit_est(1) < size(inliners,1))
        selected_inliners = inliners;
        best_fit_est = [size(inliners,1) ransac_pts(1) ransac_pts(2) ransac_pts(3)];
    end
end
%%2.6

% Now we have the best estmate points and we can grap the transformation.
ransac_pts = [best_fit_est(2) best_fit_est(3) best_fit_est(4)];

A = [];
b = [];

for i = 1: size(selected_inliners,1)
    % Get the coordinates.
    p1f1 = f1(1:2, filtered_match(selected_inliners(i),1));
    p1f2 = f2(1:2, filtered_match(selected_inliners(i),2));
    
    % Estimate the affine transformation
    A = [A; ...
        p1f1(2),p1f1(1),0,0,1,0; ...
        0,0,p1f1(2),p1f1(1),0,1];
    
    b = [b; p1f2(2); p1f2(1)];
end

x = A\b;

% Now, we have our unknowns.
T = [x(1) x(2); x(3) x(4)];
c = [x(5); x(6)];
%%%2.7
% Affine transformation.
T = maketform('affine', [x(1), x(2), 0; x(3), x(4), 0; x(5), x(6), 1]);
left_t = imtransform(left, T);
intersection = 1160; % manual padding for better looking.

composition = [ left_t zeros(size(left_t,1),intersection) ]; 
right_t = [ zeros(200, size(right,2)); right ]; 
right_window = size(right,2) - intersection; % intersecting window size.
left_window = size(composition,2) - intersection;
intersect_pt = left_window-right_window-1; 
composition(1:2600, left_window:end) = ...
    right_t(1:2600,right_window:end); 
composition(1:2600,intersect_pt:left_window) = ...
    composition(1:2600,intersect_pt:left_window)/2 + ...
    right_t(1:2600,1:right_window+2)/2;

%%Add colour
r1 = im_left(:,:,1); g1 = im_left(:,:,2); b1 = im_left(:,:,3);
r2 = im_right(:,:,1); g2 = im_right(:,:,2); b2 = im_right(:,:,3);

left_t = imtransform(r1, T);
img_comp_r = [left_t zeros(size(left_t,1),intersection)];
right_t = [zeros(200, size(r2,2)); r2];
img_comp_r(1:2600,left_window:end) = ...
    right_t(1:2600,right_window:end);
img_comp_r(1:2600,intersect_pt:left_window) = ...
    img_comp_r(1:2600,intersect_pt:left_window)/2 + ...
    right_t(1:2600,1:right_window+2)/2;

left_t = imtransform(g1, T);
img_comp_g = [left_t zeros(size(left_t,1),intersection)];
right_t = [zeros(200, size(g2,2)); g2];
img_comp_g(1:2600,left_window:end) = ...
    right_t(1:2600,right_window:end);
img_comp_g(1:2600,intersect_pt:left_window) = ...
    img_comp_g(1:2600,intersect_pt:left_window)/2 + ...
    right_t(1:2600,1:right_window+2)/2;

left_t = imtransform(b1, T);
img_comp_b = [left_t zeros(size(left_t,1),intersection)];
right_t = [zeros(200, size(b2,2)); b2];
img_comp_b(1:2600,left_window:end) = ...
    right_t(1:2600,right_window:end);
img_comp_b(1:2600,intersect_pt:left_window) = ...
    img_comp_b(1:2600,intersect_pt:left_window)/2 + ...
    right_t(1:2600,1:right_window+2)/2;

imcolor = cat(3,img_comp_r,img_comp_g,img_comp_b);

imshow(imcolor);title('Final image with colour')
end

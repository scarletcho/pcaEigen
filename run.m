% Intro to Artificial Neural Network (BRI516)
% Assignment 1 (by. Yejin Cho)
% 2016-03-22
clc;clear;close all;

%% (a)
% Generate 100¡¿1000 matrix A0 with random numbers from the Gaussian
% distribution (zero-mean and unit variance) as its elements. [5 pt]
A0 = zscore(normrnd(0,1,[100,1000]));
histfit(A0(:)); axis auto
fprintf('A0 mean: %.2f, var: %.2f\n', mean(A0(:)),var(A0(:)));

%% (b)
% Generate two sinusoidal functions (v1 and v2) of total length
% of 100 samples with periods of 10 and 50 samples.
% v1 and v2 have a zero-mean and unit variance. [5 pt]
Nsamples = 100;

% (1) v1
v1_periods = 10;
v1_samples = linspace(0,2*pi*(Nsamples/v1_periods),Nsamples);
v1 = zscore(sin(v1_samples));
fprintf('v1 mean: %.2f, var: %.2f\n', mean(v1),var(v1));

subplot(2,1,1)
plot(v1, 'o-');
ylim([-3 3]); title('v1 (of 10 samples period)')


% (2) v2
v2_periods = 50;
v2_samples = linspace(0,2*pi*(Nsamples/v2_periods),Nsamples);
v2 = zscore(sin(v2_samples));
fprintf('v2 mean: %.2f, var: %.2f\n', mean(v2),var(v2));

subplot(2,1,2)
plot(v2, 'o-')
ylim([-3 3]); title('v2 (of 50 samples period)'); shg


clearvars -except A0 v1 v2

%% (c)
% For each of the column vectors of A0,
% add v1 for the first 500 column vectors of A0
% and v2 for the remaining 500 column vectors of A0
% with randomly chosen (0-1) weights for v1 and v2.
% The resulting matrix is A1. [10 pt]

A1 = zeros(size(A0,1),size(A0,2));  % pre-allocation
rdCoeff= rand(1,2); % two random weights between 0-1

A1(:,1:500) = A0(:,1:500) ...
    + repmat(rdCoeff(1)*v1',[1,500]); % add v1 multiplied by a random weight (0-1)
A1(:,501:end) = A0(:,501:end) ...
    + repmat(rdCoeff(2)*v2',[1,500]); % add v2 multiplied by a random weight (0-1)

fprintf('random weight of v1: %.2f\nrandom weight of v2: %.2f\n', rdCoeff(1), rdCoeff(2))

%% (d)
% Perform a PCA to A1 and find a 2D representation of column vectors of A1
% using the first and second principal components.
% Show the 2D representation in a 2D plane. [10 pt]

% Perform a PCA
[Y, eigVec, eigVal] = pcaEigen(A1);

% Plot using 1st & 2nd Principal Components
figure; subplot(1,2,1)
plot(Y(:,1),Y(:,2),'o')
xlabel('1st Principal Component')
ylabel('2nd Principal Component')
title('A1 projected onto 2D plane of 1st-2nd PC')
xlim([-50 50]); ylim([-50 50]); axis square; hold on


%% (e) 
% Plot two encoding vectors of this dimension reduction. [5 pt]

encodingVec1 = [1 0]*eigVal(1)^.5;
encodingVec2 = [0 1]*eigVal(2)^.5;

PC12 = plot(encodingVec1,[0 0],'r',[0 0],encodingVec2,'g');
legend(PC12, '1st PC','2nd PC')
grid minor

%% (f)
% Repeat (d) and (e) using the third and fourth principal components. [15 pt]

% Since Y is already fully computed in (d) above,
% the encoding process doesn't have to be repeated.
% We can plot using 3rd & 4th Principal Components in Y.

subplot(1,2,2)
plot(Y(:,3),Y(:,4),'o')
xlabel('3rd Principal Component')
ylabel('4th Principal Component')
title('A1 projected onto 2D plane of 3rd-4th PC')
xlim([-50 50]); ylim([-50 50]); axis square; hold on

encodingVec3 = [1 0]*eigVal(3)^.5;
encodingVec4 = [0 1]*eigVal(4)^.5;

PC34 = plot(encodingVec3,[0 0],'r',[0 0],encodingVec4,'g');
legend(PC34, '1st PC','2nd PC')
grid minor

pause


%% [Additional visualization for understanding PCA]
% In this section, 7 additional plots are drawn
% to better visualize and help understanding the process of PCA.
% Unlike in the above, hese plots are in variable space.

figure('units','normalized','outerposition',[0 0 1 1])

% (1) Original data plot1: A0
subplot(7,1,1)
plot(A0); ylim([-10 10]); title('A0')

% (2) Original data plot: A1
subplot(7,1,2)
plot(A1); ylim([-10 10]); title('A1 (A0+v1 and A0+v2)')

% (3) Added vector plot: v1 and v2 plot as reference
subplot(7,1,3)
plot(v1); hold on
plot(v2); hold off
title('v1 and v2')


% PCA
[Y, eigVec, eigVal] = pcaEigen(A1);

% (4) PCA using 1st & 2nd PC only
subplot(7,1,4)
plot(Y(:,1:2)); axis normal
title('PCA score (of 1st and 2nd PC only)')
legend('PC1 & PC2 partially (or mostly) recovers the curvature of v1 and v2.',...
       'Location','best')

subplot(7,1,5)
plot(eigVec(:,1:2)); axis normal
title('PCA coefficients (of 1st and 2nd PC only)')
legend('The PCA coefficients of PC1 & PC2 effectively reveal the region in the data to which v1 and v2 was added',...
       'Location','best')


% (5) PCA using 3rd & 4th PC (added on the plot using the first two PCs)
subplot(7,1,6)
plot(Y(:,1:2)); hold on
plot(Y(:,3:4)); axis normal
title('PCA score (3rd and 4th PC added)')
legend('PC3 & PC4 does not provide much information as PC1 & PC2.',...
       'Location','best')

subplot(7,1,7)
plot(eigVec(:,1:2)); hold on
plot(eigVec(:,3:4), '.-'); axis normal
title('PCA coefficients (3rd and 4th PC added)')
legend('The PCA coefficients of PC3 & PC4 does not provide much information as in those of PC1 & PC2.',...
       'Location','best')

shg

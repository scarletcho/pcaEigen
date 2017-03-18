function [Y, eigVec, eigVal] = mypcaEigen(data)
% Intro to Artificial Neural Network (BRI516)
% mypcaEigen function for Assignment 1 (by. Yejin Cho)
% 2016-03-22
%% This function performs Principal Component Analysis by eigendecomposition.
% [ Input ] data   :   n-by-p input matrix (n rows of samples, p columns of variables)
% [ Output]   Y    :   encoded (reconstructed) data (or PCA scores)
%                      (c.f. Y(:,1:k) includes the encoded data projected on
%                            the first k principal components.)
%           eigVec :   eigenvectors of input data (or PCA coefficients)
%           eigVal :   eigenvalues of input data
%%
% (1) Subtract the mean of data (column-wise)
data = data - repmat(mean(data,1), size(data,1), 1);

% (2) Calculate the data's Covariance Matrix
covar = cov(data);

% (3) Eigendecomposition
[eigVec, eigVal] = eig(covar);
eigVal = flipud(diag(eigVal));  % Extract eigenvalues and flip upside-down
                                % to sort in descending order
eigVec = fliplr(eigVec);  % Flip eigenvectors left-right to match
                          % with the sorted eigVal

% (4) Encode data into Y (reconstructed data of reduced dimension)
Y = data*eigVec;
end
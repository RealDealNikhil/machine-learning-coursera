function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

% values of C and sigma we want to test together
param_vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
c_sigma = [];
for i = 1:length(param_vals),
    for j = 1:length(param_vals),
        c_sigma = [c_sigma ; param_vals(i) param_vals(j)];
    end
end

% get error values
for i = 1:length(c_sigma),
    inner_C = c_sigma(i, 1);
    inner_sigma = c_sigma(i, 2);
    model = svmTrain(X, y, inner_C, @(x1, x2) gaussianKernel(x1, x2, inner_sigma));
    predictions = svmPredict(model, Xval);
    error_vals(i) = mean(double(predictions ~= yval));
end

% get minimum error index
% use this to get best C and sigma
[min_err, min_idx] = min(error_vals);
C = c_sigma(min_idx, 1);
sigma = c_sigma(min_idx, 2);

% =========================================================================

end

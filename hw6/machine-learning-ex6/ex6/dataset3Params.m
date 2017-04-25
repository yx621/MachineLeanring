function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
CVec = [0.01 0.03 0.1 0.3 1 3 10 30];
sigmaVec = [0.01 0.03 0.1 0.3 1 3 10 30];

C = 1;
sigma = 0.1;

m = length(CVec);
n = length(sigmaVec);

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

initialError = 1000.00;

for iii = 1:m
    for jjj = 1:n
        currentC = CVec(iii);
        currentSigma = sigmaVec(jjj);
        model0 = svmTrain(X, y, currentC, @(x1, x2) gaussianKernel(x1, x2, currentSigma));
        predictions = svmPredict(model0, Xval);
        error = mean(double(predictions ~= yval));
        if (error < initialError)
            C = currentC;
            sigma = currentSigma;
            initialError = error;
        endif
    end
end

% =========================================================================

end

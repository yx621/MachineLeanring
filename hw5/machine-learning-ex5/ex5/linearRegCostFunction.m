function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% X = [ones(m, 1) X];
% disp("*****************")
% disp("*****************")
% size(X, 1)
% size(X, 2)
% size(theta, 1)
% size(theta, 2)
% size(y, 1)
% size(y, 2)
% disp("*****************")
% disp("*****************")

interMediateResult = X*theta;

% a1 = 1./(1 + exp(-interMediateResult));
a1 = interMediateResult - y;

J = sum(a1.*a1)/2/m;
% this should be a number

theta4Reg = theta(2:end);
J = J + lambda*sum(theta4Reg.*theta4Reg)/2/m;

% =========================================================================

grad = grad(:);

end

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

interResult = X*theta;

% a1 = 1./(1 + exp(-interResult));

a1 = interResult - y;

J = J + transpose(a1)*a1;

thetaInUse = theta(2:end);

J = J + lambda*transpose(thetaInUse)*thetaInUse;

J = J/2/m;

theta4Grad = [0; theta(2:end)];

sum0 = sum(a1.*X)/m;
grad = sum0 + lambda/m*transpose(theta4Grad);

% =========================================================================

grad = grad(:);

end

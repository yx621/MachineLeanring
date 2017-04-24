function g = sigmoid(z)
% the input z should be mX1 vector 

%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

exp1 = exp(-z);
dinominator = 1 + exp1;

g = 1./dinominator;

% output is also mX1 vector 

% =============================================================

end

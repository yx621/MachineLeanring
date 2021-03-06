function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% disp("************************");
% disp("************************");
% disp("************************");
% size(y, 1)
% size(y, 2)
% disp("************************");
% disp("************************");
% disp("************************");


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
X = [ones(m, 1) X];
% Use sum to get J 
for c = 1:m
    sum0 = 0;
    yi = eye(num_labels)(y(c),:);
    result1 = X(c,:)*(transpose(Theta1));
    a1 = 1./(1 + exp(-result1));
    a1 = [1 a1];
    result2 = a1*(transpose(Theta2));
    a2 = 1./(1+exp(-result2));
    sum0 = 1/m*(sum(-yi.*log(a2) - (1-yi).*log(1 - a2)));
    J = J + sum0;
    end

% add the regularization part 
theta1Reg = Theta1(:,2:end);
theta2Reg = Theta2(:,2:end);

J_reg = 0;
for c = 1:size(theta1Reg,1)
    J_reg = J_reg + sum(theta1Reg(c,:).*theta1Reg(c,:));
end

for c=1:size(theta2Reg, 1)
    J_reg = J_reg + sum(theta2Reg(c,:).*theta2Reg(c,:));
end

J_reg = lambda*J_reg/2/m;%
J = J + J_reg;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

for t = 1:m
    yi = eye(num_labels)(y(t),:);
    xi = X(t,:); %a1
    z2 = xi*transpose(Theta1);
    a2 = 1./(1 + exp(-z2));
    a2 = [1 a2];
    z3 = a2*transpose(Theta2);
    a3 = 1./(1 + exp(-z3));
    
    delta3 = a3 - yi;
    
    % disp("************************");
    % disp("************************");
    % size(delta3,1)
    % size(delta3,2)
    % size(Theta2, 1)
    % size(Theta2,2)
    % size(z2, 1)
    % size(z2, 2)
    % disp("************************");
    % disp("************************");
    z2 = [1 z2];
    delta2 = delta3*(Theta2).*sigmoidGradient(z2);
    delta2 = delta2(2:end);

    Theta1_grad += transpose(delta2)*(xi);
    Theta2_grad += transpose(delta3)*(a2);

    theta1Inuse = [zeros(size(Theta1), 1) Theta1(:,2:end)];
    theta2Inuse = [zeros(size(Theta2), 1) Theta2(:,2:end)];
end
    
    Theta1_grad += lambda*theta1Inuse;
    Theta2_grad += lambda*theta2Inuse;
    
    Theta2_grad = Theta2_grad/m;
    Theta1_grad = Theta1_grad/m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

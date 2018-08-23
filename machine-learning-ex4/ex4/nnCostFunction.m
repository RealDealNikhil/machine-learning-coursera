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

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% compute cost
% iterate over examples
% vectorize multi-classification cost for each example
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
%
% Part 3: Implement regularization with the cost function and gradients.

%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% TWO LAYER NEURAL NETWORK

label_vec = 1:num_labels;

% sum for cost
s = 0;
% Big Delta matrix with m rows, j cols, containing matrix for each layer
% Each matrix corresponds to Theta matrix
D_1 = zeros(size(Theta1));
D_2 = zeros(size(Theta2));
for i = 1:m,
    y_k = y(i) == label_vec;
    % forward prop
    X_i = [1 X(i,:)];
    a_2 = [1 sigmoid(X_i * Theta1')];
    h_i = sigmoid(a_2 * Theta2');
    % accumulate cost
    inner_s = y_k * log(h_i)' + (1 - y_k) * log(1 - h_i)';
    s += inner_s;
    % back prop
    % note we adjust d_2 for bias
    d_L = h_i - y_k;
    d_2 = (d_L * Theta2)(2:end) .* sigmoidGradient(X_i * Theta1');
    % accumulating gradients
    D_1 +=  d_2' * X_i;
    D_2 +=  d_L' * a_2;
end
J = - s / m;

% regularizing terms
% DO NOT include or regularize bias terms in cost calculation (1st columns of matrices)

y_k = zeros(m, num_labels);
% testing vectorized cost function
for i = 1:m,
    y_k(i,:) = y(i) == label_vec;
end

X = [ones(m, 1) X];
A = [ones(m,1) sigmoid(X * Theta1')];
H = sigmoid(A * Theta2');
size(H)
J = - sum((y_k .* log(H) + (1 - y_k) .* log(1 - H))(:)) / m;
reg = (lambda / (2 * m)) * sum((Theta1(:, 2:end) .^ 2)(:)) + sum((Theta2(:, 2:end) .^ 2)(:));
J += reg;


% regularizing gradients
% remember to not regularize the bias units (first column)
Theta1_grad = D_1 / m;
Theta1_grad(:, 2:end) += lambda * Theta1(:, 2:end) / m;
Theta2_grad = D_2 / m;
Theta2_grad(:, 2:end) += lambda * Theta2(:, 2:end) / m;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

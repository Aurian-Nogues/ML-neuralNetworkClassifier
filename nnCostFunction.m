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


% Setup some variables
m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%transform y matrix into a single value matrix
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);

%Forward propagation algorithm
a1 = [ ones(m,1), X ];
z2 = a1 * Theta1';
a2 = [ ones(m,1), sigmoid(z2)];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

%======================unregularized cost function ==================

%y' * h(x) works out of the box when they are vectors but here they are matrices. Could do a loop to extract K vector but this is slow.
%multiplying the matrices like that returns a lot of garbage outside the diagonal of the resulting matrices, trace does the job of summing diagonal
% to replicate sum k = 1 to num_labels while getting rid of the garbage terms outside the diagonal

J_temp = trace(-y_matrix' * log(a3)) - trace((1-y_matrix') * log(1 - a3));
J = 1/m * J_temp;

%======================cost function regularization==================

%exclude the first column of theta which is the biais unit

reg = lambda / (2 * m) * (sum(sum(Theta1(:, 2:end).^2))  + sum(sum(Theta2(:, 2:end).^2)));

J = J + reg;

%=================== gradient calculation =============

% first run a forward pass to calculate activation values through the network
% for each node j in layer l computer error term d that measures how much the node is responsible for any error in the output
% for output node we an directly calculate differenc between network's activation and true value because we know y so we know d3
% for hidden units compute dl based on weighted average of error terms of the nodes in layer l+1


% vectorised implementation so no loop

% output layer
d3 = a3 - y_matrix;

% hidden layer
sgz2 = sigmoidGradient(z2);
temp = d3 * Theta2(:, 2:end);
d2 = temp .* sgz2;

% input layer
d1 = d2' * a1;

Delta1 = d2' * a1;
Delta2 = d3' * a2;

Theta1_grad = Delta1 * 1/m;
Theta2_grad = Delta2 * 1/m;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

% ========= regularize gradient ===================

Theta1(:, 1) = 0;
Theta2(:, 1) = 0;

scaling_factor = lambda / m;

Theta1 = scaling_factor .* Theta1;
Theta2 = scaling_factor .* Theta2;

Theta1_grad = Theta1_grad + Theta1;
Theta2_grad = Theta2_grad + Theta2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

% ==================== test cases =======

% il = 2;              % input layer
% hl = 2;              % hidden layer
% nl = 4;              % number of labels
% nn = [ 1:18 ] / 10;  % nn_params
% X = cos([1 2 ; 3 4 ; 5 6]);
% y = [4; 2; 3];
% lambda = 4;
% [J grad] = nnCostFunction(nn, il, hl, nl, X, y, lambda)

% J = 19.474
% grad =
% .76614
% 0.97990
% 0.37246
% 0.49749
% 0.64174
% 0.74614
% 0.88342
% 0.56876
% 0.58467
% 0.59814
% 1.92598
% 1.94462
% 1.98965
% 2.17855
% 2.47834
% 2.50225
% 2.52644
% 2.72233
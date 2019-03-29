function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z

g = sigmoid(z) .* (1 - sigmoid(z));

end


% =================== test case ============
%   sigmoidGradient([[-1 -2 -3] ; magic(3)])
%   ans =
%     1.9661e-001  1.0499e-001  4.5177e-002
%     3.3524e-004  1.9661e-001  2.4665e-003
%     4.5177e-002  6.6481e-003  9.1022e-004
%     1.7663e-002  1.2338e-004  1.0499e-001
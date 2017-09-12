function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha
%fprintf('Plotting Data ...\n')
%data = load('ex1data1.txt');
%X = data(:, 1); y = data(:, 2);
%m = length(y);
%X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
%theta = zeros(2, 1);
%num_iters = 1500;
%alpha = 0.01;
% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
gradient=0;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
       
       %sum_m1 = sum_m1+(((theta(1)*X(row:row,1) + theta(2)*X(row:row,2))- y(row)));
       %sum_m2 = sum_m2+(((theta(1)*X(row:row,1) + theta(2)*X(row:row,2))- y(row))*(X(row:row,2)));
      h= X*theta;
      error= h-y; 
      gradient = X'*error;
      theta = theta - (alpha)*(gradient)*(1/m);
      
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
end

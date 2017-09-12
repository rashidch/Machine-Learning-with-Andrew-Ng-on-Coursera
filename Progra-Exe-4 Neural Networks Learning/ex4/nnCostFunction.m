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
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1: Feedforward the neural network  without Regularization && With
% Regularization 

        eye_matrix = eye(num_labels);
        y_matrix = eye_matrix(y,:);
        X = [ones(m, 1) X];
        a1= X;
        z2= Theta1*X';
        a2 = sigmoid(z2);
        m2= size(a2,2);
        one =ones(m2,1);
        %size(one');
        a2= [ one';a2 ];
        %size(a2);
        z3= Theta2*a2;
        a3= sigmoid(z3);
        
        h1=log(a3);
        y1= (-1)*y_matrix;
        j1= h1*y1;
        h2= log(1-a3);
        y2= 1-y_matrix;
        j2= h2*y2;
        j3= j1-j2;
        j4= eye(size(j3,1));
        j3= j3.*j4;
        j5= trace(j3);
        irc= j5/m;
%...................................
        Th1= Theta1(:,2:end);
        Th2= Theta2(:,2:end);
        rv1= sum(Th1.^2);
        rv2= sum(Th2.^2);
        j6= sum(rv1)+sum(rv2);
        j6;
        rc= (lambda/(2*m))*j6;
        J= rc+irc;
        
        
%%Part 2: Implement the backpropagation algorithm       
        %a1=a1(:,2:end);
        size(a1);
        
        %size(y_matrix)
       d3= a3- y_matrix';
       ml= Th2'*d3;
       d2= ml.*sigmoidGradient(z2);
       %size(d2)
       Delta1= d2*a1;
       
        %a2=a2(2:end,:);
       Delta2= d3*a2';
      
       D1= Delta1/m;
       D2 = Delta2/m;
       %grad=[Theta1_grad,Theta2_grad]
        
        Theta1(:,1) = 0 ;
        Theta2(:,1) = 0 ;
        RD1= (lambda/m)*(Theta1);
        
        RD2= (lambda/m)*(Theta2);

        Theta1_grad= D1+RD1;
        Theta2_grad= D2+RD2;







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

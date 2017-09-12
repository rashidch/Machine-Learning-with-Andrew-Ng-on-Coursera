function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

     %********cost (J)******
     h= sigmoid(X*theta);
     v1= (-1)*(y')*(log(h));
     v2= ((1-y)')*(log(1-h));
     v= v1-v2;
     ur=(v/m);
     theta(1)=0;
     r= lambda*(theta'*theta)/(2*m);
     J=ur+(r);
     %*************gradient (grad)**********
     V3= h-y;
     V4= X'*V3;
     ur_grad= V4/m;
     
     r_grad=((lambda/m))*theta;
     grad= ur_grad+r_grad;


% =============================================================

end

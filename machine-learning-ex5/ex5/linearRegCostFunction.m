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
J1 =0;
J2 = 0;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


J1 = sum(((X*theta) - y)'*((X*theta) - y));

J2 = sum(theta(2:length(theta)).^2);

J=J1/(2*m)+ ((lambda*J2)/(2*m)) ;

%-------------------------------------------------------------------

grad = sum(((X*theta)-y).*X)';
temp = theta; 
temp(1) = 0;   % because we don't add anything for j = 0  
temp = lambda*temp;
grad = grad + temp;
grad = (1/m).*grad;
 
%=========================================================================

grad = grad(:);

end

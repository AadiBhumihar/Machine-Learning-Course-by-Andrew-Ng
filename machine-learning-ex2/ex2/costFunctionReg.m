function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
J1 = 0;
J2 = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



J1 = sum(y.*log(sigmoid(X*theta))+(1-y).*log(1-sigmoid(X*theta))) ;

J2 = sum(theta(2:length(theta)).^2);


J=(-1)*J1/m + ((lambda*J2)/(2*m)) ;

% =============================================================

for ix = 1:length(grad)
         if (ix == 1)
		grad(ix) = sum((sigmoid(X*theta) - y).*X(:,ix))';
         else 
		grad(ix) = sum((sigmoid(X*theta) - y).*X(:,ix))' + lambda*theta(ix);
	 end
end

grad = (1/m).*grad;


% =============================================================

end

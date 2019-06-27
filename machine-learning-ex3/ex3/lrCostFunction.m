function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% cost function - summation
hyp = sigmoid(X*theta);
J = (J + -y .* log(hyp) - (1 - y) .* log(1 - hyp));
J = sum(J) / m + lambda/2/m * sum(theta(2:end).^2);

% gradient for each theta
grad(1) = 1/m * X(:,1)'*(hyp - y);
grad(2:end) = 1/m * X(:,2:end)'*(hyp - y) + lambda/m*theta(2:end);

end

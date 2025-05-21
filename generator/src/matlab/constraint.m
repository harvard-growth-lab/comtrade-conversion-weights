function [c,ceq,gc,gceq] = constraint(b,y,X)
% this function implements the nonlinear constraint (nonlcon) in fmincon ->
% inputs are b (the matrix of conversion weights, supplied by the
% minimization problem) see main file main_constrained_optimization.m
% The rows of matrix b need to sum up to one such that every product
% category has weights that sum up to one!

c = []; %no inequality constraint

temp = ones(size(y,2),1);
A = temp;
for i = 1:size(X,2)-1
    A = blkdiag(A,temp);
end

B = ones(1,size(X,2));

% size(b)
ceq = reshape(b',1,numel(b))*A-B;

if nargout > 2
    gc = []; %no gradient for inequality constraint

    gceq = A;
end

end
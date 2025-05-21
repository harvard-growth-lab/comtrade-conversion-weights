function [f, g, H] = OLSobj(b,y,X,conversion_mat)
% this function implements the OLS objective function in fmincon for the 
% optimization problem (OLS) -> see main file constrained_OLS.m
% Inputs: 
% b (the column vector of coefficients, supplied by the minimization
% problem), y (the dependent variable), X (the matrix containing
% explanatory variables)

fit = X*b(1:size(X,2),1:size(y,2));

f = sum(sum((y-fit).^2)); %objective function

if nargout > 1
    g = - 2.*X'*(y - fit); %FOC's
        g = reshape(g,numel(g),1);
end

if nargout > 2
    H_simple = 2.*(X'*X);
    H = H_simple;
    if size(y,2)>1
        for i = 1:size(y,2)-1
            H = blkdiag(H,H_simple);
        end
    end
            %Hessian (if we use fmincon and Algorithm "interior-point", the
            %main file (constrained_OLS.m) uses the function hessian_fct.m
            %or hessian_fct_con.m rather than this Hessian
end

end
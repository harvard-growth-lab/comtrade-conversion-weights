% Lukaszuk, P. & Torun, D. Harmonizing the Harmonized System SEPS Discussion Paper
% 2022-12 (2022)

function Hout = hessian_fct_con(b, lambda, y, X, country)
% this function implements the Hessian in fmincon for the CONSTRAINED
% optimization problem (OLS) -> see main file main_constrained_OLS.m
% Inputs: 
% b (the column vector of coefficients, supplied by the
% minimization problem), lambda is the inequality multiplier, y (the dependent
% matrix), X (the matrix containing explanatory variables)

H_simple = 2.*(X'*X);

H = H_simple;
if size(y,2)>1
    for i = 1:size(y,2)-1
        H = blkdiag(H,H_simple); %Hessian of objective function
    end
end
% Hessian of nonlinear inequality constraint
Hadd = zeros(size(X,2).*size(y,2),size(X,2).*size(y,2));

Hout = H + Hadd;

end
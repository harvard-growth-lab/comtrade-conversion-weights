% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% HS Conversion via Constrained OLS 

% Lukaszuk, P. & Torun, D. Harmonizing the Harmonized System SEPS Discussion Paper
% 2022-12 (2022)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

for group = groups

    clearvars -except groups group start_year end_year tol
    
    disp(' ')
    disp(strcat('Running group:', num2str(group)))

    current_dir = pwd;
    data_dir = fullfile(current_dir, '..', '..', 'data', 'matrices');

    filename = sprintf('conversion.matrix.start.%d.end.%d.group.%d.csv', start_year, end_year, group);
    filepath = fullfile(data_dir, filename);
    raw = readcell(filepath);

    raw=raw(2:end, 2:end);raw=string(raw);
    conversion_mat=raw=="True";

    filename = sprintf('source.trade.matrix.start.%d.end.%d.group.%d.csv', start_year, end_year, group);
    filepath = fullfile(data_dir, filename);
    raw = readcell(filepath);

    raw=raw(2:end, 2:end);
    old_trade_mat = zeros(size(raw));
    for r = 1:size(raw,1)
        for c = 1:size(raw,2)
            val = raw{r,c};
            if isempty(val)         % empty or missing
                old_trade_mat(r,c) = 0;      % treat as 0
            elseif isnumeric(val)   % already numeric
                old_trade_mat(r,c) = val;
            else                    % treat as text, convert to double
                tmp = str2double(val);
                if isnan(tmp)
                    old_trade_mat(r,c) = 0;  % unconvertible text => 0
                else
                    old_trade_mat(r,c) = tmp;
                end
            end
        end
    end
    old_trade_mat=old_trade_mat./sum(old_trade_mat(:));%normalize group trade in a year


    filename = sprintf('target.trade.matrix.start.%d.end.%d.group.%d.csv', start_year, end_year, group);
    filepath = fullfile(data_dir, filename);
    raw = readcell(filepath);

    raw=raw(2:end, 2:end);
    new_trade_mat = zeros(size(raw));
    for r = 1:size(raw,1)
        for c = 1:size(raw,2)
            val = raw{r,c};
            if isempty(val)         % empty or missing
                new_trade_mat(r,c) = 0;      % treat as 0
            elseif isnumeric(val)   % already numeric
                new_trade_mat(r,c) = val;
            else                    % treat as text, convert to double
                tmp = str2double(val);
                if isnan(tmp)
                    new_trade_mat(r,c) = 0;  % unconvertible text => 0
                else
                    new_trade_mat(r,c) = tmp;
                end
            end
        end
    end
    new_trade_mat=new_trade_mat./sum(new_trade_mat(:));%normalize group trade in a year


    conversion_mat=1./sum(conversion_mat>0,2).*(conversion_mat>0);%to check for pre-determined weights later

    % Specify dependent variable and explanatory variables
    y = new_trade_mat; 
    X = old_trade_mat; 

    bstart=conversion_mat;

    tic;

    for r = 1:size(conversion_mat,1)
        index_fixed(1,r) = ~any(conversion_mat(r,:)>0 & conversion_mat(r,:)<1);
            %%%identify origin codes that have a 1:1 link to a target code
            %%%(no need to estimate conversion weights for such origin
            %%%codes!) --> will be deducted from y-variable before running
            %%%the regression
    end

    y_reg = y - X(:,index_fixed)*conversion_mat(index_fixed,:);
    index_var = (index_fixed<1);
    X_reg=X(:,index_var);
    bstart=bstart(index_var,:); % --> DEDUCTED THIS!

    lb = zeros(size(bstart));
    ub = ones(size(bstart));

    conversion_mat_small = conversion_mat(index_var,:);

    for r = 1:size(bstart,1)
        for c = 1:size(bstart,2)
            if conversion_mat_small(r,c)==0
                lb(r,c)=0;
                bstart(r,c)=0;
                ub(r,c)=0;
            end
            if conversion_mat_small(r,c)==1
                lb(r,c)=1;
                bstart(r,c)=1;
                ub(r,c)=1;
            end
        end
    end

    big_mat=0;
    if numel(conversion_mat_small)>1e4
        big_mat = 1;%%if conversion matrix is very large
    end
    
    if big_mat~=1

        % % % OLS, fmincon, objective, gradient, and Hessian
        
        % Can take out all zero-rows in X
        X_reg_old = X_reg;
        y_reg_old = y_reg;
        index_zeros = sum(X_reg,2)~=0;%take out pairs with zero trade flows!
        X_reg = X_reg(index_zeros,:);
        y_reg = y_reg(index_zeros,:);
    
        OLSobjfct = @(b) OLSobj(b,y_reg,X_reg,conversion_mat_small);
        tic;
        fhes_con = @(b, lambda) hessian_fct_con(b, lambda, y_reg, X_reg); %Hessian for constrained problem
        nonlcon = @(b) constraint(b,y_reg,X_reg); %nonlinear constraint
    
        options = optimoptions('fmincon',... 
                                'Algorithm','interior-point',...    
                                'SpecifyObjectiveGradient',true,...  
                                'HessianFcn',fhes_con,...
                                'FunctionTolerance', tol,...
                                'StepTolerance', tol,...
                                'OptimalityTolerance', tol, ...
                                'Display','iter-detailed', ...
                                'UseParallel',false,...
                                'MaxIter',1e5,'MaxFunEvals',1e5); 
        [btildecon,aa_con,bb_con,output_con] = fmincon(OLSobjfct,bstart,...
                                       [],[],[],[],lb,ub,nonlcon,options);
        time_hes_con = toc;
    
        conversion_sol = conversion_mat;
        counter = 1;
        for r = 1:size(conversion_sol,1)
            if index_var(:,r) == 1
                conversion_sol(r,:) = btildecon(counter,:);  
                 counter = counter+1;
            end
        end
    
        btildecon = conversion_sol;

    else

        % Can take out all zero-rows in X
        X_reg_old = X_reg;
        y_reg_old = y_reg;
        index_zeros = sum(X_reg,2)~=0;
        X_reg = X_reg(index_zeros,:);
        y_reg = y_reg(index_zeros,:);
        X_reg = sparse(X_reg);%store as sparse matrix
        y_reg = sparse(y_reg);%store as sparse matrix

        OLSobjfct = @(b) OLSobj(b,y_reg,X_reg,conversion_mat_small);

        tic;
        K=size(conversion_mat_small,1);
        S=size(conversion_mat_small,2);
        Aeq = spalloc(K, K*S, K*S);
        beq = ones(K,1);  % each row sums to 1
        
        for i = 1:K
            colIndices = i : K : (S-1)*K + i;
            Aeq(i, colIndices) = 1;
        end

        options = optimoptions('fmincon',... 
                                'Algorithm','interior-point',...    
                                'SpecifyObjectiveGradient',true,...  
                                'HessianApproximation','lbfgs',...  'HessianFcn',fhes_con,...  
                                'FunctionTolerance', 1e-10,...
                                'StepTolerance', 1e-10,...
                                'OptimalityTolerance', 1e-10, ...
                                'Display','iter-detailed', ...
                                'UseParallel',false,...
                                'MaxIter',1500,'MaxFunEvals',1500); 
        [btildecon,aa_con,bb_con,output_con] = fmincon(OLSobjfct, bstart, ...
                                    [],[], Aeq, beq, lb, ub, [], options); %% [],[],[],[],lb,ub,nonlcon,options); %% 
        time_hes_con = toc;

        conversion_sol = conversion_mat;
        counter = 1;
        for r = 1:size(conversion_sol,1)
            if index_var(:,r) == 1
                conversion_sol(r,:) = btildecon(counter,:);  
                 counter = counter+1;
            end
        end

        btildecon = conversion_sol;

    end

    disp(' ')
    disp('Time used for optimization (in seconds):')
    disp(time_hes_con)

    weights_dir = fullfile(current_dir, '..', '..', 'data', 'conversion_weights');
    if ~exist(weights_dir, 'dir')
        mkdir(weights_dir);
    end

    filename = sprintf('conversion.weights.start.%d.end.%d.group.%d.csv', start_year, end_year, group);
    filepath = fullfile(weights_dir, filename);
    dlmwrite(filepath, btildecon, 'delimiter', ',', 'precision', 20);
    
end

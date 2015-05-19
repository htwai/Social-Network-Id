% This is the simulation file for the social network problem
% Written on 05.19.2015
clc; clear all; close all;

% The idea is to employ stubborn agents for inferring the unknown parts of
% the network.

% The simulation consists of two parts --> 1. generate the asymptotic
% opinion held by diff. users; 2. infer/regress B,D from the asymptotic opinion.

% Instead of using CVX for part 2, here we used a custom made Proximal, 
% Projected Gradient method for the task.

N_s_choice = 140 : 5 : 300;
no_mc = 1e2;

for nnn = 1 : length(N_s_choice)
%%%%%% System Parameters %%%%%%%%%%%%%%%%%%
N = 1000; % there are N agents (non-stubborn)
N_s = N_s_choice(nnn); % there are $N_s$ stubborn agents, which are probes that we can exploit
          % for simplicity, we assume that these stubborn agents form a
          % co-clique on their own.

p = 0.008; % the connectivity between the normal agents

d_s = 5; % constant degree

total_exp = 500; % no of experiments we are running, i.e., the constant K in the paper

for mc_sim = 1 : no_mc
    
fprintf('No of stubborn agents used: %i, MC Sim no: %i \n',N_s,mc_sim);    
% Generate the graph between the normal agents --> the network topology
% that I actually want to guess
G = rand(N,N) <= p; % correspond to the normal user
G = triu(G,1); G = G + G'; G = G > 0;

% count the actual number of non-zero entries in G (it's symmetric!)
fprintf('No. of present links : %i \n',sum(G(:)));

% Generate the topology from stubborn to normal agents 
G_sn = zeros(N_s,N);
for nn = 1 : N
    G_sn(randperm(N_s,d_s),nn) = 1;
end

% Combine them... (coz I want to use them as input to gossiping)
G_com = [zeros(N_s) G_sn; G_sn' G];

%%%%%% Part 1 of the simulation %%%%%%%%%%%
tmp_vec = 1 : N+N_s; 
init_op = rand(N+N_s,total_exp);

% the deterministic version %
SE_G = (G_com+eye(N+N_s)) .* rand(N+N_s); 

max_SEG = max(SE_G) + 0.01;
SE_G = SE_G - diag(diag(SE_G)) + diag(max_SEG); % self-trust is always higher
SE_G = SE_G ./ repmat(SE_G*ones(N_s+N,1),1,N_s+N);
SE_G(1:N_s,1:N_s) = eye(N_s); SE_G(1:N_s,N_s+1:end) = 0;

% asympt mixing
W_inf = SE_G^1e4; 
op_exp_result = W_inf*init_op;

%%%%%%% part 2 of the simulation --> regression on the data %%%%%%
% we want to infer B,D from the data

B_mask = vec( SE_G(N_s+1:end,1:N_s) > 0 );
BC_mask = vec( SE_G(N_s+1:end,1:N_s) == 0 );

D_true = SE_G(N_s+1:end,N_s+1:end);
B_true = SE_G(N_s+1:end,1:N_s);

% normalize D_true & B_true...
D_normalize = diag(1 ./ max(1e-10,(1 - diag(D_true)))) * D_true; 
D_normalize = D_normalize - diag(diag(D_normalize));
B_normalize = diag(1 ./ max(1e-10,(1 - diag(D_true)))) * B_true;

% Compute Y*pinv(Z)
YZ = op_exp_result(N_s+1:end,:)*((op_exp_result(1:N_s,:)*op_exp_result(1:N_s,:)')^-1*op_exp_result(1:N_s,:))';

%%%%%%%%%%%%%%%%% We use a projected gradient here... %%%%%%%%
lambda = 20000; % the penalty parameter 
D_i = zeros(N); B_i = zeros(N,N_s); % initialization with zero matrices
obj = norm( B_i - (eye(N)-D_i)*YZ, 'fro'); % initial objective
alpha = 0.1; % use a constant step size for the inner PG loop

tic; t_s = toc;
% we need subgradient update for lambda
for subgrad_iter = 1 : 2e2
    for pg_iter = 1 : 1e1
        % The projected, proximal gradient tries to minimize this:
        % min_{B,D \in C} lambda*||D||_1 + ||B-(I-D)X||_F^2
        D_old = D_i; B_old = B_i;
        % for B
        gB = (2*B_i - 2*(YZ-D_old*YZ));
        % projected gradient
        tB = max(0,B_i - alpha*gB); tB(BC_mask) = 0; B_i = tB;
        % for D
        gD = ( 2*D_i *(YZ*YZ') - 2*(YZ-B_old)*YZ' );
        % project it back...
        tD = D_i - alpha*gD; tD = tD - diag(diag(tD)); 
        % one sided proximal update
        tD = (tD>= (1/lambda)).*(tD - (1/lambda)); D_i = tD;
    end
    % Need to update lambda
    obj_fro = norm( B_i - (eye(N)-D_i)*YZ, 'fro')^2;
    lambda = lambda + (1e4)*norm( B_i - (eye(N)-D_i)*YZ, 'fro')^2;
end
t_f = toc;
D = D_i; B = B_i;

% print out the elapsed time
fprintf('Proximal Gradient Algorithm Run time: %f s \n',t_f - t_s);
%%%%%%%%%%% the algorithm ends here %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

MSE_D(nnn,mc_sim) = sum( sum( (D - D_normalize).^2 ) ) / sum(D_normalize(:).^2);

threshold = 0.01;
D_hat_binary = (D>=threshold);

SUPPORT_D(nnn,mc_sim) = sum(sum(abs(D_hat_binary - (D_normalize>0))));

SUPPORT_FAD(nnn,mc_sim) = sum(sum(max(0,D_hat_binary - (D_normalize>0))));
SUPPORT_MSD(nnn,mc_sim) = sum(sum(max((D_normalize>0)-D_hat_binary,0)));

MSE_B(nnn,mc_sim) = sum( sum( (B - B_normalize).^2 ) ) / sum(B_normalize(:).^2);

D_true_size(nnn,mc_sim) = sum(G(:));

fprintf('MSE in D: %f, MSE in B: %f, SUPPORT in D: %i \n',MSE_D(nnn,mc_sim),...
    MSE_B(nnn,mc_sim), SUPPORT_D(nnn,mc_sim) );

end

end

%% deal with the numerical issue... --> take away the NaN cases
mean_MD = mean(MSE_D');
mean_MB = mean(MSE_B'); 
mean_SD = mean(SUPPORT_D');
mean_FAD = mean(SUPPORT_FAD');
mean_MSD = mean(SUPPORT_MSD');

for iii = 1 : length(N_s_choice)
    idx_set = find( 1 - isnan(MSE_D(iii,:)) == 1 );
    mean_MD(iii) = mean(MSE_D(iii,idx_set));
    mean_MB(iii) = mean(MSE_B(iii,idx_set));
    mean_SD(iii) = mean(SUPPORT_D(iii,idx_set));
    mean_FAD(iii) = mean(SUPPORT_FAD(iii,idx_set));
    mean_MSD(iii) = mean(SUPPORT_MSD(iii,idx_set));
end
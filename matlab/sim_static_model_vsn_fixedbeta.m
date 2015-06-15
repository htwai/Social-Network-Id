% This is the simulation file for the social network problem
% Last Updated on 05.11.2015
clc; clear all; close all;
% The idea is to employ stubborn agents for inferring the unknown parts of
% the network.

% The simulation consists of two parts --> 1. generate the asymptotic
% opinion held by diff. users; 2. infer B,D from the asymptotic opinion.
N_choice = 20 : 5 : 100;
no_mc = 1e2;

for nnn = 1 : length(N_choice)
%%%%%% System Parameters %%%%%%%%%%%%%%%%%%
N = N_choice(nnn); % there are N agents (non-stubborn)

p = 0.08; % the connectivity between the normal agents

d_s = 4; % uniform degree for the Stubborn-Non-Stubborn graph

N_s = ceil(N*eval_b_fun(d_s,p)); % there are $N_s$ stubborn agents, which are probes that we can exploit
          % for simplicity, we assume that these stubborn agents form a
          % co-clique on their own.

% p_s = 0.1; % sparsity ER for the Stubborn-Normal graph

total_exp = 100; % no of experiments we are running

parfor mc_sim = 1 : no_mc
    
% fprintf('No of agents: %i, No of stubborn agents used: %i, MC Sim no: %i \n',N,N_s,mc_sim);    
% Generate the graph between the normal agents --> the network topology
% that I actually want to guess
G = rand(N,N) <= p; % correspond to the normal user
G = triu(G,1); G = G + G'; G = G > 0;

% count the actual number of non-zero entries in G (it's symmetric!)
% fprintf('No. of present links : %i \n',sum(G(:)));

% Generate the topology from stubborn to normal agents 
G_sn = zeros(N_s,N);
for nn = 1 : N
    G_sn(randperm(N_s,d_s),nn) = 1;
end

% gen G_sn by ER
% G_sn = rand(N_s,N) <= p_s;
G_com = [zeros(N_s) G_sn; G_sn' G]; % The augmented network with both stubborn and non-stubborn

%%%%%% Part 1 of the simulation %%%%%%%%%%%

tmp_vec = 1 : N+N_s; 
init_op = rand(N+N_s,total_exp); % just some random opinion

% the deterministic version %
SE_G = (G_com+eye(N+N_s)) .* rand(N+N_s); % SE_G is the \overline{W} 

max_SEG = max(SE_G) + 0.05;
SE_G = SE_G - diag(diag(SE_G)) + diag(max_SEG);
SE_G = SE_G ./ repmat(SE_G*ones(N_s+N,1),1,N_s+N);
SE_G(1:N_s,1:N_s) = eye(N_s); SE_G(1:N_s,N_s+1:end) = 0;

% asympt mixing
W_inf = SE_G^1e4; 
W_inf(:,N_s+1:end) = 0;
op_exp_result = W_inf*init_op;

%%%%%%% part 2 of the simulation --> regression on the data %%%%%%
% we want to infer B,D from the data
B_mask = vec( SE_G(N_s+1:end,1:N_s) > 0 );
BC_mask = vec( SE_G(N_s+1:end,1:N_s) == 0 ); % I know the support of B

D_true = SE_G(N_s+1:end,N_s+1:end);
B_true = SE_G(N_s+1:end,1:N_s);

% re-normalize D_true & B_true...
D_normalize = diag(1 ./ max(1e-10,(1 - diag(D_true)))) * D_true; 
D_normalize = D_normalize - diag(diag(D_normalize));
B_normalize = diag(1 ./ max(1e-10,(1 - diag(D_true)))) * B_true;

% cvx_quiet(true)
% cvx_solver('sedumi')
% cvx_begin
%     variable D(N,N) 
%     variable B(N,N_s)
%     minimize(  norm(D(:),1) );
%     subject to
%         % op_exp_result(N_s+1:end,:) corresponds to the final opinions of
%         % Non-stubborn agents, i.e., the matrix $Y$
%         % op_exp_result(1:N_s,:) corresponds to the final/initial opinions
%         % of stubborn agents, i.e., the matrix $Z$
%         op_exp_result(N_s+1:end,:) == D*op_exp_result(N_s+1:end,:) + B*op_exp_result(1:N_s,:);
%         D(:) >= 0; B(:) >= 0;
%         B(BC_mask) == 0;
%         diag(D) == 0;
%         ones(N,1) == [B D]*ones(N+N_s,1);
% cvx_end

% call for the cvx codes for parfor
[D,B] = solve_nsi_cvx(op_exp_result,BC_mask,N,N_s);


% re-normalize D & B...
D_hat_norm = diag(1 ./ (1 - diag(D))) * D; 
D_hat_norm = D_hat_norm - diag(diag(D_hat_norm));
B_hat_norm = diag(1 ./ (1 - diag(D))) * B;

MSE_D(nnn,mc_sim) = sum( sum( (D_hat_norm - D_normalize).^2 ) ) / sum(D_normalize(:).^2);

threshold = 0.5*min( D_normalize(D_normalize>0) );
D_hat_binary = (D_hat_norm>=threshold);
SUPPORT_D(nnn,mc_sim) = sum(sum(abs(D_hat_binary - (D_normalize>0))));

SUPPORT_FAD(nnn,mc_sim) = sum(sum(max(0,D_hat_binary - (D_normalize>0))));
SUPPORT_MSD(nnn,mc_sim) = sum(sum(max((D_normalize>0)-D_hat_binary,0)));

MSE_B(nnn,mc_sim) = sum( sum( (B_hat_norm - B_normalize).^2 ) ) / sum(B_normalize(:).^2);

D_true_size(nnn,mc_sim) = sum(G(:));

fprintf('No of stubborn agents used: %i, MC Sim no: %i\n No. of present links : %i\n MSE in D: %f, MSE in B: %f, SUPPORT Error in D: %i \n',...
    N_s,mc_sim,sum(G(:)),MSE_D(nnn,mc_sim),...
    MSE_B(nnn,mc_sim), SUPPORT_D(nnn,mc_sim));

end

end

%% deal with the numerical issue... --> take away the NaN cases
mean_MD = mean(MSE_D');
mean_MB = mean(MSE_B'); 
mean_SD = mean(SUPPORT_D');
mean_FAD = mean(SUPPORT_FAD');
mean_MSD = mean(SUPPORT_MSD');

for iii = 1 : length(N_choice)
    idx_set = find( 1 - isnan(MSE_D(iii,:)) == 1 );
    mean_MD(iii) = mean(MSE_D(iii,idx_set));
    mean_MB(iii) = mean(MSE_B(iii,idx_set));
    mean_SD(iii) = mean(SUPPORT_D(iii,idx_set));
    mean_FAD(iii) = mean(SUPPORT_FAD(iii,idx_set));
    mean_MSD(iii) = mean(SUPPORT_MSD(iii,idx_set));
end
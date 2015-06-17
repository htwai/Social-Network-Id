%% This is the simulation file for the social network problem
% Written on 06.16.2015
clc; clear all; close all;

% The simulation consists of two parts --> 1. generate the asymptotic
% opinion held by diff. users; 2. infer B,D from the asymptotic opinion.

% There is nothing to run monte-carlo simulation on though

s = RandStream('mt19937ar','Seed',0);

load Reed98_justA; % load the data file from Reed's College

%% Now the simulation

%%%%%% System Parameters %%%%%%%%%%%%%%%%%%
N = size(A,1);

% In the Reed's College data set, there are 962 individuals in the social
% network, so I choose ~150 most influential agents & mark them as my
% stubborn agents
N_s = 200; % there are $N_s$ stubborn agents, which are probes that we can exploit
          % for simplicity, we assume that these stubborn agents form a
          % co-clique on their own.

% d_s = 10;
% % Generate the topology from stubborn to normal agents 
% G_sn = zeros(N_s,N);
% for nn = 1 : N
%     G_sn(randperm(N_s,d_s),nn) = 1;
% end
% 
% G_com = zeros(N+N_s);
% G_com(N_s+1:end,:) = [G_sn' A];
% G = A;

% Retrieve $G$ by permutation
[degree_seq,idx] = sort( sum(A) , 'descend' );
G_com = A( [idx((N-N_s)/2 : (N+N_s)/2-1) setdiff(1:N,idx((N-N_s)/2 : (N+N_s)/2-1))],...
    [idx((N-N_s)/2 : (N+N_s)/2-1) setdiff(1:N,idx((N-N_s)/2 : (N+N_s)/2-1))] ); 
G_com(1:N_s,:) = 0;

G = G_com(N_s+1:end,N_s+1:end);
G_sn = G_com(N_s+1:end,1:N_s);


%% Next part

Ntotal = N;
%%%%%% Part 1 of the simulation %%%%%%%%%%%
total_exp = 2*N_s; % we run a total number of 2N_s experiments
init_op = rand(Ntotal,total_exp);

% we use the randomized broadcast gossip model!
% generate the "C" matrix used --> it only needs to be row stochastic
% C = ones(N+N_s,N+N_s); 
C = ones(Ntotal,Ntotal) / 2; 
C = (G_com+eye(Ntotal)).*C;
C(1:N_s,:) = 0; C(1:N_s,1:N_s) = eye(N_s);
% C = C ./ repmat( sum(C,2), 1, N+N_s);
% find the average W
W_bar = eye(Ntotal) - diag(C*ones(Ntotal,1)) / (Ntotal) + C / (Ntotal);

%% We can run the static model first... for comparison
W_inf = W_bar^1e4;
op_exp_result = (W_inf)*init_op;

D_true = W_bar(N_s+1:end,N_s+1:end); B_true = W_bar(N_s+1:end,1:N_s);

% normalize D_true & B_true...
D_normalize = diag(1 ./ max(1e-10,(1 - diag(D_true)))) * D_true; 
D_normalize = D_normalize - diag(diag(D_normalize));
B_normalize = diag(1 ./ max(1e-10,(1 - diag(D_true)))) * B_true;

BC_mask = vec( W_bar(N_s+1:end,1:N_s) == 0 );

Nt = Ntotal-N_s;
% Compute Y*pinv(Z)
YZ = op_exp_result(N_s+1:end,:)*((op_exp_result(1:N_s,:)*op_exp_result(1:N_s,:)')^-1*op_exp_result(1:N_s,:))'...
    + 0.00*randn(Nt,N_s);
% the last term is the noise observed

gamma = 0;
lambda = 1e8;

%%%%%%%%%%%%%%%%% We use a projected gradient here... %%%%%%%%
D_i = zeros(Nt); % initialization with zero matrices
B_i = zeros(Nt,N_s); 
obj = norm( B_i - (eye(Nt)-D_i)*YZ, 'fro'); % initial objective
alpha = 0.08; % use a constant step size for the inner PG loop

l_nesterov = 0; % nesterov step size

ratio_iter = zeros(1,100e2);
tD = zeros(Nt); tB = zeros(Nt,N_s);
for pg_iter = 1 : 100e2
    % The projected, proximal gradient tries to minimize this:
    % min_{B,D \in C} ||D||_1 + lambda*||B-(I-D)X||_F^2 + gamma*||B1 + D1 - 1||_2^2
    tD_old = tD; tB_old = tB;
    D_old = D_i; B_old = B_i;
    % for B
    gB = (2*B_i - 2*(YZ-D_old*YZ)) + 2*(gamma/lambda)*(D_old*ones(Nt,N_s)+B_old*ones(N_s)-ones(Nt,N_s));
    % projected gradient
    tB = max(0,B_i - alpha*gB); tB(BC_mask) = 0; 
    % for D
    gD = ( 2*D_i *(YZ*YZ') - 2*(YZ-B_old)*YZ' ) + 2*(gamma/lambda)*(D_old*ones(Nt)+B_old*ones(N_s,Nt)-ones(Nt));
    % project it back...
    tD = D_i - alpha*gD; tD = tD - diag(diag(tD));
%     D_i = tD;
    % one sided proximal update
    tD = (tD>= (1/lambda)).*(tD - (1/lambda)); 
    
    l_nestold = l_nesterov;
    l_nesterov = 0.5*(1 + sqrt(1 + 4*l_nesterov^2) );
    gam_nesterov = (1 - l_nestold) / l_nesterov;
    
    D_i = tD + gam_nesterov*(tD_old - tD);
    B_i = tB + gam_nesterov*(tB_old - tB);
    
    obj = norm( B_i - (eye(Nt)-D_i)*YZ, 'fro');
    % normalize the sum to 1
    sum_row = [D_i B_i]*ones(Ntotal,1);
    % normalize the row sum
    D = D_i ./ max(1e-15,repmat(sum_row,1,Nt)); B = B_i ./ max(1e-15,repmat(sum_row,1,N_s));
    ratio_iter(pg_iter) = sum( sum( (D - D_normalize).^2 ) ) / sum(D_normalize(:).^2);
%     sum( sum( (B - B_normalize).^2 ) ) / sum(B_normalize(:).^2)
end


plot(ratio_iter);

% % normalize the sum to 1
% sum_row = [D_i B_i]*ones(N+N_s,1);
% 
% % normalize the row sum
% D = D_i ./ max(1e-10,repmat(sum_row,1,Nt)); B = B_i ./ max(1e-10,repmat(sum_row,1,N_s));
% 
% sum( sum( (D - D_normalize).^2 ) ) / sum(D_normalize(:).^2)
% 
% sum( sum( (B - B_normalize).^2 ) ) / sum(B_normalize(:).^2)

% %% Next
% 
% G_noid = G_com;
% op_exp_result = zeros(N+N_s,total_exp);
% T0 = 1e3; no_gossip = 1e4; no_samples = 1e4-1e3-1;
% % Let's use the standard randomized gossip exchange...
% for eee = 1 : total_exp
%     x_op = init_op(:,eee);
%     
% %   % For pairwise gossip
% %     sample_instance = T0+randperm(s,no_gossip-(T0+1),no_samples);
% %     x_op_sample = zeros(N+N_s,length(sample_instance));
% %     cnt_sample = 1;
% %     
% % %     W_avg = zeros(N+N_s);
% %     % now, utilize the randomized gossip exchange...
% %     for gossip_round = 1 : no_gossip
% %         src_node = randi(N+N_s,1,1); % choose the node to wake up
% %         d_src = sum(G_com(:,src_node)); % find the degree of src
% %         random_integer = randi(d_src,1,1);
% %         dst_candidate = find(G_com(:,src_node) > 0);
% %         dst_node = dst_candidate(random_integer);
% %         
% %         if src_node > N_s
% %             x_op(src_node) = (x_op(src_node) + x_op(dst_node)) / 2;
% %         end
% %         if dst_node > N_s
% %             x_op(dst_node) = (x_op(dst_node) + x_op(src_node)) / 2;
% %         end
% %         
% % %         W_cur = eye(N+N_s);
% % %         W_cur(src_node,src_node) = 0.5; W_cur(dst_node,dst_node) = 0.5;
% % %         W_cur(dst_node,src_node) = 0.5; W_cur(src_node,dst_node) = 0.5;
% % %         W_avg = (1/gossip_round)*W_cur + (gossip_round-1)/gossip_round*W_avg;
% % 
% %         if ~isempty(find(sample_instance==gossip_round,1))
% %             % take samples of x_op & Mul_accu
% %             x_op_sample(:,cnt_sample) = x_op;
% %             cnt_sample = cnt_sample + 1;
% %         end
% %     end
% % %     W_avg(1:N_s,:) = 0; W_avg(1:N_s,1:N_s) = eye(N_s);
% 
%     % for randomized broadcast gossip
%     sample_instance = T0+randperm(s,no_gossip-(T0+1),no_samples);
%     x_op_sample = zeros(N+N_s,length(sample_instance));
%     cnt_sample = 1;
%     % now, utilize the randomized gossip exchange...
%     for gossip_round = 1 : no_gossip
%         src_node = randi(N+N_s,1,1); % choose the node to wake up
%         ek = zeros(N+N_s,1); ek(src_node) = 1;
%         W_cur = eye(N+N_s) - diag( C(:,src_node) ) + C(:,src_node)*ek';
%         x_op = W_cur*x_op;
% 
%         if ~isempty(find(sample_instance==gossip_round,1))
%             % take samples of x_op & Mul_accu
%             x_op_sample(:,cnt_sample) = x_op + 0.01*randn(N+N_s,1);
%             cnt_sample = cnt_sample + 1;
%         end
%     end
% 
%     fprintf('%i ',eee);
%     % sanity check
%     x_result = x_op_sample*ones(no_samples,1) / no_samples;
%     norm(W_bar^1e5 * init_op(:,eee) - x_result,'fro')
%     
% 
%     op_exp_result(:,eee) = x_result;
%     
% end
% fprintf('\n');
% 
% %%%%%%% part 2 of the simulation --> regression on the data %%%%%%
% % we want to infer B,D from the data
% 
% B_mask = vec( W_bar(N_s+1:end,1:N_s) > 0 );
% BC_mask = vec( W_bar(N_s+1:end,1:N_s) == 0 );
% 
% B_true = W_bar(N_s+1:end,1:N_s);
% D_true = W_bar(N_s+1:end,N_s+1:end);
% 
% % normalize D_true & B_true...
% D_normalize = diag(1 ./ (1 - diag(D_true))) * D_true; 
% D_normalize = D_normalize - diag(diag(D_normalize));
% B_normalize = diag(1 ./ (1 - diag(D_true))) * B_true;
% 
% % fprintf('CVX invoked \n');
% cvx_quiet(true)
% cvx_solver('sedumi')
% cvx_begin
%     variable D(N,N) 
%     variable B(N,N_s)
%     minimize( ...
%         1*( norm(D(:),1)) );
%     subject to
%         norm(op_exp_result(N_s+1:end,:) - D*op_exp_result(N_s+1:end,:) - B*op_exp_result(1:N_s,:),'fro')/ sqrt(total_exp) <= 1e-2;
%         D(:) >= 0; B(:) >= 0;
%         B(B_mask) >= 1e-8; B(BC_mask) == 0;
% %         diag(D) >= max(D')'; diag(D) >= max(B')';
%         diag(D) == 0;
%         ones(N,1) == [B D]*ones(N+N_s,1);
% cvx_end
% 
% % normalize D & B...
% D_hat_norm = diag(1 ./ (1 - diag(D))) * D; 
% D_hat_norm = D_hat_norm - diag(diag(D_hat_norm));
% B_hat_norm = diag(1 ./ (1 - diag(D))) * B;
% 
% threshold = 0.5*min( D_normalize ( D_normalize > 0 ) );
% 
% MSE_D(nnn,mc_sim) = sum( sum( (D_hat_norm - D_normalize).^2 ) ) / sum(D_normalize(:).^2);
% SUPPORT_D(nnn,mc_sim) = sum(sum(abs((D_hat_norm>=threshold) - (D_normalize>0))));
% 
% MSE_B(nnn,mc_sim) = sum( sum( (B_hat_norm - B_normalize).^2 ) ) / sum(B_normalize(:).^2);
% 
% D_true_size(nnn,mc_sim) = sum(G(:));
% 
% fprintf('MSE in D: %f, MSE in B: %f, SUPPORT in D: %i \n',MSE_D(nnn,mc_sim),...
%     MSE_B(nnn,mc_sim), SUPPORT_D(nnn,mc_sim) );
% 
% save sim_results_prelim
% 
% %% trash
% % % we use the randomized broadcast gossip model!
% % % generate the "C" matrix used --> it only needs to be row stochastic
% % C = ones(N+N_s,N+N_s); C = (G_com+eye(N+N_s)).*C;
% % C(1:N_s,:) = 0; C(1:N_s,1:N_s) = eye(N_s);
% % C = C ./ repmat( sum(C,2), 1, N+N_s);
% % 
% % % find the average W
% % W_bar = eye(N+N_s) - diag(C*ones(N+N_s,1)) / (N+N_s) + C / (N+N_s);
% % 
% % fprintf('Running the randomized experiments: \n');
% % % parameters for the random sampling
% % op_exp_result = zeros(N+N_s,total_exp);
% % T0 = 500; no_gossip = 5e4; no_samples = 5e2;
% % for eee = 1 : total_exp
% %     x_op = init_op(:,eee);
% %     sample_instance = T0+randperm(s,no_gossip-(T0+1),no_samples);
% %     x_op_sample = zeros(N+N_s,length(sample_instance));
% %     cnt_sample = 1;
% %     % now, utilize the randomized gossip exchange...
% %     for gossip_round = 1 : no_gossip
% %         src_node = randi(N+N_s,1,1); % choose the node to wake up
% %         ek = zeros(N+N_s,1); ek(src_node) = 1;
% %         W_cur = eye(N+N_s) - diag( C(:,src_node) ) + C(:,src_node)*ek';
% %         x_op = W_cur*x_op;
% % 
% %         if ~isempty(find(sample_instance==gossip_round,1))
% %             % take samples of x_op & Mul_accu
% %             x_op_sample(:,cnt_sample) = x_op;
% %             cnt_sample = cnt_sample + 1;
% %         end
% %     end
% %     
% %     fprintf('%i ',eee);
% %     % sanity check
% %     x_result = x_op_sample*ones(no_samples,1) / no_samples;
% %     
% %     op_exp_result(:,eee) = x_result;
% % end
% % fprintf('\n ');
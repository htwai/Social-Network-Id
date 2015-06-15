function [D,B] = solve_nsi_cvx(op_exp_result,BC_mask,N,N_s)

cvx_quiet(true)
cvx_solver('sedumi')
cvx_begin
    variable D(N,N) 
    variable B(N,N_s)
    minimize(  norm(D(:),1) );
    subject to
        % op_exp_result(N_s+1:end,:) corresponds to the final opinions of
        % Non-stubborn agents, i.e., the matrix $Y$
        % op_exp_result(1:N_s,:) corresponds to the final/initial opinions
        % of stubborn agents, i.e., the matrix $Z$
        op_exp_result(N_s+1:end,:) == D*op_exp_result(N_s+1:end,:) + B*op_exp_result(1:N_s,:);
        D(:) >= 0; B(:) >= 0;
        B(BC_mask) == 0;
        diag(D) == 0;
        ones(N,1) == [B D]*ones(N+N_s,1);
cvx_end
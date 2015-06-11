% program to evaluate the required beta based on the constants
% i.e., Theorem 1 in the journal
% Written on 05.18.2015

d = 5; % the constatnt degree
p = 8/100; % the probability

% these are some constants as defined in the theorem...
delta = 1 - 1 / (d-1);
rho = d / (d-1);
mu = (1 + rho*delta)*p;
beta = 0.001 : 0.001 : 0.99;

metric = mu*log2( beta / mu )*(d -1) - (b_ent(mu) + beta.*b_ent(mu./beta));

% print out the beta that we needed...
beta(min(find(metric > 0)))
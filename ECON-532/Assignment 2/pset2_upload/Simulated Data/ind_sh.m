function s = ind_sh(p,mean_val,alpha)
% This function computes the "individual" probabilities of choosing each brand

% Written by Aviv Nevo, May 1998.

% This function calculates a matrix of market shares given mean value and a
% matrix of individual values (for each draw).
global ns nbrn

u=exp(mean_val(:,ones(ns,1))'-alpha*p);
total=sum(u,2)+1;
s=u./total(:,ones(nbrn,1));
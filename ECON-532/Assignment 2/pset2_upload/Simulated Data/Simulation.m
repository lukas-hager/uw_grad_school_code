
clear;
clear global;
global beta alpha sigma_a xi ns nbrn
s = RandStream('mt19937ar','Seed',43211);
RandStream.setGlobalStream(s);


nmkt = 100;                              % number of markets
nbrn = 3;                                % number of products in each market
ns = 500;                                % number of consumers in each market

beta=[5,1,1]';
alpha_mean=1;
sigma_a=1;
gamma=[2,1,1];


                                        
x1=[ones(nmkt*nbrn,1),unifrnd(0,1,nmkt*nbrn,1),normrnd(0,1,nmkt*nbrn,1)];
xi_all = normrnd(0,1,nbrn*nmkt,1);        %Market characteristics

m_id = transpose(rude(nbrn.*ones(nmkt,1),transpose(1:nmkt)));
                                        %index of markets

                                        
alphas = alpha_mean+sigma_a*lognrnd(0,1, nmkt,ns);
                                        %consumer characteristics

                                        

                              
            %Cost simulation
w = repmat(normrnd(0,1,nbrn,1),[nmkt,1]);
eta = normrnd(0,1,nmkt*nbrn,1);
Z = normrnd(0,1,nmkt*nbrn,1);
mc = [ones(nbrn*nmkt,1),w,Z]*gamma'+eta;

P_opt = zeros(nmkt*nbrn,1); 
exitflag = zeros(nmkt,1);
shares=zeros(nmkt*nbrn,1);
elas=zeros(nmkt*nbrn,1);
for i=1:nmkt
    xi = xi_all(m_id==i,:);
    X = x1(m_id==i,1:end);
    mc_mkt = mc(m_id==i);
    alpha = alphas(i,:)';

    mean_val=X*beta+xi;


    ds = @(p) mean(alpha(:,ones(nbrn,1)).*ind_sh(p,mean_val,alpha).*(1-ind_sh(p,mean_val,alpha)));
    b = @(p) -mean(ind_sh(p,mean_val,alpha))./ds(p);
    f = @(p) (p - mc_mkt' + b(p));

    options=optimset('MaxFunEvals',1000000,'MaxIter',1000000,'Display','off');
    [P_opt(m_id==i),fval,exitflag(i)] = fsolve(f,mc_mkt',options);
    shares(m_id==i)=mean(ind_sh(P_opt(m_id==i)',mean_val,alpha))';
    es=[0,0,0]';
    for j=1:3
        z=[0,0,0];
        p=P_opt(m_id==i)';
        z(j)=p(j)*.00001;
        m=(mean(ind_sh(p+z,mean_val,alpha))-shares(m_id==i)')./z(j)*p(j)./shares(m_id==i)';
        es(j)=m(j);
    end
    elas(m_id==i)=es;
	%e1=reshape(P_opt./shares.*(-shares.^2-shares),[3,100])
	%e2=reshape(-1./((P_opt-reshape(mc,[3,100]))./P_opt),[3,100])

    %verify prices are optimal
%     p=P_opt(m_id==i);
%     profit=(p-mc_mkt).*mean(ind_sh(p',mean_val,alpha))';
%     for j = 1:nbrn
%         m=zeros(nbrn,1);
%         m(j)=0.001';
%         np=(p+m-mc_mkt).*mean(ind_sh((p+m)',mean_val,alpha))';
%         if np(j)>profit(j)
%             asdf
%         end
%         m(j)=-0.001';
%         np=(p+m-mc_mkt).*mean(ind_sh((p+m)',mean_val,alpha))';
%         if np(j)>profit(j)
%             asdf
%         end
%     end

end
%Average price
min(exitflag)
mean(P_opt)
sum(P_opt.*shares)/sum(shares)
sum((P_opt-mc).*shares)/sum(shares)

P_opt=reshape(P_opt, [nbrn,nmkt]);
shares=reshape(shares, [nbrn,nmkt]);


%save 10markets3products x1 xi_all alphas w eta Z P_opt shares
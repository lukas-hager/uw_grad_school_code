/*
 *  This produces the code to run the model in Problem 2.3c
 */


var pi x i ep ex;

varexo etap etax;

/*
 *  add Taylor rule parameters
 */

parameters beta gam lambda sigma rhop rhox sigmap sigmax phip phix;

beta = 0.96;
gam = .1;
lambda = 0.25;
sigma = 2;
rhop = 0.95;
rhox = 0.95;
sigmap = .01;
sigmax = .005;
phix = .5;
phip = 1.5;

/*
 *  No optimization; equilibrium conditions are NKPS, NIS, Taylor rule, and shocks
 */

model;

    i = phix * x + phip * pi;

    pi = beta * pi(+1) + lambda * x + ep;

    x = x(+1) - (1/sigma) * (i - pi(+1)) + ex;

    ep = rhop * ep(-1) + etap;

    ex = rhox * ex(-1) + etax;

end;

initval;
   ep = 0;
   ex = 0;
   i = .5;
   pi = .5;
   x = .5;
end;

shocks;
   var etap = sigmap^2;
   var etax = sigmax^2;
end;

steady;

stoch_simul(order=1);

/*
 *  This produces the code to run the model in Problem 2.1
 */

/*
 *  Define variables
 */
var pi x i ep ex;

/*
 *  Define shocks
 */
varexo etap etax;

/*
 *  Define parameters
 */
parameters beta gam lambda sigma rhop rhox sigmap sigmax;

beta = 0.96;
gam = .1;
lambda = 0.25;
sigma = 2;
rhop = 0.95;
rhox = 0.95;
sigmap = .01;
sigmax = .005;

/*
 *  Define model with equilibrium conditions
 */
model;

    lambda * (beta * pi(+1) + lambda * x + ep) + gam * x = 0;

    pi = beta * pi(+1) + lambda * x + ep;

    x = x(+1) - (1/sigma) * (i - pi(+1)) + ex;

    ep = rhop * ep(-1) + etap;

    ex = rhox * ex(-1) + etax;

end;

/*
 *  Define initial values
 */
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

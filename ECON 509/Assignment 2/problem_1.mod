/*
 *  This produces the code to run the model in Problem 1
 */

/*
 *  Define variables
 */
var c k l z w r;

/*
 *  Define shocks
 */
varexo epx;

/*
 *  Define parameters
 */
parameters beta gamma phi eta delta alpha rho sigma;

    beta = .98;
    gamma = 2.5;
    phi = .4;
    eta = 2;
    delta = .1;
    alpha = .35;
    rho = .95;
    sigma = .01;

/*
 *  Define model with equilibrium conditions
 */
model;

    (phi / w) * l^(eta) = c^(-gamma);

    c^(-gamma) = beta * (r(+1) + 1 - delta)*c(+1)^(-gamma);

    k(+1) = w * l + (r + 1 - delta) * k - c;

    r = alpha * z * k^(alpha - 1) * l^(1-alpha);

    w = (1-alpha) * z * k^(alpha) * l^(-alpha);

    log(z) = rho * log(z(-1)) + epx;

end;

/*
 *  Define initial values
 */
initval;
   l = 1;
   k = 1;
   c = 1;
   w = 1;
   r = 1;
   z = 1;
end;

shocks;
   var epx = sigma^2;
end;

steady;

stoch_simul(order=1);

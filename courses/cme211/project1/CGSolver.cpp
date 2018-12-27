#include <iostream>
#include <vector>

#include "matvecops.hpp"

int CGSolver(std::vector<double> &val,
             std::vector<int>    &row_idx,
             std::vector<int>    &col_idx,
             std::vector<double> &b,
             std::vector<double> &x,
             double              tol) {
    
    int nitermax = (int)x.size();
    if (!same_size(x,b)) {
        return 0;
    }
    else {
        std::vector<double> r = b;
        std::vector<double> Ax = matdotvec(val,row_idx,col_idx,x);
        sum(r,Ax,-1.);
        double norm0 = L2_norm(r);
        std::vector<double> p = r;
        int niter = 0;

        while (niter < nitermax) {
            niter += 1;
            std::vector<double> Ap = matdotvec(val,row_idx,col_idx,p);
            double rr = dot_prod(r,r);
            double alpha = rr/dot_prod(p,Ap);
            sum(x,p,alpha);
            sum(r,Ap,-alpha);
            double norm = L2_norm(r);
            if (norm/norm0 < tol) break;
            double beta = norm*norm/rr;
            scalar_prod(p,beta);
            sum(p,r,1.);
        }
        if (niter <=nitermax) return niter;
        else return -1;
    }
}
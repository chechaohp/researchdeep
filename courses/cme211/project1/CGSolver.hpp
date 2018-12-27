#ifndef CGSOLVER_HPP
#define CGSOLVER_HPP

#include <vector>

int CGSolver(std::vector<double> &val,
             std::vector<int>    &row_ptr,
             std::vector<int>    &col_idx,
             std::vector<double> &b,
             std::vector<double> &x,
             double              tol);

#endif

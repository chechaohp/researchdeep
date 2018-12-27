#ifndef MATVECOPS_HPP
#define MATVECOPS_HPP

#include <vector>

bool same_size(std::vector<double> &a, std::vector<double> &b);

void sum(std::vector<double> &a, std::vector<double> &b, double c);

void scalar_prod(std::vector<double> &a, double c);

double dot_prod(std::vector<double> &a, std::vector<double> &b);

double L2_norm(std::vector<double> &a);

std::vector<double> matdotvec(std::vector<double> &val,
                            std::vector<int> &row_idx,
                            std::vector<int> &col_idx,
                            std::vector<double> &x);

#endif
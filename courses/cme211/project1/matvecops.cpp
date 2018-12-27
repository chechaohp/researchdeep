#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

bool same_size(std::vector<double> &a, std::vector<double> &b) {
    if (a.size() == b.size()) {
        return true;
    }
    else {
        std::cout << "two vector diffirent sizes" << std::endl;
        return false;
    }
}

void sum(std::vector<double> &a, std::vector<double> &b, double c) {
    if (!same_size(a,b)) return;
    for (unsigned int i = 0; i < a.size(); i++) {
        a[i] += c*b[i];
    }
}

void scalar_prod(std::vector<double> &a, double c) {
    for (unsigned int i = 0; i< a.size(); i++) {
        a[i] = c*a[i];
    }
}

double dot_prod(std::vector<double> &a, std::vector<double> &b) {
    if (!same_size(a,b)) {
        return 0.;
    }
    else {
        double sum = 0.;
        for (unsigned int i = 0; i < a.size(); i++) {
            sum += a[i]*b[i];
        }
        return sum;
    }
}

double L2_norm(std::vector<double> &a) {
    return std::sqrt(dot_prod(a,a));
}

std::vector<double> matdotvec(std::vector<double> &val,
                            std::vector<int> &row_idx,
                            std::vector<int> &col_idx,
                            std::vector<double> &x) {
    std::vector<double> b;
    
    unsigned int max_col = *std::max_element(col_idx.begin(), col_idx.end());
    if (max_col+1 > x.size()) {
        std::cerr << "ERROR: Matrix size bigger than x" << std::endl;
        return b;
    }
    else {
        for (unsigned int i = 0; i< row_idx.size() - 1; i++) {
            double temp = 0.;
            for (int j = row_idx[i]; j < row_idx[i+1]; j++) {
                temp += val[j]*x[col_idx[j]];
            }
            b.push_back(temp);
        }
        return b;
    }
}
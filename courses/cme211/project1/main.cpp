// include built in library
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
//include custom library
#include "COO2CSR.hpp"
#include "CGSolver.hpp"
// Main program
int main(int argc, char *argv[]) {
	if (argc < 3) {
		std::cout << "Usage:" << std::endl;
		std::cout << "  " << argv[0] << " <input matrix file name> <output solution file name>" << std::endl;
		return 0;
	}
	// get input file and output file name
	std::string input_mat = argv[1];
	std::string output_sol = argv[2];
	// declare variable
	std::ifstream input_file;
	
	std::vector<int> ix_rows;
	std::vector<int> ix_cols;
	std::vector<double> vals;
	int cols;
	int rows;

	// read input file
	input_file.open(input_mat);
	if (input_file.is_open()) {
		// Do something
		input_file >> rows >> cols;
		int col;
		int row;
		double val;
		while (input_file >> row >> col >> val) {
			ix_rows.push_back(row);
			ix_cols.push_back(col);
			vals.push_back(val);
		};
		input_file.close();
		std::cout << rows << " " << cols << std::endl;
		for (auto v : vals) {
			std::cout << v << std::endl;
		};
		COO2CSR(vals,ix_rows,ix_cols);

		std::vector<double> b(cols,0.);
		std::vector<double> x(cols,1.);

		double tol = 1.e-5;
		int niter = CGSolver(vals, ix_rows,ix_cols,b,x,tol);

		if (niter == -1) {
			std::cout << "Solution does not converge" << std::endl;
			return 0;
		}
		else {
			std::ofstream output_file(output_sol);
			if (output_file.is_open()) {
				output_file.setf(std::ios::scientific, std::ios::floatfield);
				output_file.precision(4);
				for (auto sols : x) {
					output_file << sols << std::endl;
				};
				output_file.close();
			};
			std::cout << "FINISH in " << niter << "iters" << std::endl;
		};
	}
	else {
		std::cout << "Cannot open file " << input_mat << std::endl;
	}
	std::cin.ignore();
	return 0;
}
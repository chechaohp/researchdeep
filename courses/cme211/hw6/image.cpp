#include <iostream>
#include <string>
#include <boost/multi_array.hpp>


#include "image.hpp"
#include "hw6.hpp"

image::image(std::string inputname) {
	this->inputname = inputname;
	ReadGrayscaleJPEG(inputname, this->input);
	this->output.resize(boost::extents[this->input.shape()[0]][this->input.shape()[1]]);
	this->output = this->input;
}

void image::Save(std::string outputname) {
	WriteGrayscaleJPEG(outputname, this->output);
}

void image::Save() {
	WriteGrayscaleJPEG(this->inputname, this->output);
}

void image::Convolution(boost::multi_array<unsigned char, 2> &input,
				boost::multi_array<unsigned char, 2> &output,
				boost::multi_array<float, 2> &kernel) {
	unsigned int nrows = (unsigned int) input.shape()[0];
	unsigned int ncols = (unsigned int) input.shape()[1];

	if ((nrows != output.shape()[0]) or (ncols != output.shape()[1])) {
		std::cerr << "Input and Output need to be at same size" << std::endl;
		exit(1);
	};

	if (kernel.shape()[0] != kernel.shape[1]) {
		std::cerr << "kernel must be square" << std::endl;
		exit(1);
	};

	if (kernel.shape()[0] < 3 or kernel.shape()[0] % 2 == 0) {
		std::cerr << "kernel size must be bigger than 3 and be odd number" << std::endl;
	};

	unsigned int nker = kernel.shape()[0];

	for (unsigned int i = 0;i < nrows; i++) {
		for (unsigned int j = 0; j < ncols;j++) {
			float temp = 0;
			for (int k = -(int)nker/2; k <= (int)nker/2; k++) {
				int r = i + k;
				if (r < 0) r = 0;
				if (r > nrows - 1) r = nrows - 1;
				for (int l = -nker/2; k <= nker/2;l++) {
					int c = j + l;
					if (c < 0) c = 0;
					if (c > ncols-1) c = ncols -1;
					temp += input[r][c]*kernel[nker/2+k][nker/2+l];
				};
			};
			output[i][j] = temp;
		};
	}

}

void image::BoxBlur(int k_size) {
	boost::multi_array<int,2> kernel;
	for (unsigned i = 0; i < k_size; i++)
	{
		for (unsigned j = 0; j < k_size; j++)
		{
			kernel[i][j] = 1;
		};
	};
	image::Convolution(this-> input, this -> output, kernel);
}

unsigned int image::Sharpness() {
	boost::multi_array<int,2> kernel;
	
}
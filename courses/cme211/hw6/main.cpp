#include <iostream>
#include <string>
#include <image.hpp>
#include <hw6.hpp>

int main() {
	std::string filename = "stanford.jpg";

	image image_file =  image(filename);

	std::cout << "Orginal image: " << image_file.Sharpness() << std::setw(2) << std::endl;
	unsigned sharpness = image_file.Sharpness();
	for (int k_size = 3; k_size <= 27; k_size += 4)
	{
		image temp_image = image_file;
		temp_image.BoxBlur(k_size);
		std::cout << "Box Blur " << image_file.Sharpness() << std::setw(10) << std::endl;
	};
}

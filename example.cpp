#include "waterpixels.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char **argv) {
	if (argc != 3) {
		std::cout << "Usage: ./waterpixels_example <file name> <superpixel count>" << std::endl;
		return -1;
	}
	char* fileName = argv[1];
	Mat img = imread(fileName, IMREAD_COLOR);
	Mat result = imread(fileName, IMREAD_COLOR);
	// Mat img = imread(fileName, CV_8UC1);
	// Mat result = imread(fileName, CV_8UC1);

	imshow("Original", img);
    waitKey(0); // Wait for a keystroke in the window

	Waterpixels waterpixels = Waterpixels();
	int k = atoi(argv[2]);
	int numlabels;
	int width = img.cols;
	int height = img.rows;
	int size = width * height;
	int* labels = new int[size];

	std::cout << "height, width: " << height << " " << width << std::endl;

	waterpixels.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(img, width, height, labels,
															numlabels, k);
	waterpixels.DrawContoursAroundSegments(result, labels);

	std::cout << "label count: " << numlabels << std::endl;

	imshow("Contours around segments", result);
	waitKey(0); // Wait for a keystroke in the window
	// imwrite("output.png", img);
	return 0;
}

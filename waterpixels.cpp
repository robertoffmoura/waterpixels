#include <cfloat>
#include <cmath>
#include <iostream>
#include <fstream>
#include "waterpixels.h"

using namespace cv;

const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

Waterpixels::Waterpixels() {}
Waterpixels::~Waterpixels() {}

//=================================================================================
/// DrawContoursAroundSegments
///
/// Internal contour drawing option exists. One only needs to comment the if
/// statement inside the loop that looks at neighbourhood.
//=================================================================================
void Waterpixels::DrawContoursAroundSegments(Mat& 			img,
									  int*&					labels,
									  // const unsigned int&	color) {
									  Vec3b color) {
	std::vector<bool> istaken(size, false);
	std::vector<int> contourx(size);
	std::vector<int> contoury(size);
	int mainindex = 0;
	int cind = 0;
	for (int j = 0; j < height; j++) {
		for (int k = 0; k < width; k++) {
			int np = 0;
			for (int i = 0; i < 8; i++) {
				int x = k + dx8[i];
				int y = j + dy8[i];

				if ((x >= 0 && x < width) && (y >= 0 && y < height)) {
					int index = y*width + x;

					//if (istaken[index] == false) { //comment this to obtain internal contours
						if (labels[mainindex] != labels[index]) np++;
					// }
				}
			}
			if (np > 1) {
				contourx[cind] = k;
				contoury[cind] = j;
				istaken[mainindex] = true;
				//img[mainindex] = color;
				cind++;
			}
			mainindex++;
		}
	}

	int numboundpix = cind; //int(contourx.size());
	for (int j = 0; j < numboundpix; j++) {
		int ii = contoury[j]*width + contourx[j];
		// img.at<Vec3b>(contoury[j], contourx[j]) = Vec3b(0xff,0xff,0xff);
		img.at<Vec3b>(contoury[j], contourx[j]) = color;

		for (int n = 0; n < 8; n++) {
			int x = contourx[j] + dx8[n];
			int y = contoury[j] + dy8[n];
			if ((x >= 0 && x < width) && (y >= 0 && y < height) ) {
				int ind = y*width + x;
				if (!istaken[ind]) {
					img.at<Vec3b>(y, x) = Vec3b(0,0,0);
				}
			}
		}
	}
}

void Waterpixels::get4Neighbors(int s, int& numneighbors, int* neighbors) {
	int i = s / width;
	int j = s % width;
	if (i+1 < height) neighbors[numneighbors++] = s+width;
	if (i-1 >= 0)     neighbors[numneighbors++] = s-width;
	if (j+1 < width)  neighbors[numneighbors++] = s+1;
	if (j-1 >= 0)     neighbors[numneighbors++] = s-1;
}

void Waterpixels::get8Neighbors(int s, int& numneighbors, int* neighbors) {
	int i = s / width;
	int j = s % width;
	if (i+1 < height) {
		neighbors[numneighbors++] = s+width;
		if (j+1 < width)  neighbors[numneighbors++] = s+width+1;
		if (j-1 >= 0)     neighbors[numneighbors++] = s+width-1;
	}
	if (i-1 >= 0)     {
		neighbors[numneighbors++] = s-width;
		if (j+1 < width)  neighbors[numneighbors++] = s-width+1;
		if (j-1 >= 0)     neighbors[numneighbors++] = s-width-1;
	}
	if (j+1 < width)  neighbors[numneighbors++] = s+1;
	if (j-1 >= 0)     neighbors[numneighbors++] = s-1;
}

void Waterpixels::populateMarkersUsingHexagonalLattice(int STEP, int* markers, int& markerSize) {
	float h_i = 0;
	int i = 0;
	bool odd = false;
	float oddStep = STEP * sqrt(3)/2.0;
	float verticalStep = STEP * sqrt(3);
	float horizontalStep = STEP / 2.0;
	while (i < height) {
		float h_j = odd ? oddStep : 0;
		int j = (int) h_j;
		while (j < width) {
			markers[markerSize++] = i*width + j;
			h_j += verticalStep;
			j = (int) h_j;
		}
		h_i += horizontalStep;
		i = (int) h_i;
		odd = !odd;
	}
}

void Waterpixels::calculateDistance(int* distance, int* markers, int& markerSize) {
	int* queue = new int[size];
	for (int s = 0; s < size; s++) queue[s] = markers[s];
	int queueIndex = 0;
	int queueSize = markerSize;
	for (int s = 0; s < size; s++) distance[s] = -1;
	// calculate distance
	for (int i = 0; i < markerSize; i++) distance[markers[i]] = 0;

	while (queueSize > 0) {
		int s = queue[queueIndex];
		int currentDistance = distance[s];

		// add neighbors
		int numneighbors = 0;
		int* neighbors = new int[4];
		get4Neighbors(s, numneighbors, neighbors);
		for (int j=0; j<numneighbors; j++) {
			int neighbor = neighbors[j];
			if (distance[neighbor] != -1) continue;
			queue[queueIndex + queueSize++] = neighbor;
			distance[neighbor] = currentDistance + 1;
		}

		queueSize -= 1;
		queueIndex += 1;
	}
}

void printImageToConsole(int* a, int width, int height) {
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			std::cout << a[i*width + j] << " ";
		}
		std::cout << std::endl;
	}
}

void displayImage(int* r, int* g, int* b) {
	Mat img = imread("302003.jpg", IMREAD_COLOR);
	for (int i=0; i<img.rows; i++) {
		for (int j=0; j<img.cols; j++) {
			uint8_t * p = img.ptr(i, j);
			int index = i*img.cols + j;
			p[0] = r[index];
			p[1] = g[index];
			p[2] = b[index];
		}
	}
	imshow("Regularized gradient", img);
	waitKey(0); // Wait for a keystroke in the window
}

void displayGrayImage(int* a) {
	displayImage(a, a, a);
}

void Waterpixels::watershed(double* regularizedGradient, int* labels, int numlabels, int* markers, int markerSize) {
    std::priority_queue<pi, std::vector<pi>, std::greater<pi> > priorityQueue;
	for (int i=0; i<markerSize; i++) priorityQueue.push(std::make_pair(regularizedGradient[markers[i]], markers[i]));

	while (!priorityQueue.empty()) {
		pi top = priorityQueue.top();
		priorityQueue.pop();
		int s = top.second;

		// add neighbors
		int numneighbors = 0;
		int* neighbors = new int[4];
		get4Neighbors(s, numneighbors, neighbors);
		for (int j=0; j<numneighbors; j++) {
			int neighbor = neighbors[j];
			if (labels[neighbor] != -1) continue;
			priorityQueue.push(std::make_pair(regularizedGradient[neighbor], neighbor));
			labels[neighbor] = labels[s];
		}
	}
}

void Waterpixels::DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(
	Mat& 				img,
	const int			width,
	const int			height,
	int*&				labels,
	int&				numlabels,
	const int&			K) {      // required number of superpixels

	// const int STEP = sqrt(width*height*2/(K*sqrt(3)));
	// float a = K - 1/4.0;
	float a = K - 10.5;
	float b = height/2.0 + width/(2*sqrt(3));
	float c = -2*width*height/sqrt(3);
	const int STEP = (-b + sqrt(b*b - 4*a*c))/(2*a);

	//--------------------------------------------------
	this->width  = width;
	this->height = height;
	this->size = width*height;

	// Populate markers using hexagonal lattice
	int* markers = new int[size];
	int markerSize = 0;
	populateMarkersUsingHexagonalLattice(STEP, markers, markerSize);

	// Set labels on markers
	numlabels = markerSize;
	for (int s=0; s<size; s++) labels[s] = -1;
	for (int i=0; i<markerSize; i++) {
		labels[markers[i]] = i;
	}

	// Get distance to markers
	int* distance = new int[size];
	calculateDistance(distance, markers, markerSize);

	// Perform pre treatment of image (opening and closing)
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(STEP/4, STEP/4));
	morphologyEx(img, img, MORPH_OPEN, element);
	morphologyEx(img, img, MORPH_CLOSE, element);

	Mat gray = Mat(img);
	cvtColor(img, gray, COLOR_BGR2GRAY);
	Mat grad = Mat(img);

	Mat gradx = Mat();
	Mat grady = Mat();

	// Compute gradient
	Sobel(gray, gradx, CV_64F, 1, 0, 3);
	Sobel(gray, grady, CV_64F, 0, 1, 3);
	sqrt(gradx.mul(gradx) + grady.mul(grady), grad);

	// Threshold gradient
	double threshold = 150.0;
	for (int i=0; i<height; i++) {
		for (int j=0; j<width; j++) {
			grad.at<double>(i,j) = std::min(grad.at<double>(i,j), threshold);
		}
	}

	// Get regularized gradient as a combination of gradient and distance
	// int regularization = 800; // squared gradient
	// double regularization = 1;
	// double regularization = 5;
	double regularization = 1;
	double* regularizedGradient = new double[size];
	for (int s=0; s<size; s++) regularizedGradient[s] = grad.at<double>(s/width, s%width) + regularization*distance[s];

	// Print distance to console
	// printImageToConsole(distance, width, height);

	// Show markers distance
	// displayGrayImage(distance);

	// Show regularized gradient
	int* displayRegularizedGradient = new int[size];
	double max = 0;
	for (int s=0; s<size; s++) max = std::max(regularizedGradient[s], max);
	for (int s=0; s<size; s++) displayRegularizedGradient[s] = 255 * regularizedGradient[s] / max;
	// displayGrayImage(displayRegularizedGradient);

	// Apply the watershed algorithm
	watershed(regularizedGradient, labels, numlabels, markers, markerSize);
}

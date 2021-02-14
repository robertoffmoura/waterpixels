#include <cfloat>
#include <cmath>
#include <iostream>
#include <fstream>
#include "waterpixels.h"



// testing:
// #include <opencv2/opencv.hpp>
using namespace cv;


typedef pair<double, int> pdi;
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
	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

	vector<bool> istaken(size, false);
	vector<int> contourx(size);
	vector<int> contoury(size);
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

bool Waterpixels::valid(int i, int j) {
	return (0<=i && i<height) && (0<=j && j<width);
}

int Waterpixels::getDistanceToLabel(int p, vector<bool> isContour) {
	std::priority_queue<pi, vector<pi>, greater<pi> > priorityQueue;
	priorityQueue.push(std::make_pair(0.0, p));
	unordered_set<int> seen;
	seen.insert(p);

	while (!priorityQueue.empty()) {
		pi top = priorityQueue.top();
		priorityQueue.pop();
		double d = top.first;
		int s = top.second;
		if (isContour[s]) {
			return d;
		}

		int numneighbors = 0;
		int* neighbors = new int[4];
		get4Neighbors(s, numneighbors, neighbors);
		for (int j=0; j<numneighbors; j++) {
			int n = neighbors[j];
			if (seen.find(n) != seen.end()) continue;
			priorityQueue.push(std::make_pair(std::abs(n/width - p/width) + std::abs(n%width - p%width), n));
			// priorityQueue.push(std::make_pair(std::sqrt((pow(n/width - p/width,2) + pow(n%width - p%width,2))), n));
			seen.insert(n);
		}
	}
	return INT_MAX;
}

bool Waterpixels::belongsToBorder(int mainindex, int i, int j, int*& labels) {
	int np = 0;
	for (int k = 0; k < 8; k++) {
		int x = j + dx8[k];
		int y = i + dy8[k];
		if (!valid(y, x)) continue;

		int index = y*width + x;
		if (labels[mainindex] != labels[index]) np++;
	}
	return np > 1;
}

unordered_map<int, int> Waterpixels::getDistanceDistributionOfGroundTruthToSegmentation(
									  int*&	labels, int*& ground_truth_labels) {
	unordered_map<int, int> result;

	vector<bool> isContour(size, false);
	int mainindex = 0;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (belongsToBorder(mainindex, i, j, labels)) isContour[mainindex] = true;
			mainindex++;
		}
	}

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (belongsToBorder(mainindex, i, j, ground_truth_labels)) {
				int dist = getDistanceToLabel(mainindex, isContour);
				result[dist] += 1;
				// result[to_string(dist)] += 1;
			}
			mainindex++;
		}
	}

	return result;
}

double Waterpixels::GetContourDensity(int*& labels) {
	int count = 0;
	int mainindex = 0;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (belongsToBorder(mainindex, i, j, labels)) count++;
			mainindex++;
		}
	}
	return count / (double) size;
}

// Gets the center of mass of |superpixel|.
pi getCenter(unordered_set<pi, pair_hash> superpixel) {
	double i_sum, j_sum;
	for (auto p : superpixel) {
		i_sum += p.first;
		j_sum += p.second;
	}
	double total = superpixel.size();
	return std::make_pair(round(i_sum/total), round(j_sum/total));
}

// Centers all entries of |superpixel| on the pixel |c|.
void center(unordered_set<pi, pair_hash>& superpixel, pi c) {
	for (auto p : superpixel) {
		p.first -= c.first;
		p.second -= c.second;
	}
}

// int getAreaFromThreshold(unordered_map<pi, int, pair_hash>& sum_superpixel, int threshold) {
// 	int result = 0;
// 	for (auto p : sum_superpixel) {
// 		pi pixel = p.first;
// 		if (sum_superpixel[pixel] >= threshold) result++;
// 	}
// 	return result;
// }
//
// int getMaximalThresholdWithArea(unordered_map<pi, int, pair_hash>& sum_superpixel, int area) {
// 	// To do: implement binary search
// 	int t = 0;
// 	while (getAreaFromThreshold(sum_superpixel, t) >= area) t++;
// 	return t-1;
// }

// Returns maximum value of a hashmap.
int getMax(unordered_map<pi, int, pair_hash>& sum_superpixel) {
	int result = 0;
	for (auto p : sum_superpixel) {
		result = std::max(result, p.second);
	}
	return result;
}

// Returns a sum_superpixel whose components are greater than the threshold.
unordered_map<pi, int, pair_hash> getSuperpixelFromThreshold(unordered_map<pi, int, pair_hash>& sum_superpixel, int threshold) {
	unordered_map<pi, int, pair_hash> result;
	for (auto p : sum_superpixel) {
		pi pixel = p.first;
		if (sum_superpixel[pixel] >= threshold) result[pixel] = p.second;
	}
	return result;
}

// Performs a binary search for the highest threshold whose corresponding superpixel area
// is greater than the target.
int binarySearch(unordered_map<pi, int, pair_hash>& sum_superpixel, int start, int end, int target) {
	if (end <= start) {
		return start -1;
	}
	int mid = (start + end)/2;
	unordered_map<pi, int, pair_hash> mid_sum_superpixel = getSuperpixelFromThreshold(sum_superpixel, mid);
	if (mid_sum_superpixel.size() < target) {
		return binarySearch(sum_superpixel, start, mid-1, target);
	} else { // mid_sum_superpixel.size() >= target
		return binarySearch(mid_sum_superpixel, mid+1, end, target);
	}
}

// Returns the maximum threshold which gives a superpixel whose area is larger than |area|.
int getMaximalThresholdWithArea(unordered_map<pi, int, pair_hash>& sum_superpixel, int area) {
	int end = getMax(sum_superpixel);
	return binarySearch(sum_superpixel, 0, end, area);
}

// Returns the centered average shape of a superpixel, as a set of coordinates.
unordered_set<pi, pair_hash> Waterpixels::averageSuperpixel(int*& labels, int numlabels) {
	unordered_map<pi, int, pair_hash> sum_superpixel;
	int sum_area = 0;
	for (int l=0; l<numlabels; l++) {
		// Create superpixel
		unordered_set<pi, pair_hash> superpixel;
		int currentIndex = 0;
		for (int i=0; i<height; i++) {
			for (int j=0; j<width; j++) {
				if (labels[currentIndex] != l) continue;
				superpixel.insert(std::make_pair(i,j));
			}
		}
		// Center superpixel and add it to sum
		pi c = getCenter(superpixel);
		center(superpixel, c);
		for (auto p : superpixel) {
			sum_superpixel[p] += 1;
		}
		sum_area += superpixel.size();
	}
	// Create average superpixel and center it
	double average_area = sum_area / numlabels;
	int threshold = getMaximalThresholdWithArea(sum_superpixel, average_area);
	unordered_set<pi, pair_hash> avg_superpixel;
	for (auto p : sum_superpixel) {
		pi pixel = p.first;
		if (sum_superpixel[pixel] >= threshold) avg_superpixel.insert(pixel);
	}
	pi c = getCenter(avg_superpixel);
	center(avg_superpixel, c);
	return avg_superpixel;
}

// Returns the mismatch factor between two superpixels.
double mismatchFactor(unordered_set<pi, pair_hash> superpixelA, unordered_set<pi, pair_hash> superpixelB) {
	int intersectionAB = 0;
	for (auto p : superpixelA) {
		if (superpixelB.find(p) != superpixelB.end()) intersectionAB += 1;
	}
	int unionAB = superpixelA.size() + superpixelB.size() - intersectionAB;
	return 1 - (intersectionAB / (double) unionAB);
}

double Waterpixels::GetAverageMismatchFactor(int*& labels, int numlabels) {
	unordered_set<pi, pair_hash> avg_superpixel = averageSuperpixel(labels, numlabels);
	double factor_sum = 0;
	for (int l=0; l<numlabels; l++) {
		// Create superpixel
		unordered_set<pi, pair_hash> superpixel;
		int currentIndex = 0;
		for (int i=0; i<height; i++) {
			for (int j=0; j<width; j++) {
				if (labels[currentIndex] != l) continue;
				superpixel.insert(std::make_pair(i,j));
			}
		}
		// Center superpixel
		pi c = getCenter(superpixel);
		for (auto p : superpixel) {
			p.first -= c.first;
			p.second -= c.second;
		}
		// Compute mismatch factor and add it to sum
		factor_sum += mismatchFactor(superpixel, avg_superpixel);
	}
	return factor_sum / numlabels;
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

// typedef tuple<int, int, int> ti;
void Waterpixels::watershed(double* regularizedGradient, int* labels, int numlabels, int* markers, int markerSize) {
    std::priority_queue<pi, vector<pi>, greater<pi> > priorityQueue;
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

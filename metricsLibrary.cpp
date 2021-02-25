#include "metricsLibrary.h"
#include <queue>
#include <climits>

const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

// Gets the center of mass of |superpixel|.
pi getCenter(std::unordered_set<pi, pair_hash> superpixel) {
	double i_sum, j_sum;
	for (auto p : superpixel) {
		i_sum += p.first;
		j_sum += p.second;
	}
	double total = superpixel.size();
	return std::make_pair(std::round(i_sum/total), std::round(j_sum/total));
}

// Centers all entries of |superpixel| on the pixel |c|.
void center(std::unordered_set<pi, pair_hash>& superpixel, pi c) {
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
int getMax(std::unordered_map<pi, int, pair_hash>& sum_superpixel) {
	int result = 0;
	for (auto p : sum_superpixel) {
		result = std::max(result, p.second);
	}
	return result;
}

// Returns a sum_superpixel whose components are greater than the threshold.
std::unordered_map<pi, int, pair_hash> getSuperpixelFromThreshold(std::unordered_map<pi, int, pair_hash>& sum_superpixel, int threshold) {
	std::unordered_map<pi, int, pair_hash> result;
	for (auto p : sum_superpixel) {
		pi pixel = p.first;
		if (sum_superpixel[pixel] >= threshold) result[pixel] = p.second;
	}
	return result;
}

// Performs a binary search for the highest threshold whose corresponding superpixel area
// is greater than the target.
int binarySearch(std::unordered_map<pi, int, pair_hash>& sum_superpixel, int start, int end, int target) {
	if (end <= start) {
		return start -1;
	}
	int mid = (start + end)/2;
	std::unordered_map<pi, int, pair_hash> mid_sum_superpixel = getSuperpixelFromThreshold(sum_superpixel, mid);
	if (mid_sum_superpixel.size() < target) {
		return binarySearch(sum_superpixel, start, mid-1, target);
	} else { // mid_sum_superpixel.size() >= target
		return binarySearch(mid_sum_superpixel, mid+1, end, target);
	}
}

// Returns the maximum threshold which gives a superpixel whose area is larger than |area|.
int getMaximalThresholdWithArea(std::unordered_map<pi, int, pair_hash>& sum_superpixel, int area) {
	int end = getMax(sum_superpixel);
	return binarySearch(sum_superpixel, 0, end, area);
}

// Returns the mismatch factor between two superpixels.
double mismatchFactor(std::unordered_set<pi, pair_hash> superpixelA, std::unordered_set<pi, pair_hash> superpixelB) {
	int intersectionAB = 0;
	for (auto p : superpixelA) {
		if (superpixelB.find(p) != superpixelB.end()) intersectionAB += 1;
	}
	int unionAB = superpixelA.size() + superpixelB.size() - intersectionAB;
	return 1 - (intersectionAB / (double) unionAB);
}


MetricsLibrary::MetricsLibrary(int height, int width) {
	this->width  = width;
	this->height = height;
	this->size = width*height;
}

std::unordered_map<int, int> MetricsLibrary::getDistanceDistributionOfGroundTruthToSegmentation(
									  int*&	labels, int*& ground_truth_labels) {
	std::unordered_map<int, int> result;

	std::vector<bool> isContour(size, false);
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

int MetricsLibrary::getDistanceToLabel(int p, std::vector<bool> isContour) {
	std::priority_queue<pi, std::vector<pi>, std::greater<pi> > priorityQueue;
	priorityQueue.push(std::make_pair(0.0, p));
	std::unordered_set<int> seen;
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

bool MetricsLibrary::belongsToBorder(int mainindex, int i, int j, int*& labels) {
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

double MetricsLibrary::GetContourDensity(int*& labels) {
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

std::unordered_set<pi, pair_hash> MetricsLibrary::averageSuperpixel(int*& labels, int numlabels) {
	std::unordered_map<pi, int, pair_hash> sum_superpixel;
	int sum_area = 0;
	for (int l=0; l<numlabels; l++) {
		// Create superpixel
		std::unordered_set<pi, pair_hash> superpixel;
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
	std::unordered_set<pi, pair_hash> avg_superpixel;
	for (auto p : sum_superpixel) {
		pi pixel = p.first;
		if (sum_superpixel[pixel] >= threshold) avg_superpixel.insert(pixel);
	}
	pi c = getCenter(avg_superpixel);
	center(avg_superpixel, c);
	return avg_superpixel;
}

double MetricsLibrary::GetAverageMismatchFactor(int*& labels, int numlabels) {
	std::unordered_set<pi, pair_hash> avg_superpixel = averageSuperpixel(labels, numlabels);
	double factor_sum = 0;
	for (int l=0; l<numlabels; l++) {
		// Create superpixel
		std::unordered_set<pi, pair_hash> superpixel;
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

bool MetricsLibrary::valid(int i, int j) {
	return (0<=i && i<height) && (0<=j && j<width);
}

void MetricsLibrary::get4Neighbors(int s, int& numneighbors, int* neighbors) {
	int i = s / width;
	int j = s % width;
	if (i+1 < height) neighbors[numneighbors++] = s+width;
	if (i-1 >= 0)     neighbors[numneighbors++] = s-width;
	if (j+1 < width)  neighbors[numneighbors++] = s+1;
	if (j-1 >= 0)     neighbors[numneighbors++] = s-1;
}

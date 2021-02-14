#if !defined(_Waterpixels_H_INCLUDED_)
#define _Waterpixels_H_INCLUDED_


#include <vector>
#include <string>
#include <algorithm>
using namespace std;

#include <opencv2/opencv.hpp>

#include <unordered_set>
typedef pair<int, int> pi;
struct pair_hash {
    inline std::size_t operator()(const std::pair<int,int> & v) const {
        return v.first*31+v.second;
    }
};

class Waterpixels {
public:
	Waterpixels();
	virtual ~Waterpixels();
	//============================================================================
	// Superpixel segmentation for a given number of superpixels
	//============================================================================
    void DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(
		cv::Mat& 					img,
		const int					width,
		const int					height,
		int*&						klabels,
		int&						numlabels,
		const int&					K); // required number of superpixel

	//============================================================================
	// Function to draw boundaries around superpixels of a given 'color'.
	// Can also be used to draw boundaries around supervoxels, i.e layer by layer.
	//============================================================================
	void DrawContoursAroundSegments(
		cv::Mat& 					result,
		int*&						labels,
		// const unsigned int&			color );
		cv::Vec3b color = cv::Vec3b(0xff,0xff,0xff));

	//============================================================================
	// Returns the distribution of distances from every ground truth border pixel
	// to the closes superpixel segmentation border pixel.
	//============================================================================
	unordered_map<int, int> getDistanceDistributionOfGroundTruthToSegmentation(int*& labels, int*& ground_truth_labels);
	//============================================================================
	// Returns the count of pixels which are in a border between two labels divided
	// by the total pixel count.
	//============================================================================
	double GetContourDensity(int*& labels);
	// Returns the average mismatch factor (its computation is specified in the
	// waterpixel article).
	double GetAverageMismatchFactor(int*& labels, int numlabels);

private:
	// Populate the |markers| array with the indices of pixels which are centered
	// in a hexagonal grid. update |markerSize| to the count of markers. One pixel
	// per hexagon.
	void populateMarkersUsingHexagonalLattice(int STEP, int* markers, int& markerSize);
	// Computes the |distance| array which contains, for each pixel, the distance
	// to the closest marker. Computed using a BFS starting on the markers.
	void calculateDistance(int* distance, int* markers, int& markerSize);
	// Perform the watershed algorithm. With a priority queue, starts from
	// the markers (bottom-most level). After each element is popped from the
	// queue, adds its neighbors sorted by regularizedGradient.
	void watershed(double* regularizedGradient, int* labels, int numlabels, int* markers, int markerSize);
	// Gets the 4 connected pixel neighbors to |s|. There may be less than 4 if
	// |s| is on a border. |numneighbors| reflects how many neighbors there are
	// in fact.
	void get4Neighbors(int s, int& numneighbors, int* neighbors);
	// Gets the 8 connected pixel neighbors to |s|. There may be less than 8 if
	// |s| is on a border. |numneighbors| reflects how many neighbors there are
	// in fact.
	void get8Neighbors(int s, int& numneighbors, int* neighbors);
	// Whether the pixel coordinates fall inside the boundaries of the image's
	// width and height.
	bool valid(int i, int j);
	// Returns the distance from the pixel index |p| to the closest superpixel
	// segmentation border pixel. |isContour| is true for those pixels.
	int getDistanceToLabel(int p, vector<bool> isContour);
	// Returns whether the pixel at |mainindex|, whose coordinates are (i,j),
	// belongs to a border of a label in |labels|.
	bool belongsToBorder(int mainindex, int i, int j, int*& labels);
	// Returns the average superpixel.
	unordered_set<pi, pair_hash> averageSuperpixel(int*& labels, int numlabels);

	int		width;
	int		height;
	int		size;
	int		m_depth;
};

#endif // !defined(_Waterpixels_H_INCLUDED_)

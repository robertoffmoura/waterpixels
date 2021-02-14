#if !defined(_Waterpixels_H_INCLUDED_)
#define _Waterpixels_H_INCLUDED_


#include <vector>
#include <string>
#include <algorithm>

typedef std::pair<int, int> pi;
#include <opencv2/opencv.hpp>

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

	int	width;
	int	height;
	int	size;
	int	m_depth;
};

#endif // !defined(_Waterpixels_H_INCLUDED_)

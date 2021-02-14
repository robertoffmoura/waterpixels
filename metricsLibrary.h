#if !defined(_MetricsLibrary_H_INCLUDED_)
#define _MetricsLibrary_H_INCLUDED_

#include <vector>
#include <unordered_set>
#include <unordered_map>

typedef std::pair<int, int> pi;
struct pair_hash {
    inline std::size_t operator()(const std::pair<int,int> & v) const {
        return v.first*31+v.second;
    }
};

class MetricsLibrary {
public:
	MetricsLibrary(int height, int width);
	//============================================================================
	// Returns the distribution of distances from every ground truth border pixel
	// to the closes superpixel segmentation border pixel.
	//============================================================================
	std::unordered_map<int, int> getDistanceDistributionOfGroundTruthToSegmentation(int*& labels, int*& ground_truth_labels);
	//============================================================================
	// Returns the count of pixels which are in a border between two labels divided
	// by the total pixel count.
	//============================================================================
	double GetContourDensity(int*& labels);
	//============================================================================
	// Returns the average mismatch factor (its computation is specified in the
	// waterpixel article).
	//============================================================================
	double GetAverageMismatchFactor(int*& labels, int numlabels);
private:
	// Gets the 4 connected pixel neighbors to |s|. There may be less than 4 if
	// |s| is on a border. |numneighbors| reflects how many neighbors there are
	// in fact.
	void get4Neighbors(int s, int& numneighbors, int* neighbors);
	// Returns the distance from the pixel index |p| to the closest superpixel
	// segmentation border pixel. |isContour| is true for those pixels.
	int getDistanceToLabel(int p, std::vector<bool> isContour);
	// Returns whether the pixel at |mainindex|, whose coordinates are (i,j),
	// belongs to a border of a label in |labels|.
	bool belongsToBorder(int mainindex, int i, int j, int*& labels);
	// Returns the centered average shape of a superpixel, as a set of coordinates.
	std::unordered_set<pi, pair_hash> averageSuperpixel(int*& labels, int numlabels);
	// Whether the pixel coordinates fall inside the boundaries of the image's
	// width and height.
	bool valid(int i, int j);

	int	width;
	int	height;
	int	size;
};

#endif // !defined(_MetricsLibrary_H_INCLUDED_)

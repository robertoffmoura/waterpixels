#include "waterpixels.h"
#include <iostream>
#include <opencv2/opencv.hpp>

#include <fstream>
#include <filesystem>
namespace fs = std::filesystem;
#include <dirent.h>

using namespace cv;

vector<string> getFileNames(const char* path) {
	vector<string> segmentation_file_names;
	DIR *dir; struct dirent *diread;
	if ((dir = opendir(path)) != nullptr) {
		while ((diread = readdir(dir)) != nullptr) {
			string file_name = string(diread->d_name);
			if (file_name.compare(".") == 0 || file_name.compare("..") == 0 || file_name.compare(".DS_Store") == 0) {
				continue;
			}
			segmentation_file_names.push_back(file_name);
		}
		closedir(dir);
	}
	return segmentation_file_names;
}

int* getGroundTruthLabels(int size, int width, string file_path) {
	int* ground_truth_labels = new int[size];
	ifstream image_segmentation_file;
	image_segmentation_file.open(file_path);
	char data[200];
	while (image_segmentation_file >> data) {
		if (strcmp(data, "data") == 0) break;
	}
	int label, i, j1, j2;
	while (image_segmentation_file >> label >> i >> j1 >> j2) {
		for (int j=j1; j<=j2; j++) {
			ground_truth_labels[i*width+j] = label;
		}
	}
	image_segmentation_file.close();
	return ground_truth_labels;
}

void writeUnorderedMapToFile(unordered_map<int, int> mapToSave, string fileNameToSaveMap) {
	ofstream file;
	file.open(fileNameToSaveMap);
	for (const auto& d: mapToSave) {
		file << d.first << " " << d.second << std::endl;
	}
	file.close();
}

void writeStringToFile(string str, string fileName) {
	ofstream file;
	file.open(fileName);
	file << str << std::endl;
	file.close();
}

void appendStringToFile(string stringToAppend, string fileName) {
	ofstream file;
	file.open(fileName, fstream::app);
	file << stringToAppend << std::endl;
	file.close();
}

void computeAndWriteRegularityMeasures(int* ks, int kCount) {
	string path_string = "images/";
	const char* path = path_string.c_str();
	vector<string> image_file_names = getFileNames(path);
	std::cout << "Computing regularity measures (contour density and average mismatch factor) and writing the results to results/measures/" << std::endl;
	int file_count = image_file_names.size();
	int file_index = 1;
	for (auto file_name : image_file_names) {
		// Write the header of measures.csv
		string measuresFileName = "results/measures/" + file_name.substr(0, file_name.size() - 4) + ".csv";
		string header = "measure,k,value";
		writeStringToFile(header, measuresFileName);

		for (int i_k=0; i_k<kCount; i_k++) {
			int k = ks[i_k];
			Mat img = imread("images/" + file_name, IMREAD_COLOR);

			Waterpixels waterpixels = Waterpixels();
			int numlabels;
			int width = img.cols;
			int height = img.rows;
			int size = width * height;
			int* labels = new int[size];

			waterpixels.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(img, width, height, labels, numlabels, k);

			// Save contour density
			double contourDensity = waterpixels.GetContourDensity(labels);
			// string measuresFileName = "results/" + to_string(user) + "/" + file_name.substr(0, file_name.size() - 4) + "_measures.csv";
			string contourDensityLine = "contourDensity, " + to_string(k) + ", " + to_string(contourDensity);
			appendStringToFile(contourDensityLine, measuresFileName);

			// Save average mismatch factor
			double avgMismatchFactor = waterpixels.GetAverageMismatchFactor(labels, numlabels);
			// string measuresFileName = "results/" + to_string(user) + "/" + file_name.substr(0, file_name.size() - 4) + "_measures.csv";
			string avgMismatchFactorLine = "avgMismatchFactor, " + to_string(k) + ", " + to_string(avgMismatchFactor);
			appendStringToFile(avgMismatchFactorLine, measuresFileName);
		}
		std::cout << "(" << file_index++ << "/" << file_count << ") Done. " << measuresFileName << std::endl;
	}
}

void computeAndWriteSegmentationAndDistanceDistribution(int* ks, int kCount) {
	int users[] = {1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111,
					1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1121, 1122,
					1123, 1124, 1126, 1127, 1128, 1129, 1130, 1132};
	int userCount = 28;
	std::cout << "Computing superpixel segmentation and distance distribution and writing the results to results/segmentation/ and results/" << std::endl;

	// for (int i_user=1; i_user<2; i_user++) {
	for (int i_user=0; i_user<userCount; i_user++) {
		int user = users[i_user];

		string path_string = "human-labels/human/color/" + to_string(user) + "/";
		const char* path = path_string.c_str();
	    vector<string> segmentation_file_names = getFileNames(path);

		int segmentation_file_count = segmentation_file_names.size();
		int segmentation_file_index = 1;
	    for (auto file_name : segmentation_file_names) {
			for (int i_k=0; i_k<3; i_k++) {
				int k = ks[i_k];
				Mat img = imread("images/" + file_name.substr(0, file_name.size() - 4) + ".jpg", IMREAD_COLOR);
				Mat result = imread("images/" + file_name.substr(0, file_name.size() - 4) + ".jpg", IMREAD_COLOR);

				Waterpixels waterpixels = Waterpixels();
				int numlabels;
				int width = img.cols;
				int height = img.rows;
				int size = width * height;
				int* labels = new int[size];

				waterpixels.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(img, width, height, labels, numlabels, k);

				// Save segmentation
				waterpixels.DrawContoursAroundSegments(result, labels);
				imwrite("results/segmentation/" + file_name.substr(0, file_name.size() - 4) + "_seg" + to_string(k) + ".png", result);

				// Save ground truth
				Mat ground_truth = imread("images/" + file_name.substr(0, file_name.size() - 4) + ".jpg", IMREAD_COLOR);
				string path_and_name = path + file_name;
				int* ground_truth_labels = getGroundTruthLabels(size, width, path_and_name);
				waterpixels.DrawContoursAroundSegments(ground_truth, ground_truth_labels);
				imwrite("results/" + to_string(user) + "/" + file_name.substr(0, file_name.size() - 4) + "_gt.png", ground_truth);

				// Mat groundTruthAndSegmentation = Mat(result);
				// waterpixels.DrawContoursAroundSegments(groundTruthAndSegmentation, ground_truth_labels, Vec3b(0xff,0x00,0x00));
				// imshow("Ground truth + segmentation", groundTruthAndSegmentation);
				// waitKey(0);
				//
				// fs::create_directories("results/" + to_string(user) + "/");

				// Save distance distribution
				unordered_map<int, int> distanceDistribution = waterpixels.getDistanceDistributionOfGroundTruthToSegmentation(labels, ground_truth_labels);
				string fileNameToSaveMap = "results/" + to_string(user) + "/" + file_name.substr(0, file_name.size() - 4) + "_dist" + to_string(k) + ".txt";
				writeUnorderedMapToFile(distanceDistribution, fileNameToSaveMap);
			}
			std::cout << "(" << (i_user + 1) << "/" << userCount << ") User. " << "(" << segmentation_file_index++ << "/" << segmentation_file_count << ") Segmentation files done. " << "results/" + to_string(user) + "/" + file_name.substr(0, file_name.size() - 4) << std::endl;
		}
	}
}

int main(int argc, char **argv) {
	if (argc != 2 || (strcmp(argv[1], "adherence") != 0 && strcmp(argv[1], "regularity") != 0)) {
		std::cout << "Usage: ./waterpixels_metrics <metrics>" << std::endl;
		std::cout << "<metrics> can be either \"adherence\" or \"regularity\"." << std::endl;
		return -1;
	}
	int ks[] = {50, 100, 200};
	int kCount = 3;

	if (strcmp(argv[1], "adherence") == 0) {
		computeAndWriteSegmentationAndDistanceDistribution(ks, kCount);
	} else if (strcmp(argv[1], "regularity") == 0) {
		computeAndWriteRegularityMeasures(ks, kCount);
	}
}

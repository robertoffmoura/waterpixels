# Waterpixels
Implementation of the Waterpixels algorithm from
```
Vaïa Machairas, Matthieu Faessel, David Cárdenas-Peña, Théodore Chabardes, Thomas Walter, et al.. Waterpixels. IEEE Transactions on Image Processing, Institute of Electrical and Electronics Engineers, 2015, 24 (11), pp.3707 - 3716. 10.1109/TIP.2015.2451011 . hal-01212760
```

## Setup
To compile, run the following lines on a terminal window, from the root folder:
```
mkdir build
cd build
cmake ..
make
cd ..
```
If you want to get metrics for the resulting segmentations of a dataset, create a directory named `images` and one named `human-labels`, both children of the root folder. Place the images straight inside the `images` folder. Then, place the `.seg` segmentation files inside the `human-labels` folder, but separate them into folders for each user. For example:
```
human-labels/1102/33039.seg
```
That's the segmentation file for image `33039.jpg` made by user `1102`.

## Running
To display the superpixel segmentation result from a single image, run the following line:
```
./build/waterpixels_example <file name> <superpixel count>
```

For example:
```
./build/waterpixels_example 302003.jpg 100
```
### Boundary Adherence Metrics
To get all boundary adherence metrics from a dataset, run
```
./build/waterpixels_metrics adherence
```
This will save the ground-truth segmented image under the folder `results`.
For example:
```
results/1102/33039_gt.png
```
This will also save the distribution of Manhattan Distances from each ground truth border pixel to the closest superpixel segmentation border pixel. For example, for a superpixel count of 50:
```
results/1102/33039_dist50.txt
```
And finally, this will also save superpixel segmentation under `results/segmentation`.

### Lattice Regularity Metrics

To get the contour density and average mismatch factor of the resulting segmentations, simply run:
```
./build/waterpixels_metrics regularity
```
This will save these metrics under `results/measures`, in a `.csv` format.

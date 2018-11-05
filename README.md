Written by Steve S. Lee
stevesl@umich.edu
		
# Images2Points

Python script that can:
1) Detect feature points in two images.
2) Export detected feature points in csv format.
3) Find matching feature points between two images.
4) Export matching feature points in csv format.
5) Export an image that graphically shows detected feature points and matched feature points.

## Getting Started

This module uses SIFT, SURF, and ORB, some of which are patented and cannot be used for commmerical purposes without license. Academic use only.

### Prerequisites

This module requires cv2, numpy, csv, skimage, and matplotlib. Install these before using the module.


## Included Functions

### getPointsFromImages(...)

Required Parameters:
1) firstImage(BGR.)
2) secondImage(BGR.)

Optional Parameters:
1) outputcsvFileName(String. No file exported by default.)
2) detectorType(String. Uses "ORB" by default.)
* Options:
1) "ORB", "SIFT", and "SURF".
3) crossCheck(Bool. Set "True" by default".)
4) normType(String. Uses "cv2.NORM_HAMMING" by default.)
5) createImageWithPtsAndLines(Bool. No image exported by default.)

```
Give an example
```

Recommendations:
1) If ORB detector is used, use "cv2.NORM_HAMMING".
2) If SIFT or SURF detector is used, use either "cv2.NORM_L1" or "cv2.NORM_L2".

## Acknowledgments

* Sponsored by Raj Rao Nadakuditi.

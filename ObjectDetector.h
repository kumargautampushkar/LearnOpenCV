#ifndef OBJECT_DETECTOR_H
#define OBJECT_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

// ==========================================
// Abstract Base Class
// ==========================================
class ObjectDetectorBase {
protected:
    // Shared Data
    Mat query_image;
    Mat input_image;
    
    // Feature Data
    vector<KeyPoint> kp_query, kp_input;
    vector<DMatch> good_matches;

    // Results
    Mat homography;
    vector<Point2f> scene_corners;
    
    // Config
    const int MIN_MATCH_COUNT = 10;

public:
    ObjectDetectorBase(Mat query, Mat input);
    virtual ~ObjectDetectorBase();

    // Pure Virtual Function
    virtual void extract_and_match() = 0;

    // Shared Logic
    Mat calc_homography();
    vector<Point2f> calc_bbx_coordinate();
    void draw_results();

    // Getters
    vector<DMatch> get_good_matches() const;
    vector<KeyPoint> get_kp_query() const;
    vector<KeyPoint> get_kp_input() const;
    Mat getHomography() const;
};

// ==========================================
// SIFT Derived Class
// ==========================================
class SiftDetector : public ObjectDetectorBase {
public:
    SiftDetector(Mat query, Mat input);
    void extract_and_match() override;
};

// ==========================================
// ORB Derived Class
// ==========================================
class OrbDetector : public ObjectDetectorBase {
public:
    OrbDetector(Mat query, Mat input);
    void extract_and_match() override;
};

#endif // OBJECT_DETECTOR_H
#include "ObjectDetector.h"

// ==========================================
// Base Class Implementations
// ==========================================
ObjectDetectorBase::ObjectDetectorBase(Mat query, Mat input) 
    : query_image(query), input_image(input) {}

ObjectDetectorBase::~ObjectDetectorBase() {}

Mat ObjectDetectorBase::calc_homography() {
    if (good_matches.size() < MIN_MATCH_COUNT) {
        cout << "Not enough matches found: " << good_matches.size() << "/" << MIN_MATCH_COUNT << endl;
        return Mat();
    }

    vector<Point2f> obj;
    vector<Point2f> scene;

    for (size_t i = 0; i < good_matches.size(); i++) {
        obj.push_back(kp_query[good_matches[i].queryIdx].pt);
        scene.push_back(kp_input[good_matches[i].trainIdx].pt);
    }

    homography = findHomography(obj, scene, RANSAC);
    
    if (homography.empty()) {
        cout << "Homography calculation failed." << endl;
    } else {
        cout << "Homography calculated successfully." << endl;
    }
    return homography;
}

vector<Point2f> ObjectDetectorBase::calc_bbx_coordinate() {
    if (homography.empty()) return vector<Point2f>();

    vector<Point2f> obj_corners(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f((float)query_image.cols, 0);
    obj_corners[2] = Point2f((float)query_image.cols, (float)query_image.rows);
    obj_corners[3] = Point2f(0, (float)query_image.rows);

    perspectiveTransform(obj_corners, scene_corners, homography);
    cout << "Bounding box coordinates calculated." << endl;
    return scene_corners;
}

void ObjectDetectorBase::draw_results() {
    Mat img_matches;
    
    drawMatches(query_image, kp_query, input_image, kp_input, good_matches, img_matches, 
                Scalar::all(-1), Scalar::all(-1), vector<char>(), 
                DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    if (!scene_corners.empty()) {
        vector<Point2f> shifted_corners = scene_corners;
        for(size_t i = 0; i < 4; i++) {
            shifted_corners[i] += Point2f((float)query_image.cols, 0);
        }

        line(img_matches, shifted_corners[0], shifted_corners[1], Scalar(0, 255, 0), 4);
        line(img_matches, shifted_corners[1], shifted_corners[2], Scalar(0, 255, 0), 4);
        line(img_matches, shifted_corners[2], shifted_corners[3], Scalar(0, 255, 0), 4);
        line(img_matches, shifted_corners[3], shifted_corners[0], Scalar(0, 255, 0), 4);
    }

    string window_name = "Detection Result";
    namedWindow(window_name, WINDOW_AUTOSIZE);
    imshow(window_name, img_matches);
    cout << "Press any key to exit..." << endl;
    waitKey(0);
}

// Getters
vector<DMatch> ObjectDetectorBase::get_good_matches() const {
    return good_matches;
}

vector<KeyPoint> ObjectDetectorBase::get_kp_query() const {
    return kp_query;
}

vector<KeyPoint> ObjectDetectorBase::get_kp_input() const {
    return kp_input;
}

Mat ObjectDetectorBase::getHomography() const {
    return homography;
}

// ==========================================
// SIFT Implementations
// ==========================================
SiftDetector::SiftDetector(Mat query, Mat input) : ObjectDetectorBase(query, input) {}

void SiftDetector::extract_and_match() {
    cout << "Running SIFT Detector..." << endl;
    
    Ptr<SIFT> detector = SIFT::create();
    Mat des1, des2;

    detector->detectAndCompute(query_image, noArray(), kp_query, des1);
    detector->detectAndCompute(input_image, noArray(), kp_input, des2);

    if (des1.empty() || des2.empty()) {
        cout << "SIFT: No descriptors found." << endl;
        return;
    }

    BFMatcher matcher(NORM_L2, false);
    vector<vector<DMatch>> knn_matches;
    matcher.knnMatch(des1, des2, knn_matches, 2);

    const float ratio_thresh = 0.75f;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    cout << "SIFT Matches found: " << good_matches.size() << endl;
}

// ==========================================
// ORB Implementations
// ==========================================
OrbDetector::OrbDetector(Mat query, Mat input) : ObjectDetectorBase(query, input) {}

void OrbDetector::extract_and_match() {
    cout << "Running ORB Detector..." << endl;

    Ptr<ORB> detector = ORB::create(1500);
    Mat des1, des2;

    detector->detectAndCompute(query_image, noArray(), kp_query, des1);
    detector->detectAndCompute(input_image, noArray(), kp_input, des2);

    if (des1.empty() || des2.empty()) {
        cout << "ORB: No descriptors found." << endl;
        return;
    }

    BFMatcher matcher(NORM_HAMMING, false);
    vector<vector<DMatch>> knn_matches;
    matcher.knnMatch(des1, des2, knn_matches, 2);

    const float ratio_thresh = 0.8f;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    cout << "ORB Matches found: " << good_matches.size() << endl;
}
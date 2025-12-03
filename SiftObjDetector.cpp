#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
// #include <opencv2/xfeatures2d.hpp> 


// NOTE: SURF is part of the opencv_contrib modules (xfeatures2d).
// If you do not have opencv_contrib installed, comment out the SURF-related lines.
// #include <opencv2/xfeatures2d.hpp> 

using namespace cv;
using namespace std;

class FeatureObjectDetector {
public:
    string method;
    Ptr<Feature2D> detector;
    NormTypes normType;
    const int MIN_MATCH_COUNT = 10;

    FeatureObjectDetector(string m) {
        method = m;
        // Convert to uppercase
        transform(method.begin(), method.end(), method.begin(), ::toupper);

        if (method == "SIFT") {
            detector = SIFT::create();
            normType = NORM_L2;
        }
        else if (method == "ORB") {
            detector = ORB::create(1500);
            normType = NORM_HAMMING;
        }
        else if (method == "SURF") {
            // UNCOMMENT BELOW IF YOU HAVE OPENCV_CONTRIB INSTALLED
            // detector = xfeatures2d::SURF::create(400);
            // normType = NORM_L2;
            
            cout << "SURF requires opencv_contrib (xfeatures2d). Defaulting to ORB for safety." << endl;
            detector = ORB::create(1500);
            normType = NORM_HAMMING;
            method = "ORB (Fallback)";
        }
        else {
            cerr << "Unknown method: " << method << ". Defaulting to ORB." << endl;
            detector = ORB::create(1500);
            normType = NORM_HAMMING;
        }

        cout << "Initialized Detector: " << method << endl;
    }

    void detectAndLocate(Mat& queryImg, Mat& trainImg) {
        if (queryImg.empty() || trainImg.empty()) {
            cerr << "Empty images provided!" << endl;
            return;
        }

        vector<KeyPoint> kp1, kp2;
        Mat des1, des2;

        // 1. Detect and Compute
        detector->detectAndCompute(queryImg, noArray(), kp1, des1);
        detector->detectAndCompute(trainImg, noArray(), kp2, des2);

        if (des1.empty() || des2.empty()) {
            cout << "Not enough features found." << endl;
            return;
        }

        // 2. Match Features (KNN)
        BFMatcher matcher(normType, false);
        vector<vector<DMatch>> knn_matches;
        matcher.knnMatch(des1, des2, knn_matches, 2);

        // 3. Ratio Test (Lowe's)
        const float ratio_thresh = (method == "ORB") ? 0.8f : 0.75f;
        vector<DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++) {
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
                good_matches.push_back(knn_matches[i][0]);
            }
        }

        cout << "Matches found: " << good_matches.size() << "/" << knn_matches.size() << endl;

        Mat img_matches;
        
        // 4. Compute Homography & Draw Bounding Box
        if (good_matches.size() >= MIN_MATCH_COUNT) {
            vector<Point2f> obj;
            vector<Point2f> scene;

            for (size_t i = 0; i < good_matches.size(); i++) {
                // Get the keypoints from the good matches
                obj.push_back(kp1[good_matches[i].queryIdx].pt);
                scene.push_back(kp2[good_matches[i].trainIdx].pt);
            }

            Mat H = findHomography(obj, scene, RANSAC);

            if (!H.empty()) {
                // Get corners from the query image (the object to be detected)
                vector<Point2f> obj_corners(4);
                obj_corners[0] = Point2f(0, 0);
                obj_corners[1] = Point2f((float)queryImg.cols, 0);
                obj_corners[2] = Point2f((float)queryImg.cols, (float)queryImg.rows);
                obj_corners[3] = Point2f(0, (float)queryImg.rows);

                vector<Point2f> scene_corners(4);
                perspectiveTransform(obj_corners, scene_corners, H);

                // Draw matches
                drawMatches(queryImg, kp1, trainImg, kp2, good_matches, img_matches, 
                            Scalar::all(-1), Scalar::all(-1), vector<char>(), 
                            DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

                // Draw lines between the corners (the mapped object in the scene)
                // Note: scene_corners need to be shifted because drawMatches places images side-by-side
                for(size_t i = 0; i < 4; i++) {
                    scene_corners[i] += Point2f((float)queryImg.cols, 0);
                }

                line(img_matches, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 4);
                line(img_matches, scene_corners[1], scene_corners[2], Scalar(0, 255, 0), 4);
                line(img_matches, scene_corners[2], scene_corners[3], Scalar(0, 255, 0), 4);
                line(img_matches, scene_corners[3], scene_corners[0], Scalar(0, 255, 0), 4);

                cout << "Object detected! Bounding box computed." << endl;
            } else {
                cout << "Homography failed." << endl;
                drawMatches(queryImg, kp1, trainImg, kp2, good_matches, img_matches);
            }
        } else {
            cout << "Not enough matches found - " << good_matches.size() << "/" << MIN_MATCH_COUNT << endl;
            drawMatches(queryImg, kp1, trainImg, kp2, good_matches, img_matches);
        }

        // Show result
        namedWindow("Feature Detection", WINDOW_AUTOSIZE);
        imshow("Feature Detection", img_matches);
        
        // Wait for key press
        cout << "Press any key to exit..." << endl;
        waitKey(0);
    }
};



int main(int argc, char** argv) {
    Mat queryImg, sceneImg;

    queryImg = imread("/run/media/runtime_terror/mass storage/coding/courses/LearnOpenCV/images/obj_detection/gt.jpeg", IMREAD_GRAYSCALE);
    sceneImg = imread("/run/media/runtime_terror/mass storage/coding/courses/LearnOpenCV/images/obj_detection/s4.jpeg", IMREAD_GRAYSCALE);

    // 2. Select Method
    // Options: "SIFT", "ORB", 
    string method = "SIFT"; 
    if (argc > 1) method = argv[1];

    FeatureObjectDetector detector(method);

    // 3. Run Detection
    detector.detectAndLocate(queryImg, sceneImg);

    return 0;
}
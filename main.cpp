#include <iostream>
#include <algorithm> // For transform
#include "ObjectDetector.h"

using namespace cv;
using namespace std;



int main(int argc, char** argv) {
    Mat queryImg, sceneImg;

    queryImg = imread("/run/media/runtime_terror/mass storage/coding/courses/ObjectDetection/images/obj_detection/gt.jpeg", IMREAD_GRAYSCALE);
    sceneImg = imread("/run/media/runtime_terror/mass storage/coding/courses/ObjectDetection/images/obj_detection/s4.jpeg", IMREAD_GRAYSCALE);
    
    cout<<"Please press 1 for SIFT and 2 for ORB"<<endl;
    int choice;
    cin>>choice;
    string method = "ORB";
    if(choice == 1) {
        method = "SIFT";
    } else if(choice == 2) {
        method = "ORB";
    } else {
        cout << "Invalid choice, defaulting to ORB." << endl;
    }


    // string Method = "SIFT";
    ObjectDetectorBase* detector = nullptr;

    if (method == "SIFT") {
        detector = new SiftDetector(queryImg, sceneImg);
    } else {
        if (method != "ORB") cout << "Unknown method, defaulting to ORB." << endl;
        detector = new OrbDetector(queryImg, sceneImg);
    }

    if (detector) {
        detector->extract_and_match();
        detector->calc_homography();
        detector->calc_bbx_coordinate();
        
        Mat homography = detector->getHomography();
        cout<<"Computed Homography Matrix:"<<endl;
        cout<<homography<<endl;
        cout<<"-------------------------------"<<endl;
        vector<Point2f> bbx = detector->calc_bbx_coordinate();
        cout << "Bounding Box Coordinates:" << endl;
        for (const auto& point : bbx) {
            cout << point << endl;
        }
        cout<<"-------------------------------"<<endl;
        
        detector->draw_results();
        
        delete detector;
        detector = nullptr;
    }

    return 0;
}
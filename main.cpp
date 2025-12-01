#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 1. Define the path to your image
    // Make sure 'sample.jpg' exists in the same directory, or provide an absolute path.
    std::string imagePath = "sample.jpg";

    // 2. Read the image
    // cv::imread loads the image into a Matrix object (cv::Mat)
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);

    // 3. Error Handling
    // Always check if the image is empty. A common error is a wrong file path.
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image at " << imagePath << std::endl;
        return -1;
    }

    // 4. Process the Image
    // Convert from BGR (Blue-Green-Red) to Grayscale
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    
    // 

    // 5. Display the Images
    // Create windows to display the result
    cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Grayscale Image", cv::WINDOW_AUTOSIZE);

    cv::imshow("Original Image", image);
    cv::imshow("Grayscale Image", grayImage);

    // 6. Wait for User Input
    // waitKey(0) blocks the code until a key is pressed. 
    // Without this, the program terminates immediately and windows close.
    std::cout << "Press any key to exit..." << std::endl;
    cv::waitKey(0);

    return 0;
}
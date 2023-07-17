#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

#compile - g++ -I/usr/local/include/opencv4/ -L/usr/local/lib/ -g -o face_detection face_detection.cpp -lopencv_core -lopencv_highgui -lopencv_videoio -lopencv_imgproc -lopencv_objdetect

int main()
{
    cv::VideoCapture cap("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)256, height=(int)144,format=(string)NV12, framerate=(fraction)15/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink");

    if (!cap.isOpened())
    {
        std::cout << "Failed to open the camera." << std::endl;
        return -1;
    }

    cv::namedWindow("Camera Feed", cv::WINDOW_NORMAL);
    cv::resizeWindow("Camera Feed", 640, 480);

    // Load the pre-trained face cascade
    cv::CascadeClassifier faceCascade;
    if (!faceCascade.load("haarcascade_frontalface_default.xml"))
    {
        std::cout << "Failed to load the face cascade." << std::endl;
        return -1;
    }

    int noFaceCount = 0;
    const int maxNoFaceCount = 10100000; // Maximum consecutive frames without a face

    while (true)
    {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty())
        {
            std::cout << "Failed to capture frame." << std::endl;
            break;
        }

        cv::Mat grayFrame;
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        std::vector<cv::Rect> faces;
        faceCascade.detectMultiScale(grayFrame, faces, 1.1, 3, 0, cv::Size(30, 30));

        if (!faces.empty())
        {
            noFaceCount = 0;
            // Draw rectangles around detected faces
            for (const auto& face : faces)
            {
                cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);
            }
        }
        else
        {
            noFaceCount++;
            if (noFaceCount >= maxNoFaceCount)
            {
                std::cout << "No face detected for " << maxNoFaceCount << " consecutive frames. Exiting." << std::endl;
                break;
            }
        }

        cv::imshow("Camera Feed", frame);

        if (cv::waitKey(1) == 'q')
            break;
    }

    cv::destroyAllWindows();

    return 0;
}


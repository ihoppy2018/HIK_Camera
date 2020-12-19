#include "string.h"
#include "time.h"
#include <zbar.h>
#include "opencv2/opencv.hpp"
#include <iostream>
#include "windows.h"
#include "string"


using namespace cv;
using namespace zbar;
using namespace std;

cv::Mat detect_bar_auto(cv::Mat input)
{
    if (input.empty())
    {
        cout << "could not load image..." << endl;
        return Mat();
    }
    Mat src;
    if (input.channels() == 1)
    {
        cvtColor(input, src, COLOR_GRAY2BGR);
    }
    else
    {
        src = input;
    }

    Mat draw2 = src.clone();
    Mat gray, gauss_img;
    cvtColor(src, gray, COLOR_BGR2GRAY);

    GaussianBlur(gray, gauss_img, Size(3, 3), 1, 1);
    Mat bin;
    threshold(gauss_img, bin, 100, 255, THRESH_BINARY_INV | THRESH_OTSU);
    Mat dilate_bin;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(7, 7));
    dilate(bin, dilate_bin, kernel, Point(-1, -1), 5);

    vector<vector<Point>> contours;
    vector<Vec4i> hierachy;

    // ��������
    findContours(dilate_bin, contours, hierachy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

    // ��������
    Mat draw1 = Mat::zeros(src.rows, src.cols, CV_8UC3);

    vector<Mat> barcodes;
    vector<Point> rBoxCenter;
    for (int i = 0; i < contours.size(); i++)
    {
        vector<Point> cn = contours[i];
        int area = contourArea(cn);
        int length = arcLength(cn, true);
        if ((area > 250000) || (area < 150000)) continue;

        Scalar color = Scalar(0, 0, 255);
        drawContours(draw1, contours, i, color, 4, LINE_4, hierachy, 0, Point(0, 0));
        // ���Ƕ�
        RotatedRect rBox = minAreaRect(cn);
        // ����ת����
        Mat vertices;
        boxPoints(rBox, vertices);
        for (auto i = 0; i < vertices.rows; ++i)
        {
            Point p1 = Point(vertices.at<float>(i, 0), vertices.at<float>(i, 1));
            Point p2 = Point(vertices.at<float>((i + 1) % 4, 0), vertices.at<float>((i + 1) % 4, 1));
            line(draw2, p1, p2, Scalar(255, 0, 0), 6);
        }
        // ���src
        Mat dst, src_border;
        int w = src.cols;
        int h = src.rows;
        copyMakeBorder(src, src_border, h / 2, h / 2, w / 2, w / 2, BORDER_CONSTANT, Scalar(0, 0, 0));
        // ����任����
        Point newCenter = Point(rBox.center.x + w / 2, rBox.center.y + h / 2);
        Mat m = getRotationMatrix2D(newCenter, rBox.angle, 1.0);

        // ����任
        warpAffine(src_border, dst, m, src_border.size());
        Point ltPt = Point((rBox.center.x - rBox.size.width / 2.0 + w / 2.0),
            (rBox.center.y - rBox.size.height / 2.0) + h / 2.0);
        Rect newBox = Rect(ltPt.x, ltPt.y, rBox.size.width, rBox.size.height);
        //rectangle(dst, newBox, Scalar(0, 255, 0), 6);
        barcodes.push_back(dst(newBox));
        rBoxCenter.push_back(rBox.center);

        // �����Ƕȵ�
        Rect box = rBox.boundingRect();
        //rectangle(draw2, box, Scalar(0, 255, 0), 6);
        //barcodes.push_back(src(box));
    }

    // ʶ��
    // �����Ƕȵ�Rect + �ֶ�ROI + ��תRect
    /*namedWindow("select ROI", WINDOW_NORMAL);
    vector<Rect> boxs;
    selectROIs("select ROI", src, boxs, false, false);
    destroyWindow("select ROI");
    for (int i = 0; i < boxs.size(); i++)
    {
        barcodes.push_back(src(boxs[i]));
    }*/

    if (barcodes.empty())
    {
        cout << "could not find barcodes..." << endl;
        return draw2;
    }
    //namedWindow("result", WINDOW_FREERATIO);

    for (int i = 0; i < barcodes.size(); i++)
    {
        //Mat barcode = barcodes[i].clone();
        Mat barcode = barcodes[i];

        cvtColor(barcode, barcode, COLOR_BGR2GRAY); // ת�Ҷȣ�ͬʱҲ���ԶϿ���dst�Ĺ�ϵ
        ImageScanner scanner;
        scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);
        int width = barcode.step;
        int height = barcode.rows;
        uchar* raw = (uchar*)barcode.data;
        Image imageZbar(width, height, "Y800", raw, width * height);
        scanner.scan(imageZbar); //ɨ������
        Image::SymbolIterator symbol = imageZbar.symbol_begin();
        if (imageZbar.symbol_begin() == imageZbar.symbol_end())
        {
            cout << "��ѯ����ʧ�ܣ�����ͼƬ��" << endl;
        }
        for (; symbol != imageZbar.symbol_end(); ++symbol)
        {
            cout << "���ͣ�" << endl << symbol->get_type_name() << endl << endl;
            cout << "���룺" << endl << symbol->get_data() << endl << endl;
            //cout << "distance:" << distance(symbol, imageZbar.symbol_end()) << endl;
            putText(draw2, symbol->get_data(), rBoxCenter[i], FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 2);
        }
        imageZbar.set_data(NULL, 0);
    }
    //imshow("result", draw2);
    //string filename = GetTimeAsFileName();
    //dstSavePathName.append(filename);
    //imwrite(dstSavePathName, draw2);
    return draw2;
}

// ������
void saveResult(cv::Mat img, std::string path)
{
    bool isSave = cv::imwrite(path, img);
    if (!isSave)
    {
        std::cout << "save failed..." << std::endl;
    }
    else
    {
        std::cout << "save finished..." << std::endl;
    }
}

int main(int argc, char** argv)
{
    cv::Mat image = cv::imread("./pic_test2/test.bmp");
    cv::Mat algo_img = detect_bar_auto(image);

    waitKey(0);
    return 0;
}
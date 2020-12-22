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

// �Զ�ʶ���ά��
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
    vector<float> rBoxAngle;

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
        rBoxAngle.push_back(rBox.angle);

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
        scanner.set_config(ZBAR_QRCODE, ZBAR_CFG_ENABLE, 1);
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
            cout << "angle:" << rBoxAngle[i] << endl;
            
            //cout << "distance:" << distance(symbol, imageZbar.symbol_end()) << endl;
            putText(draw2, format("%s", symbol->get_data().c_str()), rBoxCenter[i], FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 2);
            putText(draw2, format("angle %f", rBoxAngle[i]), Point(rBoxCenter[i].x, rBoxCenter[i].y + 60), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 2);
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

//��ȡ���������ĵ�
Point Center_cal(vector<vector<Point> > contours, int i) {
    int centerx = 0, centery = 0, n = static_cast<int>(contours[i].size());
    //����ȡ��С�����εı߽���ÿ���ܳ���������ȡһ��������꣬
    //������ȡ�ĸ����ƽ�����꣨��ΪС�����εĴ������ģ�
    centerx = (contours[i][n / 4].x + contours[i][n * 2 / 4].x + contours[i][3 * n / 4].x + contours[i][n - 1].x) / 4;
    centery = (contours[i][n / 4].y + contours[i][n * 2 / 4].y + contours[i][3 * n / 4].y + contours[i][n - 1].y) / 4;
    Point point1 = Point(centerx, centery);
    return point1;
}

// �����������е�  ���ڶԽ����ϵ�������

//�Զ�ʶ���ά�� ������
cv::Mat detect_decode_qrcode(cv::Mat input)
{
    if (input.empty())
    {
        cout << "input image is empty" << endl;
        return Mat();
    }

    Mat img;
    if (input.channels() == 1)
    {
        cvtColor(input, img, COLOR_GRAY2BGR);
    }
    else
    {
        img = input;
    }

    //�ҶȻ�
    Mat draw1 = img.clone();
    Mat img_gray, img_bin;
    cvtColor(img, img_gray, COLOR_BGR2GRAY);

    threshold(img_gray, img_bin, 100, 255, THRESH_BINARY_INV | THRESH_OTSU);  //THRESH_BINARY_INV  ��ֵ��ȡ��	
    /*Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    erode(img_bin, img_bin, kernel, Point(-1, -1), 3);*/
    vector<vector<Point>> contours, contours2; //������  �ҵ��������������ķ�ʽ����
    vector<Vec4i> hierarchy;
    findContours(img_bin, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
    int c = 0, ic = 0, area = 0;
    int parentIdx = -1;
    
    for (int i = 0; i < contours.size(); i++)
    {											//�������еĴ�����									
        if (hierarchy[i][2] != -1 && ic == 0)  //��� ���������û�и����� hierarchy[i][2] != -1 ˵�����Ǵ�����������
        {
            parentIdx = i;
            ic++;
        }
        else if (hierarchy[i][2] != -1)
        {
            ic++;
        }
        //���������0
        else if (hierarchy[i][2] == -1)
        {
            ic = 0;
            parentIdx = -1;
        }
        //�ҵ���λ����Ϣ
        if (ic == 2)
        {
            contours2.push_back(contours[parentIdx]);
            // ����������λ��
            drawContours(draw1, contours, parentIdx, Scalar(0, 0, 255), 4);
            ic = 0;
            parentIdx = -1;
        }
    }

    //��ά���м���Ӧ�����������������ģ��������3 ��ô����Ϊ�����ж�ά���
    if (contours2.size() != 3)
    {
        printf("finding 3 rects fails \n");
    }

    //��ȡ������λ�ǵ���������
    Point cenPts[3];
    for (int i = 0; i < contours2.size(); i++) {
        cenPts[i] = Center_cal(contours2, i);
    }

   
    for (int i = 0; i < contours2.size(); i++) {
        //����������λ�ǵ���������
        line(draw1, cenPts[i % contours2.size()], cenPts[(i + 1) % contours2.size()], Scalar(255, 0, 0), 2, 8);
    }

    // �����ά�����ת�Ƕ�
    float max_distance = 0;
    // ǰ����Ϊ�Խ���Ԫ�� ��һ��Ϊyֵ��С�ĵ� ������Ϊ���Խ��ߵ����һ����
    Point diagPts[3];  

    for (int i = 0; i < contours2.size(); i++)
    {
        float dis = pow(cenPts[i%3].x - cenPts[(i + 1)%3].x, 2) + pow(cenPts[i%3].y - cenPts[(i + 1)%3].y, 2);
        if (dis > max_distance)
        {
            max_distance = dis;
           
            if (cenPts[i % 3].y < cenPts[(i + 1) % 3].y)
            {
                diagPts[0] = cenPts[i % 3];
                diagPts[1] = cenPts[(i + 1) % 3];
            }
            else
            {
                diagPts[1] = cenPts[i % 3];
                diagPts[0] = cenPts[(i + 1) % 3];
            }
            diagPts[2] = cenPts[(i + 2) % 3];
        }
    }
    circle(draw1, diagPts[0], 15, Scalar(0, 0, 255), -1);
    circle(draw1, diagPts[1], 15, Scalar(0, 0, 255), -1);
    Point midPt = Point((diagPts[0].x + diagPts[1].x) / 2, (diagPts[0].y + diagPts[1].y) / 2);
    // theta = 
    float theta = 0;
    // ��֤��������յ�һ��
    if (midPt.x > diagPts[2].x)
    {
        // һ������
        theta = atan2(diagPts[1].y - diagPts[0].y, diagPts[1].x - diagPts[0].x) * 180 / 3.1415926;
    }
    else
    {
        // ��������
        theta = atan2(diagPts[0].y - diagPts[1].y, diagPts[0].x - diagPts[1].x) * 180 / 3.1415926 + 360.0;

    }
    
     // ����ϵת�� y��-y,����theta = arctan(x)��˵��ҲӦ��ȡ�� -��theta - 45�� ���ϣ������˱�׼λ�ã����ټ�90���
     float angle = - theta  + 45 + 90;
    
    cout << "�Խ��߽Ƕ�theta����������ϵ�£�:" << theta << endl;
    cout << "������ĽǶ�angle��" << angle << endl;

    // ��ö�ά������
    Mat erode_img;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(11, 11));
    dilate(img_bin, erode_img, kernel, Point(-1, -1), 5);

    vector<vector<Point>> contours_erode, contours2_erode; //������  �ҵ��������������ķ�ʽ����
    vector<Vec4i> hierarchy_erode;
    vector<Mat> barcodes;
    vector<Point> rBoxCenter;
    vector<float> rBoxAngle;
    findContours(erode_img, contours_erode, hierarchy_erode, RETR_TREE, CHAIN_APPROX_NONE);
    for (int i = 0; i < contours_erode.size(); i++)
    {
        vector<Point> cn = contours_erode[i];
        //drawContours(draw1, contours_erode, i, Scalar(255, 155, 100), 4);
        float cnArea = contourArea(cn);
        if ((cnArea > 610000) || (cnArea < 500000)) continue;
        contours2_erode.push_back(cn);

        Scalar color = Scalar(0, 0, 255);
        drawContours(draw1, contours, i, color, 4, LINE_4, hierarchy_erode, 0, Point(0, 0));
        // ���Ƕ�
        RotatedRect rBox = minAreaRect(cn);
        // ����ת����
        Mat vertices;
        boxPoints(rBox, vertices);
        for (auto i = 0; i < vertices.rows; ++i)
        {
            Point p1 = Point(vertices.at<float>(i, 0), vertices.at<float>(i, 1));
            Point p2 = Point(vertices.at<float>((i + 1) % 4, 0), vertices.at<float>((i + 1) % 4, 1));
            line(draw1, p1, p2, Scalar(255, 0, 0), 6);
        }
        // ���src
        Mat dst, src_border;
        int w = img.cols;
        int h = img.rows;
        copyMakeBorder(img, src_border, h / 2, h / 2, w / 2, w / 2, BORDER_CONSTANT, Scalar(0, 0, 0));
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
        rBoxAngle.push_back(rBox.angle);

        // �����Ƕȵ�
        Rect box = rBox.boundingRect();
        //rectangle(draw2, box, Scalar(0, 255, 0), 6);
        //barcodes.push_back(src(box));
    }

    //�Ѷ�ά������������������һ���µĵ㼯 ��ʵû�õ�����ģ���Ϊ�������������ά������Ļ�����ά���ɨ�費����
    //Rect new_rect;
    //vector<Point> all_points;
    //
    //for (int i = 0; i < contours2_erode.size(); i++)
    //{
    //    drawContours(draw1, contours2_erode, i, Scalar(255, 0, 100), 4);
    //    for (int j = 0; j < contours2_erode[i].size(); j++)
    //        all_points.push_back(contours2_erode[i][j]);
    //}

    //new_rect = boundingRect(all_points);  //���ݶ�ά�빹�ɵõ㼯���ҵ�һ����С��������е㼯 �ľ���
    //                                      //  Rect rect(230, 5, 280, 290);//�������꣨x,y���;��εĳ�(x)��(y)
    //                                     //  cv::rectangle(src, rect, Scalar(255, 0, 0),1, LINE_8,0);
    //cv::rectangle(draw1, new_rect, Scalar(0, 0, 255), 8, 0);
    //Mat result_img = img_gray(new_rect);   //���ҵ��ľ��� �Ž��Ҷ�ͼ�У�����ͼƬ�Ϳ��Ը��ݾ����и������
    
    if (barcodes.size() == 0)
    {
        cout << "could not find Qtbar" << endl;
        return draw1;
    }
    Mat result_img = barcodes[0];
    ImageScanner scanner;
    scanner.set_config(ZBAR_QRCODE, ZBAR_CFG_ENABLE, 1);
    int width = result_img.step;  //��Ϊ��һС�����ǽ�ȡ������
    int height = result_img.rows;
    uchar* raw = (uchar*)result_img.data;
    Image imageZbar(width, height, "Y800", raw, width * height);
    scanner.scan(imageZbar);
    Image::SymbolIterator symbol = imageZbar.symbol_begin();
    if (imageZbar.symbol_begin() == imageZbar.symbol_end())
    {
        cout << "��ѯ��ά��ʧ�ܣ�����ͼƬ��" << endl;
    }
    for (; symbol != imageZbar.symbol_end(); ++symbol)
    {
        cout << "���ͣ�" << endl << symbol->get_type_name() << endl << endl;
        cout << "��ά�룺" << endl << symbol->get_data() << endl << endl;

        putText(draw1, format("%s", symbol->get_data().c_str()), midPt, FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 2);
        putText(draw1, format("angle %f", angle), Point(midPt.x, midPt.y + 60), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 2);
    }
    imageZbar.set_data(NULL, 0);
    //imshow("mat",img);
    //imshow("mat1", result_img);
    return draw1;
}

int main(int argc, char** argv)
{
    cv::Mat image = cv::imread("./pic_test2/pos6.bmp");
    //cv::Mat algo_img = detect_bar_auto(image);
    cv::Mat algo_img = detect_decode_qrcode(image);
    waitKey(0);
    return 0;
}
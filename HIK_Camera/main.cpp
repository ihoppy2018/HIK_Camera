#include <stdio.h>
#include <process.h>
#include <conio.h>
#include "windows.h"
#include "MvCameraControl.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "string.h"
#include "time.h"
#include <zbar.h>
#include <iostream>

using namespace zbar;
using namespace std;
using namespace cv;

HWND g_hwnd = NULL;
bool g_bExit = false;
unsigned int g_nPayloadSize = 0;

char filename[100] = { 0 };
std::string srcSavePathName = "./pictures_src/";
std::string dstSavePathName = "./pictures_dst/";

// 以时间作为图片名 精确到毫秒
void GetTimeAsFileName()
{
    SYSTEMTIME st = { 0 };
    GetLocalTime(&st);  //获取当前时间 可精确到ms
    sprintf(filename, "%d%02d%02d_%02d%02d%02d%03d.bmp", 
        st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
}

// 识别二维码 手动ROI模式
void detect_bar(cv::Mat inputImg)
{
    namedWindow("select ROI", WINDOW_NORMAL);
    cv::Rect box = selectROI("select ROI", inputImg, false, false);
    destroyWindow("select ROI");
    cv::Mat result_img = inputImg(box);
    if (result_img.empty())
    {
        cout << "请重新选择ROI区域" << endl;
        return;
    }
    cv::imshow("bar", result_img);

    ImageScanner scanner;
    scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);
    int width = result_img.step;
    int height = result_img.rows;
    uchar* raw = (uchar*)result_img.data;
    Image imageZbar(width, height, "Y800", raw, width * height);
    scanner.scan(imageZbar); //扫描条码      
    Image::SymbolIterator symbol = imageZbar.symbol_begin();
    if (imageZbar.symbol_begin() == imageZbar.symbol_end())
    {
        cout << "查询条码失败，请检查图片！" << endl;
    }
    for (; symbol != imageZbar.symbol_end(); ++symbol)
    {
        cout << "类型：" << endl << symbol->get_type_name() << endl;
        cout << "条码：" << endl << symbol->get_data() << endl;
    }
    //cv::putText();
    imageZbar.set_data(NULL, 0);
}

// 识别二维码 自动识别模式
void detect_bar_auto(cv::Mat input)
{
    if (input.empty())
    {
        cout << "could not load image..." << endl;
        return;
    }
    Mat src;
    cvtColor(input, src, COLOR_GRAY2BGR);
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

    // 轮廓发现
    findContours(dilate_bin, contours, hierachy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

    // 轮廓分析
    Mat draw1 = Mat::zeros(src.rows, src.cols, CV_8UC3);

    vector<Mat> barcodes;
    vector<Point> rBoxCenter;
    for (int i = 0; i < contours.size(); i++)
    {
        vector<Point> cn = contours[i];
        int area = contourArea(cn);
        int length = arcLength(cn, true);
        if ((area > 150000) || (area < 10000)) continue;

        Scalar color = Scalar(0, 0, 255);
        drawContours(draw1, contours, i, color, 4, LINE_4, hierachy, 0, Point(0, 0));
        // 带角度
        RotatedRect rBox = minAreaRect(cn);
        // 画旋转矩形
        Mat vertices;
        boxPoints(rBox, vertices);
        for (auto i = 0; i < vertices.rows; ++i)
        {
            Point p1 = Point(vertices.at<float>(i, 0), vertices.at<float>(i, 1));
            Point p2 = Point(vertices.at<float>((i + 1) % 4, 0), vertices.at<float>((i + 1) % 4, 1));
            line(draw2, p1, p2, Scalar(255, 0, 0), 6);
        }
        // 填充src
        Mat dst, src_border;
        int w = src.cols;
        int h = src.rows;
        copyMakeBorder(src, src_border, h / 2, h / 2, w / 2, w / 2, BORDER_CONSTANT, Scalar(0, 0, 0));
        // 计算变换矩阵
        Point newCenter = Point(rBox.center.x + w / 2, rBox.center.y + h / 2);
        Mat m = getRotationMatrix2D(newCenter, rBox.angle, 1.0);

        // 仿射变换
        warpAffine(src_border, dst, m, src_border.size());
        Point ltPt = Point((rBox.center.x - rBox.size.width / 2.0 + w / 2.0),
            (rBox.center.y - rBox.size.height / 2.0) + h / 2.0);
        Rect newBox = Rect(ltPt.x, ltPt.y, rBox.size.width, rBox.size.height);
        //rectangle(dst, newBox, Scalar(0, 255, 0), 6);
        barcodes.push_back(dst(newBox));
        rBoxCenter.push_back(rBox.center);

        // 不带角度的
        Rect box = rBox.boundingRect();
        //rectangle(draw2, box, Scalar(0, 255, 0), 6);
        //barcodes.push_back(src(box));
    }

    // 识别
    // 不带角度的Rect + 手动ROI + 旋转Rect
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
        return;
    }
    namedWindow("result", WINDOW_FREERATIO);

    for (int i = 0; i < barcodes.size(); i++)
    {
        //Mat barcode = barcodes[i].clone();
        Mat barcode = barcodes[i];

        cvtColor(barcode, barcode, COLOR_BGR2GRAY); // 转灰度，同时也可以断开和dst的关系
        ImageScanner scanner;
        scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);
        int width = barcode.step;
        int height = barcode.rows;
        uchar* raw = (uchar*)barcode.data;
        Image imageZbar(width, height, "Y800", raw, width * height);
        scanner.scan(imageZbar); //扫描条码      
        Image::SymbolIterator symbol = imageZbar.symbol_begin();
        if (imageZbar.symbol_begin() == imageZbar.symbol_end())
        {
            cout << "查询条码失败，请检查图片！" << endl;
        }
        for (; symbol != imageZbar.symbol_end(); ++symbol)
        {
            cout << "类型：" << endl << symbol->get_type_name() << endl << endl;
            cout << "条码：" << endl << symbol->get_data() << endl << endl;
            //cout << "distance:" << distance(symbol, imageZbar.symbol_end()) << endl;
            putText(draw2, symbol->get_data(), rBoxCenter[i], FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 2);
        }

        imageZbar.set_data(NULL, 0);
    }
    imshow("result", draw2);
    dstSavePathName.append(filename);
    imwrite(dstSavePathName, draw2);
}



// ch:等待按键输入 | en:Wait for key press
void WaitForKeyPress(void)
{
    while (!_kbhit())
    {
        Sleep(10);
    }
    _getch();
}

bool PrintDeviceInfo(MV_CC_DEVICE_INFO* pstMVDevInfo)
{
    if (NULL == pstMVDevInfo)
    {
        printf("The Pointer of pstMVDevInfo is NULL!\n");
        return false;
    }
    if (pstMVDevInfo->nTLayerType == MV_GIGE_DEVICE)
    {
        int nIp1 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24);
        int nIp2 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16);
        int nIp3 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8);
        int nIp4 = (pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff);

        // ch:打印当前相机ip和用户自定义名字 | en:print current ip and user defined name
        printf("CurrentIp: %d.%d.%d.%d\n", nIp1, nIp2, nIp3, nIp4);
        printf("UserDefinedName: %s\n\n", pstMVDevInfo->SpecialInfo.stGigEInfo.chUserDefinedName);
        printf("DeviceVersion: %s\n\n", pstMVDevInfo->SpecialInfo.stGigEInfo.chDeviceVersion);
        printf("ManufacturerName: %s\n\n", pstMVDevInfo->SpecialInfo.stGigEInfo.chManufacturerName);


    }
    else if (pstMVDevInfo->nTLayerType == MV_USB_DEVICE)
    {
        printf("UserDefinedName: %s\n", pstMVDevInfo->SpecialInfo.stUsb3VInfo.chUserDefinedName);
        printf("Serial Number: %s\n", pstMVDevInfo->SpecialInfo.stUsb3VInfo.chSerialNumber);
        printf("Device Number: %d\n\n", pstMVDevInfo->SpecialInfo.stUsb3VInfo.nDeviceNumber);
    }
    else
    {
        printf("Not support.\n");
    }

    return true;
}

int RGB2BGR(unsigned char* pRgbData, unsigned int nWidth, unsigned int nHeight)
{
    if (NULL == pRgbData)
    {
        return MV_E_PARAMETER;
    }

    for (unsigned int j = 0; j < nHeight; j++)
    {
        for (unsigned int i = 0; i < nWidth; i++)
        {
            unsigned char red = pRgbData[j * (nWidth * 3) + i * 3];
            pRgbData[j * (nWidth * 3) + i * 3] = pRgbData[j * (nWidth * 3) + i * 3 + 2];
            pRgbData[j * (nWidth * 3) + i * 3 + 2] = red;
        }
    }

    return MV_OK;
}


// convert data stream in Mat format
bool Convert2Mat(MV_FRAME_OUT_INFO_EX* pstImageInfo, unsigned char* pData, cv::Mat& mat_img)
{
    cv::Mat srcImage;
    if (pstImageInfo->enPixelType == PixelType_Gvsp_Mono8)
    {
        srcImage = cv::Mat(pstImageInfo->nHeight, pstImageInfo->nWidth, CV_8UC1, pData);
    }
    else if (pstImageInfo->enPixelType == PixelType_Gvsp_RGB8_Packed)
    {
        RGB2BGR(pData, pstImageInfo->nWidth, pstImageInfo->nHeight);
        srcImage = cv::Mat(pstImageInfo->nHeight, pstImageInfo->nWidth, CV_8UC3, pData);
    }
    else
    {
        printf("unsupported pixel format\n");
        return false;
    }

    if (NULL == srcImage.data)
    {
        return false;
    }

    //save converted image in a local file
    GetTimeAsFileName();
    try {
#if defined (VC9_COMPILE)
        cvSaveImage(filename, &(IplImage(srcImage)));
#else
        srcSavePathName.append(filename);
        //cv::imwrite(srcSavePathName, srcImage);
#endif
    }
    catch (cv::Exception& ex) {
        fprintf(stderr, "Exception saving image to bmp format: %s\n", ex.what());
    }
    mat_img = srcImage.clone();
    srcImage.release();

    return true;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_DESTROY:
        PostQuitMessage(0);
        g_hwnd = NULL;
        break;
    }

    return DefWindowProc(hWnd, msg, wParam, lParam);
}

static  unsigned int __stdcall CreateRenderWindow(void* pUser)
{
    HINSTANCE hInstance = ::GetModuleHandle(NULL);              //获取应用程序的模块句柄
    WNDCLASSEX wc;
    wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW;              //窗口的风格
    wc.cbClsExtra = 0;
    wc.cbWndExtra = 0;
    wc.hInstance = hInstance;
    wc.hIcon = ::LoadIcon(NULL, IDI_APPLICATION);    //图标风格
    wc.hIconSm = ::LoadIcon(NULL, IDI_APPLICATION);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);          //背景色
    wc.hCursor = ::LoadCursor(NULL, IDC_ARROW);        //鼠标风格
    wc.lpfnWndProc = WndProc;                              //自定义消息处理函数
    wc.lpszMenuName = NULL;
    wc.lpszClassName = "RenderWindow";                       //该窗口类的名称

    if (!RegisterClassEx(&wc))
    {
        return 0;
    }

    DWORD style = WS_OVERLAPPEDWINDOW;
    DWORD styleEx = WS_EX_APPWINDOW | WS_EX_WINDOWEDGE;
    RECT rect = { 0, 0, 640, 480 };

    AdjustWindowRectEx(&rect, style, false, styleEx);

    HWND hWnd = CreateWindowEx(styleEx, "RenderWindow", "Display", style, 0, 0,
        rect.right - rect.left, rect.bottom - rect.top, NULL, NULL, hInstance, NULL);
    if (hWnd == NULL)
    {
        return 0;
    }

    ::UpdateWindow(hWnd);
    ::ShowWindow(hWnd, SW_SHOW);

    g_hwnd = hWnd;

    MSG msg = { 0 };
    while (msg.message != WM_QUIT)
    {
        if (PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    return 0;
}

static  unsigned int __stdcall WorkThread(void* pUser)
{
    int nRet = MV_OK;

    MV_FRAME_OUT_INFO_EX stImageInfo = { 0 };
    MV_DISPLAY_FRAME_INFO stDisplayInfo = { 0 };
    unsigned char* pData = (unsigned char*)malloc(sizeof(unsigned char) * (g_nPayloadSize));
    if (pData == NULL)
    {
        return 0;
    }
    unsigned int nDataSize = g_nPayloadSize;
   /* cv::namedWindow("test_img", cv::WINDOW_FREERATIO);
    cv::namedWindow("edge_img", cv::WINDOW_FREERATIO);*/
    cv::namedWindow("bar", cv::WINDOW_FREERATIO);


    for (int i = 0; ; i++)
    {
        cv::waitKey(0);
        nRet = MV_CC_GetOneFrameTimeout(pUser, pData, nDataSize, &stImageInfo, 1000);
        if (nRet == MV_OK)
        {
            printf("Get One Frame: Width[%d], Height[%d], nFrameNum[%d]\n",
                stImageInfo.nWidth, stImageInfo.nHeight, stImageInfo.nFrameNum);

            if (g_hwnd)
            {
                stDisplayInfo.hWnd = g_hwnd;
                stDisplayInfo.pData = pData;
                stDisplayInfo.nDataLen = stImageInfo.nFrameLen;
                stDisplayInfo.nWidth = stImageInfo.nWidth;
                stDisplayInfo.nHeight = stImageInfo.nHeight;
                stDisplayInfo.enPixelType = stImageInfo.enPixelType;

                MV_CC_DisplayOneFrame(pUser, &stDisplayInfo);

                // 数据去转换
                bool bConvertRet = false;
                cv::Mat mat_img;
                
                bConvertRet = Convert2Mat(&stImageInfo, pData, mat_img);
                cv::imshow("bar", mat_img);
                // print result
                if (bConvertRet)
                {
                    printf("OpenCV format convert finished.\n");
                    //cv::threshold(mat_img, threshold_img, 100, 255, cv::THRESH_BINARY);
                    //cv::cvtColor(mat_img, gray, cv::COLOR_BGR2GRAY);
                    //cv::Canny(mat_img, edge, 100, 200, 3);
                    //cv::imshow("test_img", threshold_img);
                    //cv::imshow("edge_img", edge);

                    //detect_bar(mat_img);
                    detect_bar_auto(mat_img);

                     /*free(pData);
                     pData = NULL;*/
                }
                else
                {
                    printf("OpenCV format convert failed.\n");
                   /* free(pData);
                    pData = NULL;
                    break;*/
                }
            }
        }
        else
        {
            printf("No data[0x%x]\n", nRet);
        }
        if (g_bExit)
        {
            break;
        }
        
    }
    
    cv::destroyAllWindows();
    free(pData);

    return 0;
}

int main()
{
    int nRet = MV_OK;   // 
    void* handle = NULL;

    do
    {
        // 获取SDK版本信息
        unsigned int sdk_V = MV_CC_GetSDKVersion();
        printf("SDK version is [0x%x]\n", sdk_V);
        // 获取支持的传输层
        int tls = MV_CC_EnumerateTls();
        printf("surpport tls is: [%d]\n", tls);
        // 根据厂商名字枚举设备
       /* MV_CC_DEVICE_INFO_LIST stDeviceList_ex = { 0 };
        nRet = MV_CC_EnumDevicesEx(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList_ex, "hikvision");
        if (MV_OK != nRet)
        {
            printf("Enum Devices with ManufacturerName fail! nRet [0x%x]\n", nRet);
            break;
        }*/

        // ch:枚举设备 | en:Enum device
        MV_CC_DEVICE_INFO_LIST stDeviceList = { 0 };
        nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
        if (MV_OK != nRet)
        {
            printf("Enum Devices fail! nRet [0x%x]\n", nRet);
            break;
        }

        if (stDeviceList.nDeviceNum > 0)
        {
            for (unsigned int i = 0; i < stDeviceList.nDeviceNum; i++)
            {
                printf("[device %d]:\n", i);
                MV_CC_DEVICE_INFO* pDeviceInfo = stDeviceList.pDeviceInfo[i];
               /* bool access = MV_CC_IsDeviceAccessible(pDeviceInfo, 1);
                if (!access)
                {
                    printf("[device %d]:could not access... nRet:[0x%x]\n", i, nRet);
                }*/
                if (NULL == pDeviceInfo)
                {
                    break;
                }
                PrintDeviceInfo(pDeviceInfo);
            }
        }
        else
        {
            printf("Find No Devices!\n");
            break;
        }


        printf("Please Input camera index:");
        unsigned int nIndex = 0;
        scanf_s("%d", &nIndex);

        if (nIndex >= stDeviceList.nDeviceNum)
        {
            printf("Input error!\n");
            break;
        }

        // ch:选择设备并创建句柄 | en:Select device and create handle
        nRet = MV_CC_CreateHandle(&handle, stDeviceList.pDeviceInfo[nIndex]);
        if (MV_OK != nRet)
        {
            printf("Create Handle fail! nRet [0x%x]\n", nRet);
            break;
        }

        
        // ch:打开设备 | en:Open device
        nRet = MV_CC_OpenDevice(handle);
        if (MV_OK != nRet)
        {
            printf("Open Device fail! nRet [0x%x]\n", nRet);
            break;
        }

        // 设置相机常见参数
        nRet = MV_CC_SetFrameRate(handle, 10.f);
        if (MV_OK != nRet)
        {
            printf("set FrameRate fail! nRet [0x%x]\n", nRet);
            break;
        }

        nRet = MV_CC_SetExposureTime(handle, 5000.f);
        if (MV_OK != nRet)
        {
            printf("set Exposure Time fail! nRet [0x%x]\n", nRet);
            break;
        }

        // 保存相机参数
        printf("Start export the camera properties to the file\n");
        printf("Wait......\n");

        nRet = MV_CC_FeatureSave(handle, "FeatureFile.ini");
        if (MV_OK != nRet)
        {
            printf("Save Feature fail! nRet [0x%x]\n", nRet);
            break;
        }
        printf("Finish export the camera properties to the file\n\n");

        // ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
        if (stDeviceList.pDeviceInfo[nIndex]->nTLayerType == MV_GIGE_DEVICE)
        {
            int nPacketSize = MV_CC_GetOptimalPacketSize(handle);
            if (nPacketSize > 0)
            {
                nRet = MV_CC_SetIntValue(handle, "GevSCPSPacketSize", nPacketSize);
                if (nRet != MV_OK)
                {
                    printf("Warning: Set Packet Size fail nRet [0x%x]!", nRet);
                }
            }
            else
            {
                printf("Warning: Get Packet Size fail nRet [0x%x]!", nPacketSize);
            }
        }

        // ch:设置触发模式为off | en:Set trigger mode as off
        nRet = MV_CC_SetEnumValue(handle, "TriggerMode", 0);

        if (MV_OK != nRet)
        {
            printf("Set Trigger Mode fail! nRet [0x%x]\n", nRet);
            break;
        }

        // ch:获取数据包大小 | en:Get payload size
        MVCC_INTVALUE stParam = { 0 };
        nRet = MV_CC_GetIntValue(handle, "PayloadSize", &stParam);
        if (MV_OK != nRet)
        {
            printf("Get PayloadSize fail! nRet [0x%x]\n", nRet);
            break;
        }
        g_nPayloadSize = stParam.nCurValue;

        unsigned int nThreadID = 0;
        void* hCreateWindow = (void*)_beginthreadex(NULL, 0, CreateRenderWindow, handle, 0, &nThreadID);
        if (NULL == hCreateWindow)
        {
            break;
        }

        // ch:开始取流 | en:Start grab image
        nRet = MV_CC_StartGrabbing(handle);
        if (MV_OK != nRet)
        {
            printf("Start Grabbing fail! nRet [0x%x]\n", nRet);
            break;
        }

        nThreadID = 0;
        void* hThreadHandle = (void*)_beginthreadex(NULL, 0, WorkThread, handle, 0, &nThreadID);
        if (NULL == hThreadHandle)
        {
            break;
        }

        printf("Press a key to stop grabbing.\n");
        WaitForKeyPress();

        g_bExit = true;
        WaitForSingleObject(hThreadHandle, INFINITE);
        CloseHandle(hThreadHandle);

        // ch:停止取流 | en:Stop grab image
        nRet = MV_CC_StopGrabbing(handle);
        if (MV_OK != nRet)
        {
            printf("Stop Grabbing fail! nRet [0x%x]\n", nRet);
            break;
        }

        // ch:关闭设备 | Close device
        nRet = MV_CC_CloseDevice(handle);
        if (MV_OK != nRet)
        {
            printf("ClosDevice fail! nRet [0x%x]\n", nRet);
            break;
        }

        // ch:销毁句柄 | Destroy handle
        nRet = MV_CC_DestroyHandle(handle);
        if (MV_OK != nRet)
        {
            printf("Destroy Handle fail! nRet [0x%x]\n", nRet);
            break;
        }
    } while (0);


    if (nRet != MV_OK)
    {
        if (handle != NULL)
        {
            MV_CC_DestroyHandle(handle);
            handle = NULL;
        }
    }

    printf("Press a key to exit.\n");
    WaitForKeyPress();

    return 0;
}

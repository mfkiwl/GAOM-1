#include "common.h"
#include "compute.h"
#include "TW_MemManager.h"
#include <fstream>
#include <string>
#include<opencv2\opencv.hpp>

using namespace std;

void MatToFloat(cv::Mat& img, float **fImg);
void testICGN(cv::Mat& imgR, cv::Mat& imgT);


int main()
{
	// The difference between these two images are simulated to be 0.05 pixels displacement
	// along the x axis
	cv::Mat Rmat = cv::imread("Test\\fu_0.bmp");
	cv::Mat Tmat = cv::imread("Test\\fu_20.bmp");

	auto wm_iWidth = Rmat.cols;
	auto wm_iHeight = Rmat.rows;

	cv::Mat Rmatnew(cv::Size(wm_iWidth, wm_iHeight), CV_8UC1);//创建图像矩阵
	cv::Mat Tmatnew(cv::Size(wm_iWidth, wm_iHeight), CV_8UC1);
	
	cv::cvtColor(Rmat, Rmatnew, CV_BGR2GRAY);//颜色空间转换，此处转换为灰色
	cv::cvtColor(Tmat, Tmatnew, CV_BGR2GRAY);

	testICGN(Rmatnew, Tmatnew);

	return 0;
}

void MatToFloat(cv::Mat& img, float **fImg)
{
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			fImg[i][j] = float(img.at<uchar>(i, j));
		}
	}
}

void testICGN(cv::Mat& imgR, cv::Mat& imgT)
{
	int m_iImgHeight = imgR.rows;
	int m_iImgWidth = imgR.cols;
	int m_iSubsetX = 16;
	int m_iSubsetY = 16;
	int m_iMarginX = 5;
	int m_iMarginY = 5;
	int m_iGridSpaceX = 5;
	int m_iGridSpaceY = 5;
	int m_iMaxIteration = 20;
	double m_dNormDeltaP = 0.001;

	float **m_dImg1;//作为指针的指针
	float **m_dImg2;
	cucreateptr(m_dImg1, m_iImgHeight, m_iImgWidth);//m_dImg1指向地址的指针分别指向图像数组的各行首个像素值
	cucreateptr(m_dImg2, m_iImgHeight, m_iImgWidth);//cucreateptr实现指针对图像各行各列进行定位，方便进行访问

	// Load opencv images to m_dImg1, m_dImg2;
	//将图像数据转化为float型储存到 m_dImg1, m_dImg2
	MatToFloat(imgR, m_dImg1);
	MatToFloat(imgT, m_dImg2);
	printf("%lf", **m_dImg1);
	int ***m_dPXY;
	float ***m_dP;
	int ** m_iIterationNum;
	int m_iIteration;
	ICGN_d_Handle m_Handle;//结构体数据
	Timer m_Time;

	//Define the size of region of interest (ROI)
	int m_iWidth = m_iImgWidth - 2; // set margin = 1 column
	int m_iHeight = m_iImgHeight - 2; // set margin = 1 row

									  //Define the size of subset window for IC-GN algorithm
	int m_iSubsetW = m_iSubsetX * 2 + 1;
	int m_iSubsetH = m_iSubsetY * 2 + 1;
	//Define the size of subset window for FFT-CC algorithm
	int m_iFFTSubW = m_iSubsetX * 2;
	int m_iFFTSubH = m_iSubsetY * 2;

	//Estimate the number of points of interest(POIs)
	int m_iNumberX = int(floor((m_iWidth - m_iSubsetX * 2 - m_iMarginX * 2) / double(m_iGridSpaceX))) + 1;
	int m_iNumberY = int(floor((m_iHeight - m_iSubsetY * 2 - m_iMarginY * 2) / double(m_iGridSpaceY))) + 1;

	icgn_gpu_prepare(m_Handle,
		m_iImgWidth, m_iImgHeight,
		m_iSubsetX, m_iSubsetY,
		m_iMarginX, m_iMarginY,
		m_iGridSpaceX, m_iGridSpaceY,
		m_Time);//在GPU申请内存
	icgn_prepare(m_Handle, m_dPXY, m_dP, m_iIterationNum);

	icgn_algorithm(m_dImg1, m_dImg2, m_Handle, m_iMaxIteration, m_dNormDeltaP, m_dPXY, m_dP,
		m_iIterationNum, m_iIteration, m_Time,m_iFFTSubW,m_iFFTSubH);



	
	// Output results
	ofstream oFile;
	oFile.open("Test\\ICGN_data.csv", ios::out | ios::trunc);//　ios::trunc：　　如果文件存在，把文件长度设为0  ios::out：　　　文件以输出方式打开(内存数据输出到文件)
	oFile << "X" << "," << "Y" << "," << "U" << "," << "Ux" << "," << "Uy"
		<< "," << "V" << "," << "Vx" << "," << "Vy" << ","
		<< "Interation" << "," << endl;
	for (int i = 0; i < m_iNumberY; i++)
	{
		for (int j = 0; j < m_iNumberX; j++)
		{
			m_iIteration += m_iIterationNum[i][j];
			oFile << int(m_dPXY[i][j][1]) << ","
				<< int(m_dPXY[i][j][0]) << "," << m_dP[i][j][0] << "," << m_dP[i][j][1] << "," << m_dP[i][j][2] << ","
				<< m_dP[i][j][3] << "," << m_dP[i][j][4] << "," << m_dP[i][j][5] << ","
				<< m_iIterationNum[i][j] << endl;
		}
	}
	oFile.close();

	// Deallocate memory
	icgn_gpu_finalize(m_Handle, m_Time);//释放内存
	icgn_finalize(m_dPXY, m_dP, m_iIterationNum);

	oFile.open("Test\\ICGN_infor.txt", ios::out | ios::trunc);

	oFile << "Interval (X-axis): " << m_iGridSpaceX << " [pixel]" << endl;
	oFile << "Interval (Y-axis): " << m_iGridSpaceY << " [pixel]" << endl;
	oFile << "Number of POI: " << m_iNumberY*m_iNumberX << " = " << m_iNumberX << " X " << m_iNumberY << endl;
	oFile << "Subset dimension: " << m_iSubsetW << "x" << m_iSubsetH << " pixels" << endl;
	oFile << "Time for device Mem Allocation: " << m_Time.m_dDevMemAlloc << " [millisec]" << endl;
	oFile << "Time for copy: " << m_Time.m_dMemCpy << " [millisec]" << endl;
	oFile << "Time for device Mem Free: " << m_Time.m_dDevMemFree << " [millisec]" << endl;
	oFile << "Time comsumed: " << m_Time.m_dConsumedTime << " [millisec]" << endl;
	oFile << "Time for Pre-computation: " << m_Time.m_dPrecomputeTime << " [millisec]" << endl;
	oFile << "Time for FFT_CC " << m_Time.m_dFFTCCTime << " [millisec]" << endl;
	oFile << "Time for all sub-pixel registration: " << m_Time.m_dICGNTime << " [millisec]" << endl;
	oFile << "Time for sub-pixel registration: " << m_Time.m_dICGNTime / (m_iNumberY*m_iNumberX) << " [millisec]" << " for average iteration steps of " << double(m_iIteration) / (m_iNumberY*m_iNumberX) << endl;

	oFile.close();


	cout << "Interval (X-axis): " << m_iGridSpaceX << " [pixel]" << endl;
	cout << "Interval (Y-axis): " << m_iGridSpaceY << " [pixel]" << endl;
	cout << "Number of POI: " << m_iNumberY*m_iNumberX << " = " << m_iNumberX << " X " << m_iNumberY << endl;
	cout << "Subset dimension: " << m_iSubsetW << "x" << m_iSubsetH << " pixels" << endl;
	cout << "Time for device Mem Allocation: " << m_Time.m_dDevMemAlloc << " [millisec]" << endl;
	cout << "Time for copy: " << m_Time.m_dMemCpy << " [millisec]" << endl;
	cout << "Time for device Mem Free: " << m_Time.m_dDevMemFree << " [millisec]" << endl;
	cout << "Time comsumed: " << m_Time.m_dConsumedTime << " [millisec]" << endl;
	cout << "Time for Pre-computation: " << m_Time.m_dPrecomputeTime << " [millisec]" << endl;
	cout << "Time for FFT_CC " << m_Time.m_dFFTCCTime << " [millisec]" << endl;
	cout << "Time for all sub-pixel registration: " << m_Time.m_dICGNTime << " [millisec]" << endl;
	cout << "Time for sub-pixel registration: " << m_Time.m_dICGNTime / (m_iNumberY*m_iNumberX) << " [millisec]" << " for average iteration steps of " << double(m_iIteration) / (m_iNumberY*m_iNumberX) << endl;
	cout << "Results are savec to ICGN_DATA.csv" << endl;

	//Destroy the created matrices
	cudestroyptr(m_dImg1);
	cudestroyptr(m_dImg2);
}

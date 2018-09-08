
#include"Header.h"

#include <cufft.h>
#include "fftw3.h"
#include <cufftw.h>
#include<device_functions.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include "icgn.cuh"
#include "reducer.cuh"
#include "common.h"
#include "compute.h"
#include "TW_MemManager.h"
#include <fstream>
#include <string>
#include<stdio.h>
#include<iostream>
//#include "engine.h"

using namespace std;
using namespace cv;
/*
const int BUFFER_SIZE = 1024;
char buffer[BUFFER_SIZE];
int size_of_subset = 32;//子区为32*32
*/
struct abc{
	int x;
	int y;
};

//int search_radius = 10;//搜索半径为10个像素
/*
void test()
{
	Engine* ep;
	mxArray *x1 = NULL;
	mxArray *y1 = NULL;
	if ((ep = engOpen("")) == NULL)
	{
		printf("Engine Fail");
	}
	engOutputBuffer(ep, buffer, BUFFER_SIZE);
	printf("Init Success");

	float x[5] = { 1.0, 2.5, 3.7, 4.4, 5.1 };
	float y[5] = { 3.3, 4.7, 9.6, 15.6, 21.3 };
	x1 = mxCreatefloatMatrix(1, 5, mxREAL);
	y1 = mxCreatefloatMatrix(1, 5, mxREAL);

	memcpy((void *)mxGetPr(x1), (void *)x, sizeof(x));
	memcpy((void *)mxGetPr(y1), (void *)y, sizeof(y));

	engPutVariable(ep, "x", x1);
	engPutVariable(ep, "y", y1);

	engEvalString(ep, "plot(x,y)");
	getchar();
	engClose(ep);
}
*/
/*
void huitu(float *displacement_x, float *displacement_y)
{
	Engine* ep;
	mxArray *x1 = NULL;
	mxArray *y1 = NULL;
	// mxArray *x = NULL;
	// mxArray *y = NULL;
	// float *a = (float*)malloc(256 * sizeof(float *));
	// float *b = (float*)malloc(256 * sizeof(float *));

	if ((ep = engOpen("")) == NULL)
	{
		printf("Engine Fail");
	}
	engOutputBuffer(ep, buffer, BUFFER_SIZE);
	printf("Init Success");

	for (int i = 0; i < 256; i++)
	{
	a[i] = i;
	b[i] = i;
	}
	
	// float x[10] = { 0, 3, 6, 8, 9, 3, 5, 4, 1, 7 };
	// float y[10] = { 2, 5, 7, 9, 3, 6, 4, 8, 1, 4 };

	x1 = mxCreatefloatMatrix(15, 15, mxREAL);
	y1 = mxCreatefloatMatrix(15, 15, mxREAL);

	//x = mxCreatefloatMatrix(1, 256, mxREAL);
	// y = mxCreatefloatMatrix(1, 256, mxREAL);
	// memcpy((void *)mxGetPr(x1), (void *)x, sizeof(x));
	// memcpy((void *)mxGetPr(y1), (void *)y, sizeof(y));
	memcpy((void *)mxGetPr(x1), (void *)displacement_x, 225 * sizeof(float));
	memcpy((void *)mxGetPr(y1), (void *)displacement_y, 225 * sizeof(float));
	// memcpy((void *)mxGetPr(x), (void *)a, sizeof(a));
	// memcpy((void *)mxGetPr(y), (void *)b, sizeof(b));

	//engPutVariable(ep, "u", x1);
	// engPutVariable(ep, "v", y1);
	engPutVariable(ep, "u", x1);
	engPutVariable(ep, "v", y1);

	engEvalString(ep, "[x,y]=meshgrid(16:32:464,16:32:464);");
	//engEvalString(ep, "y=0:1:500;");

	engEvalString(ep, "quiver(x,y,v,u,1)");
	getchar();
	engClose(ep);
}
*/



__global__ void latched_position(float *Mats, float *displacement_x, float *displacement_y, int i)
{
	int x = threadIdx.x;
	int y = threadIdx.y;
	//float* mat_of_subset;
	//mat_of_subset = Mat[i + j*gridDim.x];
	__shared__ float max[32 * 32];
	__shared__ int a[32 * 32], b[32 * 32];
	max[y + blockDim.y*x] = Mats[y + blockDim.y*x];
	max[y + blockDim.y*x + blockDim.x*blockDim.y] = Mats[y + blockDim.y*x + blockDim.x*blockDim.y];
	int k = 2 * blockDim.x*blockDim.y / 2;
	a[y + blockDim.y*x] = x;
	a[y + blockDim.y*x + blockDim.x*blockDim.y] = x + blockDim.x;
	b[y + blockDim.y*x] = y;
	b[y + blockDim.y*x + blockDim.x*blockDim.y] = y;
	while (k != 0)
	{
		if (max[y + blockDim.y*x] < max[y + blockDim.y*x + k])
		{
			max[y + blockDim.y*x] = max[y + blockDim.y*x + k];
			a[y + blockDim.y*x] = a[y + blockDim.y*x + k];
			b[y + blockDim.y*x] = b[y + blockDim.y*x + k];
		}
		__syncthreads();
		k = k / 2;

	}
	if (y + blockDim.y*x == i)
	{
		if (max[0] < 0.03)
		{
			displacement_x[y + blockDim.y*x] = 0;
			displacement_y[y + blockDim.y*x] = 0;
		}
		else
		{

			displacement_x[y + blockDim.y*x] = a[0];
			displacement_y[y + blockDim.y*x] = b[0];
		}
	}
}




inline int iAlignUp(int a, int b)
{
	return (a % b != 0) ? (a - a % b + b) : a;
}

inline __device__ void mulAndScale(cufftfloatComplex &a, const cufftfloatComplex &b, const float &c)//一复数矩阵共轭点乘另一复数矩阵
{
	cufftfloatComplex t = { c *(a.x * b.x + a.y * b.y), c *(-a.y * b.x + a.x * b.y) };
	a = t;
}

__global__ void conjugate(cufloatComplex *comp)
{
	int tid = threadIdx.y + threadIdx.x*blockDim.y;
	comp[tid].y = -comp[tid].y;

}

void H_conjugate(fftw_complex *comp，int )
{

}


__global__ void multiplication(cufloatComplex *comp1, cufloatComplex *comp2)
{
	int tid = threadIdx.y + threadIdx.x*blockDim.y;
	float t;
	t = comp1[tid].x;
	comp1[tid].x = comp1[tid].x * comp2[tid].x - comp1[tid].y * comp2[tid].y;
	comp1[tid].y = t * comp2[tid].y + comp1[tid].y * comp2[tid].x;
}
__global__ void get_FFTAveR_kernel_all_iteration(float*dR, int *dPXY,
	int iSubsetH, int iSubsetW, int iHeight, int iWidth,
	float *whole_dSubSet, float *whole_dSubsetAve)
{

	//默认smsize和dim大小相等
	//dim取64
	__shared__ float sm[BLOCK_SIZE_128];
	int tid = threadIdx.x;
	int dim = blockDim.x;
	int bid = blockIdx.x;
	int size = iSubsetH*iSubsetW;
	float avg;// = 0;
	float mySum = 0;
	float tempt;
	float *dSubSet = whole_dSubSet + size*bid;//子区大小为size,单个block处理单个子区，子区数目对应POI
	float *dSubsetAve = whole_dSubsetAve + (size + 1)*bid;//为何加1???因为第一个数据为均方差，其余方为各点灰度值减去均值

	int t_dpxy0 = dPXY[bid * 2];//t_dpxy0,t_dpxy1分别对应POI的x,y坐标
	int t_dpxy1 = dPXY[bid * 2 + 1];
	int t_iidx = iWidth*(int)(t_dpxy0 - iSubsetY) + (int)t_dpxy1 - iSubsetX;//少个l*iWidth（POI子区左上角顶点在大图像中的索引）
	float* p_dR = dR + t_iidx;//dR为图像，此为POI子区左上角顶点在大图像中的指针索引

	for (int id = tid; id<size; id += dim)
	{
		int	l = id / iSubsetW;//子区y方向索引
		int m = id % iSubsetW;//子区x方向索引
		int t_sidx = m + l*iWidth;//子区各点索引
		tempt = *(p_dR + t_sidx);;// dR[int(dPXY[bid * 2] - iSubsetY + l)*iWidth + int(dPXY[bid * 2 + 1] - iSubsetX + m)];
		dSubSet[id] = tempt;
		mySum += tempt / size;
	}
	__syncthreads();
	sumReduceBlock<BLOCK_SIZE_128, float>(sm, mySum, tid);
	__syncthreads();
	avg = sm[0];
	mySum = 0;
	for (int id = tid; id<size; id += dim)
	{
		tempt = dSubSet[id] - avg;
		mySum += tempt*tempt;
		dSubsetAve[id + 1] = tempt;
	}
	__syncthreads();
	sumReduceBlock<BLOCK_SIZE_128, float>(sm, mySum, tid);
	__syncthreads();
	if (tid == 0)
	{
		dSubsetAve[0] = sqrt(sm[tid]);
	}
	for (int id = tid; id<size; id += dim)
	{
		dSubsetAve[id + 1] = dSubsetAve[id + 1]/ dSubsetAve[0];
	}

}


void FFTW_cc
(int m_iNumberX,int m_iNumberY,int m_iFFTSubW,int m_iFFTSubH,
	float **dR,float **m_dImg1,int PXY,
	int iHeight,int iWidth,
	float*subsetR, float*subsetT,float* subset_aveR,float* subset_aveT)
{
	float *H_subsetR, *H_subsetT;
	float *H_subset_aveR，H_subset_aveT;
	int iNumbersize = m_iNumberX* m_iNumberY;
	int subsetsize = m_iFFTSubW*m_iFFTSubW;
	get_FFTAveR_kernel_all_iteration << <iNumbersize,BLOCK_SIZE_128 >> > (dR, dPXY,
		m_iFFTSubW, m_iFFTSubW, iHeight, iWidth,
		subsetR, subset_aveR);

	get_FFTAveR_kernel_all_iteration << <iNumbersize, BLOCK_SIZE_128 >> > (dT, dPXY,
		m_iFFTSubW, m_iFFTSubW, iHeight, iWidth,
		subsetT, subset_aveT);
	checkCudaErrors(cudaMemcpy(H_subset_aveR, subset_aveR, iNumbersize *(subsetsize+1)* sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(H_subset_aveT, subset_aveT, iNumbersize *(subsetsize + 1) * sizeof(float), cudaMemcpyDeviceToHost));


    #pragma omp parallel for firstprivate(m_iFFTSubW, m_iFFTSubH, H_subset_aveR,H_subset_aveT,subsetsize)
	{
		for (int i = 0;i < iNumbersize;i++)
		{
			fftw_complex *FT_R, *FT_T;
			fftw_plan p1,p2 q;
			FT_R = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) *m_iFFTSubW* (m_iFFTSubH/2+1));
			FT_T = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) *m_iFFTSubW* (m_iFFTSubH / 2 + 1));
			p1 = fftw_plan_dft_2d(m_iFFTSubW, m_iFFTSubH, H_subset_aveR+1+i*(subsetsize + 1), FT_R, FFTW_FORWARD, FFTW_MEASURE);
			p2 = fftw_plan_dft_2d(m_iFFTSubW, m_iFFTSubH, H_subset_aveT + 1 + i*(subsetsize + 1), FT_T, FFTW_FORWARD, FFTW_MEASURE);
			fftw_execute(p1);
			fftw_execute(p2);
			q = fftw_plan_dft_2d(m_iFFTSubW, m_iFFTSubH, out, in2, FFTW_BACKWARD, FFTW_MEASURE);

		}
	}
}


#include"Header.h"

#include <cufft.h>
#include "fftw3.h"

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

__global__ void conjugate(cuFloatComplex *comp)
{
	int tid = threadIdx.x + threadIdx.y*blockDim.x;
	int bid = blockIdx.x;
	int dim = blockDim.x*blockDim.y;
	comp[bid*dim + tid].y = -comp[bid*dim + tid].y;

}



__global__ void multiplication(cuFloatComplex *comp1, cuFloatComplex *comp2)
{
	int tid = threadIdx.x + threadIdx.y*blockDim.x;
	int bid = blockIdx.x;
	int dim = blockDim.x*blockDim.y;
	float t;
	t = comp1[bid*dim + tid].x;
	comp1[bid*dim + tid].x = comp1[bid*dim + tid].x * comp2[bid*dim + tid].x - comp1[bid*dim + tid].y * comp2[bid*dim + tid].y;
	comp1[bid*dim + tid].y = t * comp2[bid*dim + tid].y + comp1[bid*dim + tid].y * comp2[bid*dim + tid].x;
}

__global__ void latched_position(cufftReal *Mats, int *displacement_x, int *displacement_y, int subsetsize)
{
	int x = threadIdx.x;
	int y = threadIdx.y;
	int bid = blockIdx.x;
	int dim = blockDim.x*blockDim.y;
	__shared__ float max[64 * 64];
	__shared__ int a[64 * 64], b[64 * 64];
	if (x + dim*y == 0)
	{
		for (int j = 0;j < 64 * 64;j++)
		{
			max[j] = 0;
			a[j] = 0;
			b[j] = 0;
		}
	}
	max[x + blockDim.x*y] = Mats[bid*dim + x + blockDim.x*y] / subsetsize;
	int k = 64 * 64 / 2;
	a[x + blockDim.x*y] = x;
	b[x + blockDim.x*y] = y;
	while (k != 0)
	{
		if (max[x + blockDim.x*y] < max[x + blockDim.x*y + k])
		{
			max[x + blockDim.x*y] = max[x + blockDim.x*y + k];
			a[x + blockDim.x*y] = a[x + blockDim.x*y + k];
			b[x + blockDim.x*y] = b[x + blockDim.x*y + k];
		}
		__syncthreads();
		k = k / 2;

	}
	if (x + blockDim.x*y == 0)
	{
		if (max[0] < 0.03)
		{
			displacement_x[bid] = 0;
			displacement_y[bid] = 0;
		}
		else
		{

			displacement_x[bid] = a[0];
			displacement_y[bid] = b[0];
		}
	}
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
	float avg_sqrt_sum;
	float avg;// = 0;
	float mySum = 0;
	float tempt;
	float *dSubSet = whole_dSubSet + size*bid;//子区大小为size,单个block处理单个子区，子区数目对应POI
	float *dSubsetAve = whole_dSubsetAve + (size )*bid;//为何加1???因为第一个数据为均方差，其余方为各点灰度值减去均值

	int t_dpxy0 = dPXY[bid * 2];//t_dpxy0,t_dpxy1分别对应POI的x,y坐标
	int t_dpxy1 = dPXY[bid * 2 + 1];
	int t_iidx = iWidth*(int)(t_dpxy0 - iSubsetH/2+1) + (int)t_dpxy1 - iSubsetH/2+1;//少个l*iWidth（POI子区左上角顶点在大图像中的索引）
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
		dSubsetAve[id] = tempt;
	}
	__syncthreads();
	sumReduceBlock<BLOCK_SIZE_128, float>(sm, mySum, tid);
	__syncthreads();
	if (tid == 0)
	{
		avg_sqrt_sum = sqrt(sm[tid]);
	}
	for (int id = tid; id<size; id += dim)
	{
		dSubsetAve[id] = dSubsetAve[id]/ avg_sqrt_sum;
	}

}

__global__ void chang_coordinate(cufftReal *CUFFT, int iSubsetW,int subsetsize)
{
	int tid = threadIdx.x;
	int dim = blockDim.x;
	int bid = blockIdx.x;
	float *subset_CUFFT = CUFFT+subsetsize*bid;
	float temp;
	for (int id = tid; id<subsetsize/4; id += dim)
	{
		int	l = 2*id / iSubsetW;//子区y方向索引
		int m = 2* id % iSubsetW;//子区x方向索引
		int t_sidx = m + l*iSubsetW;//子区各点索引
		int t_sidx2 = m + iSubsetW / 2 + ( l + iSubsetW / 2) * iSubsetW;
		temp = *(subset_CUFFT + t_sidx);// dR[int(dPXY[bid * 2] - iSubsetY + l)*iWidth + int(dPXY[bid * 2 + 1] - iSubsetX + m)];
		*(subset_CUFFT + t_sidx) = *(subset_CUFFT + t_sidx2);
		*(subset_CUFFT + t_sidx2)=temp;
	}
	for (int id = tid; id<subsetsize / 4; id += dim)
	{
		int	l = 2 * id / iSubsetW;//子区y方向索引
		int m = 2 * id % iSubsetW;//子区x方向索引
		int t_sidx = m + iSubsetW / 2 + l*iSubsetW;//子区各点索引
		int t_sidx2 = m  + (l + iSubsetW / 2) * iSubsetW;
		temp = *(subset_CUFFT + t_sidx);// dR[int(dPXY[bid * 2] - iSubsetY + l)*iWidth + int(dPXY[bid * 2 + 1] - iSubsetX + m)];
		*(subset_CUFFT + t_sidx) = *(subset_CUFFT + t_sidx2);
		*(subset_CUFFT + t_sidx2) = temp;
	}


}

void FFTW_cc
(int m_iNumberX,int m_iNumberY,int m_iFFTSubW,int m_iFFTSubH,
	float *dR,float *dT,int *dPXY,
	int iHeight,int iWidth,	int *d_iU,int *d_iV)
{
	float *H_subsetR, *H_subsetT;
	float *H_subset_aveR,H_subset_aveT;
	int iNumbersize = m_iNumberX* m_iNumberY;
	int subsetsize = m_iFFTSubW*m_iFFTSubH;
	float*subsetR;
	float*subsetT;
	float* subset_aveR;
	float* subset_aveT;
	double time0 = getTickCount();
	checkCudaErrors(cudaMalloc((void **)&subsetR, iNumbersize *subsetsize * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&subsetT, iNumbersize *subsetsize * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&subset_aveR, iNumbersize *(subsetsize) * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&subset_aveT, iNumbersize *(subsetsize) * sizeof(float)));

	cufftComplex *CU_FT_R, *CU_FT_T;
	cufftReal * CuFFT;
	//cufftReal *test_in;
	//checkCudaErrors(cudaMalloc((void **)&test_in, m_iFFTSubW *m_iFFTSubH * sizeof(cufftReal)));
	//cuFloatComplex *test;
	//checkCudaErrors(cudaMalloc((void **)&test, m_iFFTSubW *(m_iFFTSubH / 2 + 1) * sizeof(cuFloatComplex)));
	checkCudaErrors(cudaMalloc((void **)&CU_FT_R, iNumbersize* m_iFFTSubW *(m_iFFTSubH / 2 + 1) * sizeof(cuFloatComplex)));
	checkCudaErrors(cudaMalloc((void **)&CU_FT_T, iNumbersize*  m_iFFTSubW *(m_iFFTSubH / 2 + 1) * sizeof(cuFloatComplex)));
	checkCudaErrors(cudaMalloc((void **)&CuFFT, iNumbersize*m_iFFTSubW *m_iFFTSubH * sizeof(cufftReal)));

	double time1 = getTickCount();
	get_FFTAveR_kernel_all_iteration << <iNumbersize,BLOCK_SIZE_128 >> > (dR, dPXY,
		m_iFFTSubW, m_iFFTSubW, iHeight, iWidth,
		subsetR, subset_aveR);

	get_FFTAveR_kernel_all_iteration << <iNumbersize, BLOCK_SIZE_128 >> > (dT, dPXY,
		m_iFFTSubW, m_iFFTSubW, iHeight, iWidth,
		subsetT, subset_aveT);

	double time2 = getTickCount();

	cufftHandle plan1, plan2;
	checkCudaErrors(cufftPlan2d(&plan1, m_iFFTSubW, m_iFFTSubH, CUFFT_R2C));
	checkCudaErrors(cufftPlan2d(&plan2, m_iFFTSubW, m_iFFTSubH, CUFFT_C2R));
	double time3 = getTickCount();
	//checkCudaErrors(cufftExecR2C(plan1, (cufftReal *)(test_in), (cuFloatComplex *)(test)));
	dim3 block(iNumbersize);
	dim3 threadnew(m_iFFTSubH, m_iFFTSubW / 2 + 1);
	dim3 thread(m_iFFTSubH, m_iFFTSubW);
	for (int i = 0;i < iNumbersize;i++)
	{
		//checkCudaErrors(cufftExecR2C(plan1, (cufftReal *)test_in, (cuFloatComplex *)(CU_FT_R + i* m_iFFTSubW *(m_iFFTSubH / 2 + 1))));
		checkCudaErrors(cufftExecR2C(plan1, (cufftReal *)(subset_aveR + i * (subsetsize)), (cufftComplex *)(CU_FT_R + i* m_iFFTSubW *(m_iFFTSubH / 2 + 1))));
		checkCudaErrors(cufftExecR2C(plan1, (cufftReal *)(subset_aveT + i * (subsetsize)), (cufftComplex *)(CU_FT_T + i* m_iFFTSubW *(m_iFFTSubH / 2 + 1))));
	}

	double time4 = getTickCount();

	conjugate << <block, threadnew >> > (CU_FT_R);
	multiplication << <block, threadnew >> > (CU_FT_R, CU_FT_T);

	double time5 = getTickCount();

	for (int j = 0;j < iNumbersize;j++)
	{
		checkCudaErrors(cufftExecC2R(plan2, (cuFloatComplex *)CU_FT_R, (cufftReal *)(CuFFT + subsetsize)));
	}

	double time6 = getTickCount();
	chang_coordinate << <block, thread >> > (CuFFT, m_iFFTSubW, subsetsize);

	latched_position << <block, thread >> > (CuFFT, d_iU, d_iV, subsetsize);

	double time7 = getTickCount();

	double stime1 = (time1 - time0) / getTickFrequency();
    double stime2 = (time2 - time1) / getTickFrequency();
	double stime3 = (time3 - time2) / getTickFrequency();
	double stime4 = (time4 - time3) / getTickFrequency();
	double stime5 = (time5 - time4) / getTickFrequency();
	double stime6 = (time6 - time5) / getTickFrequency();
	double stime7 = (time7 - time6) / getTickFrequency();

	cout << "Malloc" << stime1 << endl;
	cout << "Malloc" << stime2 << endl;
	cout << "Malloc" << stime3 << endl;
	cout << "Malloc" << stime4 << endl;
	cout << "Malloc" << stime5 << endl;
	cout << "Malloc" << stime6 << endl;
	cout << "Malloc" << stime6 << endl;

	checkCudaErrors(cudaFree(subsetR));
	checkCudaErrors(cudaFree(subsetT));
	checkCudaErrors(cudaFree(subset_aveR));
	checkCudaErrors(cudaFree(subset_aveT));
	checkCudaErrors(cudaFree(CU_FT_R));
	checkCudaErrors(cudaFree(CU_FT_T));
	checkCudaErrors(cudaFree(CuFFT));


	/*
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
	*/
}

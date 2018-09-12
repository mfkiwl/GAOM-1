#ifndef FFTW_CC_CUH
#define FFTW_CC_CUH

#include "common.h"
#include "compute.h"

void FFTW_cc
(int m_iNumberX, int m_iNumberY, int m_iFFTSubW, int m_iFFTSubH,
	float *dR, float *dT, int *PXY,
	int iHeight, int iWidth,
	int *d_iU, int *d_iV);
#endif // FFTW_CC_CUH


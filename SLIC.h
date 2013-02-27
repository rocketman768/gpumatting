// SLIC.h: interface for the SLIC class.
//===========================================================================
// This code implements the saliency method described in:
//
// Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Susstrunk,
// "SLIC Superpixels",
// EPFL Technical Report no. 149300, June 2010.
//===========================================================================
//	Copyright (c) 2010 Radhakrishna Achanta [EPFL]. All rights reserved.
//===========================================================================
// Email: firstname.lastname@epfl.ch
//////////////////////////////////////////////////////////////////////

#ifndef _SLIC_H_INCLUDED_
#define _SLIC_H_INCLUDED_

#include <vector>
#include <string>
#include <algorithm>

#ifndef SLIC_MAX
#define SLIC_MAX(a,b) ((a>b)?a:b)
#endif

#ifndef SLIC_MIN
#define SLIC_MIN(a,b) ((a<b)?a:b)
#endif

// Longest number of characters in a file name.
#define _MAX_FNAME 128

/*! \brief Just do the damn segmentation.
 * \author Philip G. Lee
 *
 * \param labels width*height*sizeof(int) row-major buffer for the pixel labels
 * \param in width*height*sizeof(unsigned int) row-major ARGB pixel buffer
 * \param width width of input image and output labels
 * \param height height of input image and output labels
 * \param nseg approximate number of output segments
 * \param spatialConsistency higher means segments are more spatially smooth and regular.
 * \returns the number of segments produced
 */
int slicSegmentation( int* labels, const unsigned int* in, int width, int height, int nseg, double spatialConsistency = 10.0 );

class SLIC  
{
public:
	SLIC();
	virtual ~SLIC();
        /*!
	 * Superpixel segmentation for a given step size (superpixel size ~= step*step)
	 * INPUT:
	 * \b ubuff - argb array in raster-scan order representing the image.
	 * \b width - image width
	 * \b height - image height
	 * \b STEP - step size. Each superpixel will be approximately STEPxSTEP.
	 * \b m - Parameter (>0) controlling spatial consistency?
	 * OUTPUT:
	 * \b klabels - appears to be an image of the labels
	 * \b numlabels - number of labels
	 */
	void DoSuperpixelSegmentation_ForGivenStepSize(
		const unsigned int*			ubuff,//Each 32 bit unsigned int contains ARGB pixel values.
		const int					width,
		const int					height,
		int**						klabels,
		int&						numlabels,
		const int&					STEP,
		const double&				m);
	//============================================================================
	// Superpixel segmentation for a given number of superpixels
	//============================================================================
	void DoSuperpixelSegmentation_ForGivenK(
		const unsigned int*			ubuff,
		const int					width,
		const int					height,
		int*						klabels,
		int&						numlabels,
		const int&					K,
		const double&				m);
	//============================================================================
	// Supervoxel segmentation for a given step size (supervoxel size ~= step*step*step)
	//============================================================================
	void DoSupervoxelSegmentation(
		const unsigned int**		ubuffvec,
		const int&					width,
		const int&					height,
		const int&					depth,
		int**						klabels,
		int&						numlabels, const int&					STEP,
		const double&				m);
	//============================================================================
	// Save superpixel labels in a text file in raster scan order
	//============================================================================
	void SaveSuperpixelLabels(
		const int*&					labels,
		const int&					width,
		const int&					height,
		const std::string&				filename,
		const std::string&				path);
	//============================================================================
	// Save superpixel labels in a text file in raster scan, depth order
	//============================================================================
	void SaveSupervoxelLabels(
		const int**&				labels,
		const int&					width,
		const int&					height,
		const int&					depth,
		const std::string&				filename,
		const std::string&				path);
	//============================================================================
	// Function to draw boundaries around superpixels of a given 'color'.
	// Can also be used to draw boundaries around supervoxels, i.e layer by layer.
	//============================================================================
	void DrawContoursAroundSegments(
		unsigned int*				segmentedImage,
		const int*					labels,
		const int&					width,
		const int&					height,
		const unsigned int&			color );

private:
	//============================================================================
	// The main SLIC algorithm for generating superpixels
	//============================================================================
	void PerformSuperpixelSLIC(
		std::vector<double>&				kseedsl,
		std::vector<double>&				kseedsa,
		std::vector<double>&				kseedsb,
		std::vector<double>&				kseedsx,
		std::vector<double>&				kseedsy,
		int*						klabels,
		const int&					STEP,
		const std::vector<double>&		edgemag,
		const double&				m);
	//============================================================================
	// The main SLIC algorithm for generating supervoxels
	//============================================================================
	void PerformSupervoxelSLIC(
		std::vector<double>&				kseedsl,
		std::vector<double>&				kseedsa,
		std::vector<double>&				kseedsb,
		std::vector<double>&				kseedsx,
		std::vector<double>&				kseedsy,
		std::vector<double>&				kseedsz,
		int**						klabels,
		const int&					STEP,
		const double&				m);
	//============================================================================
	// Pick seeds for superpixels when step size of superpixels is given.
	//============================================================================
	void GetLABXYSeeds_ForGivenStepSize(
		std::vector<double>&				kseedsl,
		std::vector<double>&				kseedsa,
		std::vector<double>&				kseedsb,
		std::vector<double>&				kseedsx,
		std::vector<double>&				kseedsy,
		const int&					STEP,
		const bool&					perturbseeds,
		const std::vector<double>&		edgemag);
	//============================================================================
	// Pick seeds for superpixels when number of superpixels is input.
	//============================================================================
	void GetLABXYSeeds_ForGivenK(
		std::vector<double>&				kseedsl,
		std::vector<double>&				kseedsa,
		std::vector<double>&				kseedsb,
		std::vector<double>&				kseedsx,
		std::vector<double>&				kseedsy,
		const int&					STEP,
		const bool&					perturbseeds,
		const std::vector<double>&		edges);
	//============================================================================
	// Pick seeds for supervoxels
	//============================================================================
	void GetKValues_LABXYZ(
		std::vector<double>&				kseedsl,
		std::vector<double>&				kseedsa,
		std::vector<double>&				kseedsb,
		std::vector<double>&				kseedsx,
		std::vector<double>&				kseedsy,
		std::vector<double>&				kseedsz,
		const int&					STEP);
	//============================================================================
	// Move the seeds to low gradient positions to avoid putting seeds at region boundaries.
	//============================================================================
	void PerturbSeeds(
		std::vector<double>&				kseedsl,
		std::vector<double>&				kseedsa,
		std::vector<double>&				kseedsb,
		std::vector<double>&				kseedsx,
		std::vector<double>&				kseedsy,
		const std::vector<double>&		edges);
	//============================================================================
	// Detect color edges, to help PerturbSeeds()
	//============================================================================
	void DetectLabEdges(
		const double*				lvec,
		const double*				avec,
		const double*				bvec,
		const int&					width,
		const int&					height,
		std::vector<double>&				edges);
	//============================================================================
	// xRGB to XYZ conversion; helper for RGB2LAB()
	//============================================================================
	void RGB2XYZ(
		const int&					sR,
		const int&					sG,
		const int&					sB,
		double&						X,
		double&						Y,
		double&						Z);
	//============================================================================
	// sRGB to CIELAB conversion
	//============================================================================
	void RGB2LAB(
		const int&					sR,
		const int&					sG,
		const int&					sB,
		double&						lval,
		double&						aval,
		double&						bval);
	//============================================================================
	// sRGB to CIELAB conversion for 2-D images
	//============================================================================
	void DoRGBtoLABConversion(
		const unsigned int*&		ubuff,
		double*&					lvec,
		double*&					avec,
		double*&					bvec);
	//============================================================================
	// sRGB to CIELAB conversion for 3-D volumes
	//============================================================================
	void DoRGBtoLABConversion(
		const unsigned int**&		ubuff,
		double**&					lvec,
		double**&					avec,
		double**&					bvec);

	//============================================================================
	// Post-processing of SLIC segmentation, to avoid stray labels.
	//============================================================================
	void EnforceLabelConnectivity(
		const int*					labels,
		const int&					width,
		const int&					height,
		int*						nlabels,//input labels that need to be corrected to remove stray labels
		int&						numlabels,//the number of labels changes in the end if segments are removed
		const int&					K); //the number of superpixels desired by the user

	//============================================================================
	// Find next superpixel label; helper for EnforceLabelConnectivity()
	//============================================================================
	void FindNext(
		const int*					labels,
		int*						nlabels,
		const int&					height,
		const int&					width,
		const int&					h,
		const int&					w,
		const int&					lab,
		int*						xvec,
		int*						yvec,
		int&						count);

	//============================================================================
	// Post-processing of SLIC supervoxel segmentation, to avoid stray labels.
	//============================================================================
	void EnforceLabelConnectivity_supervoxels(
		const int&					width,
		const int&					height,
		const int&					depth,
		int**						nlabels,//input labels that to be corrected. output is stored in here too.
		int&						numlabels,//the number of labels changes in the end if segments are removed
		const int&					STEP); //step size, it helps decide the minimum acceptable size

	//============================================================================
	// Find next supervoxel label; helper for EnforceLabelConnectivity()
	//============================================================================
	void FindNext_supervoxels(
		int**						labels,
		int**						nlabels,
		const int&					depth,
		const int&					height,
		const int&					width,
		const int&					d,
		const int&					h,
		const int&					w,
		const int&					lab,
		int*						xvec,
		int*						yvec,
		int*						zvec,
		int&						count);

   /*!
    * \author Philip G. Lee
    * Given path = "/path/to/file.ext", returns
    * file = "/path/to/file"
    * extension = ".ext"
    */
   void splitpath( const char* path, char* file, char* extension );

private:
	int										m_width;
	int										m_height;
	int										m_depth;

	double*									m_lvec;
	double*									m_avec;
	double*									m_bvec;

	double**								m_lvecvec;
	double**								m_avecvec;
	double**								m_bvecvec;
};

#endif // _SLIC_H_INCLUDED_

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "highgui.h"
#include "opencv2/opencv.hpp"
#include <string>
#include <fstream>

#include "includes/helperfunctions.h"

// DLoopDetector and DBoW2
#include <DBoW2/DBoW2.h> // defines Surf64Vocabulary
#include "DLoopDetector.h" // defines Surf64LoopDetector
#include <DUtilsCV/DUtilsCV.h> // defines macros CVXX

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
// Demo
#include "includes/demoDetector.h"

using namespace DLoopDetector;
using namespace DBoW2;
using namespace std;
using namespace cv;


#define PI 3.14159265
Matrix aboutX = Matrix::rotMatX(-PI,4);
Matrix new_tr;
Matrix new_tr_global;

double valx;
double valz;

string IMG_DIR1 = "/home/iiith/Downloads/newrun/loop2";
string IMG_DIR2 = "/home/iiith/Downloads/newrun/loop3";

static const string VOC_FILE = "./resources/iiit_voc.voc.gz"; //
//static const string IMAGE_DIR =  + "left";
static const int IMAGE_W = 1280; // image size
static const int IMAGE_H = 720;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
//void my_dloop(std::vector<int> &index1, std::vector<int> &index2,string file, string img_dir, int a, int b);
void my_dloop(std::vector<loops> &loop, string VOC_FILE1,string IMAGE_DIR1,string IMAGE_DIR2, int IMAGE_W1, int IMAGE_H1);
/// This functor extracts SURF64 descriptors in the required format
class SurfExtractor: public FeatureExtractor<FSurf64::TDescriptor>
{
public:
  /** 
   * Extracts features from an image
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
  virtual void operator()(const cv::Mat &im, 
    vector<cv::KeyPoint> &keys, vector<vector<float> > &descriptors) const;
};

// 
struct Quat{
    double w,x,y,z;
};

Quat CalculateRotation(Matrix a) {
  Quat q;
  float trace = a.getData(0,0) + a.getData(1,1) + a.getData(2,2); // I removed + 1.0f; see discussion with Ethan
  if( trace > 0 ) {// I changed M_EPSILON to 0
    float s = 0.5f / sqrtf(trace+ 1.0f);
    q.w = 0.25f / s;
    q.x = ( a.getData(2,1) - a.getData(1,2) ) * s;
    q.y = ( a.getData(0,2) - a.getData(2,0) ) * s;
    q.z = ( a.getData(1,0) - a.getData(0,1) ) * s;
  } else {
    if ( a.getData(0,0) > a.getData(1,1) && a.getData(0,0) > a.getData(2,2) ) {
      float s = 2.0f * sqrtf( 1.0f + a.getData(0,0) - a.getData(1,1) - a.getData(2,2));
      q.w = (a.getData(2,1) - a.getData(1,2) ) / s;
      q.x = 0.25f * s;
      q.y = (a.getData(0,1) + a.getData(1,0) ) / s;
      q.z = (a.getData(0,2) + a.getData(2,0) ) / s;
    } else if (a.getData(1,1) > a.getData(2,2)) {
      float s = 2.0f * sqrtf( 1.0f +  a.getData(1,1) -  a.getData(0,0) -  a.getData(2,2));
      q.w = (a.getData(0,2) - a.getData(2,0) ) / s;
      q.x = (a.getData(0,1) + a.getData(1,0) ) / s;
      q.y = 0.25f * s;
      q.z = (a.getData(1,2) + a.getData(2,1) ) / s;
    } else {
      float s = 2.0f * sqrtf( 1.0f +  a.getData(2,2) -  a.getData(0,0) -  a.getData(1,1) );
      q.w = (a.getData(1,0) - a.getData(0,1) ) / s;
      q.x = (a.getData(0,2) + a.getData(2,0) ) / s;
      q.y = (a.getData(1,2) + a.getData(2,1) ) / s;
      q.z = 0.25f * s;
    }
  }
  return q;
}




// For single loop

int main()
{

  ofstream traj_local, traj_global, for_plot, for_g2o_traj, for_g2o_traj_with_loops,for_g2o_traj_with_loops_quat;
  traj_local.open("data/loop1/Tr_local_1.txt");
  traj_global.open("data/loop1/Tr_global_1.txt");
  for_plot.open("data/loop1/2d_coords_1.txt");
  for_g2o_traj.open("data/loop1/traj1.g2o");
  for_g2o_traj_with_loops.open("data/merging/traj1_traj2_with_inliers.g2o");
  for_g2o_traj_with_loops_quat.open("data/merging/traj1_traj2_quat.g2o");

  int numImages1 =0, numImages2 =0;
  std::string path1 = IMG_DIR1 + "/left";
	char * dst1 = new char[path1.length() + 1];
	std::strcpy(dst1,path1.c_str());

  std::string path2 = IMG_DIR2 + "/left";
  char * dst2 = new char[path2.length() + 1];
  std::strcpy(dst2,path2.c_str());

	numImages1 = listdir(dst1);
	numImages2 = listdir(dst2);
	
	std::vector<Matrix> Tr_local;
	std::vector<Matrix> Tr_global;

	my_libviso2(Tr_local,Tr_global,IMG_DIR1,numImages1);
        for_g2o_traj_with_loops << "VERTEX3 1 0 0 0 0 0 0 1000 0 0 0 0 0 1000 0 0 0 0 1000 0 0 0 1000 0 0 1000 0 1000" << endl;
        for_g2o_traj_with_loops_quat << "VERTEX_SE3:QUAT 1 0 0 0 0 0 0 1 1000 0 0 0 0 0 1000 0 0 0 0 1000 0 0 0 1000 0 0 1000 0 1000" << endl;

	for (int i=0; i<Tr_global.size(); i++){
            	new_tr_global=Tr_global[i];
                double dx = new_tr_global.getData(0,3);
                double dy = new_tr_global.getData(1,3);
                double dz = new_tr_global.getData(2,3);
                Matrix rot = new_tr_global.getMat(0,0,2,2);
                Quat q = CalculateRotation(rot); 
                double thetay = atan2(new_tr_global.getData(0,2), sqrt(new_tr_global.getData(0,0)*new_tr_global.getData(0,0) + new_tr_global.getData(0,1)*new_tr_global.getData(0,1)));

                double thetax = atan2(-new_tr_global.getData(1,2), new_tr_global.getData(2,2));

                double thetaz = atan2(-new_tr_global.getData(0,1), new_tr_global.getData(0,0));
                for_g2o_traj_with_loops << "VERTEX3 " << i+2 <<' '<< dx <<' '<< dy << ' ' << dz <<' '<< thetax<<' '<<thetay <<' '<<thetaz<<' '<< "1000 0 0 0 0 0 1000 0 0 0 0 1000 0 0 0 1000 0 0 1000 0 1000" << endl;
                for_g2o_traj_with_loops_quat << "VERTEX_SE3:QUAT " << i+2 <<' '<< dx <<' '<< dy << ' ' << dz <<' '<<q.x<<' '<<q.y <<' '<<q.z<<' '<< q.w<<' '<< "1000 0 0 0 0 0 1000 0 0 0 0 1000 0 0 0 1000 0 0 1000 0 1000" << endl;

        }


	for (int i=0; i<Tr_global.size(); i++)
  {
  	traj_local << "frame: " << i+1 << endl << Tr_local[i] << endl << endl;
  	traj_global << "frame: " << i+1 << endl << Tr_global[i] << endl << endl; 
  	
  	valx = Tr_global[i].getData(0,3);
  	valz = Tr_global[i].getData(2,3); 
  	
  	for_plot << valx << '\t' << valz << endl;

  	new_tr = Tr_local[i];//aboutX*Tr_local[i];
  double dx = new_tr.getData(0,3);
  double dy = new_tr.getData(1,3);
  double dz = new_tr.getData(2,3);
  double thetay = atan2(new_tr.getData(0,2), sqrt(new_tr.getData(0,0)*new_tr.getData(0,0) + new_tr.getData(0,1)*new_tr.getData(0,1)));

  double thetax = atan2(-new_tr.getData(1,2), new_tr.getData(2,2));

  double thetaz = atan2(-new_tr.getData(0,1), new_tr.getData(0,0));
  Matrix rot = new_tr.getMat(0,0,2,2);
                  Quat q = CalculateRotation(rot); 


  	//double thetay = atan2(-new_tr.getData(2,0), sqrt(new_tr.getData(2,1)*new_tr.getData(2,1) + new_tr.getData(2,2)*new_tr.getData(2,2)));


  		for_g2o_traj << "EDGE_SE2 " << i+1 <<' '<< i+2 <<' '<< -dz <<' '<< -dx <<' '<< thetay <<' '<< "1000 0 0 1000 0 10000" << endl;
  		//for_g2o_traj_with_loops	<< "EDGE_SE2 " << i+1 <<' '<< i+2 <<' '<< -dz <<' '<< -dx <<' '<< thetay <<' '<< "1000 0 0 1000 0 10000" << endl;
	for_g2o_traj_with_loops << "EDGE3 " << i+1 <<' '<< i+2 <<' '<< dx <<' '<< dy << ' ' << dz <<' '<< thetax<<' '<<thetay <<' '<<thetaz<<' '<< "1000 0 0 0 0 0 1000 0 0 0 0 1000 0 0 0 1000 0 0 1000 0 1000" << endl;
                for_g2o_traj_with_loops_quat << "EDGE_SE3:QUAT " << i+1 <<' ' << i+2<<' '<< dx <<' '<< dy << ' ' << dz <<' '<<q.x<<' '<<q.y <<' '<<q.z<<' '<< q.w<<' '<< "1000 0 0 0 0 0 1000 0 0 0 0 1000 0 0 0 1000 0 0 1000 0 1000" << endl;

  	
  }

  traj_local.close();
  traj_global.close();
  for_plot.close();
	for_g2o_traj.close();

	for_g2o_traj_with_loops << endl;
	for_g2o_traj_with_loops_quat << endl;

  Tr_local.clear();
  Tr_global.clear();

  if(Tr_local.empty() && Tr_global.empty())
  my_libviso2(Tr_local,Tr_global,IMG_DIR2,numImages2);

  traj_local.open("data/loop2/Tr_local_2.txt");
  traj_global.open("data/loop2/Tr_global_2.txt");
  for_plot.open("data/loop2/2d_coords_2.txt");
  for_g2o_traj.open("data/loop2/traj2.g2o");
  for_g2o_traj_with_loops << "VERTEX3 " << 2+numImages1 <<" 0 0 0 0 0 0 1000 0 0 0 0 0 1000 0 0 0 0 1000 0 0 0 1000 0 0 1000 0 1000" << endl;
  for_g2o_traj_with_loops_quat << "VERTEX_SE3:QUAT " << 2+numImages1 <<" 0 0 0 0 0 0 1 1000 0 0 0 0 0 1000 0 0 0 0 1000 0 0 0 1000 0 0 1000 0 1000" << endl;

    for (int i=0; i<Tr_global.size(); i++){
        new_tr_global = Tr_global[i];//aboutX*Tr_local[i];

        double dx = new_tr_global.getData(0,3);
        double dy = new_tr_global.getData(1,3);
        double dz = new_tr_global.getData(2,3);
        Matrix rot = new_tr_global.getMat(0,0,2,2);
                        Quat q = CalculateRotation(rot); 

        double thetay = atan2(new_tr_global.getData(0,2), sqrt(new_tr_global.getData(0,0)*new_tr_global.getData(0,0) + new_tr_global.getData(0,1)*new_tr_global.getData(0,1)));

        double thetax = atan2(-new_tr_global.getData(1,2), new_tr_global.getData(2,2));

        double thetaz = atan2(-new_tr_global.getData(0,1), new_tr_global.getData(0,0));
        for_g2o_traj_with_loops << "VERTEX3 " << i+3+numImages1 <<' '<< dx <<' '<< dy << ' ' << dz <<' '<< thetax<<' '<<thetay <<' '<<thetaz<<' '<< "1000 0 0 0 0 0 1000 0 0 0 0 1000 0 0 0 1000 0 0 1000 0 1000" << endl;
        for_g2o_traj_with_loops_quat << "VERTEX_SE3:QUAT " << i+3+numImages1 <<' '<< dx <<' '<< dy << ' ' << dz <<' '<<q.x<<' '<<q.y <<' '<<q.z<<' '<< q.w<<' '<< "1000 0 0 0 0 0 1000 0 0 0 0 1000 0 0 0 1000 0 0 1000 0 1000" << endl;

    }

  for (int i=0; i<Tr_global.size(); i++)
  {
    traj_local << "frame: " << i+1 << endl << Tr_local[i] << endl << endl;
    traj_global << "frame: " << i+1 << endl << Tr_global[i] << endl << endl; 
    
    valx = Tr_global[i].getData(0,3);
    valz = Tr_global[i].getData(2,3); 
    
    for_plot << valx << '\t' << valz << endl;

    new_tr = Tr_local[i];//aboutX*Tr_local[i];

  double dx = new_tr.getData(0,3);
  double dy = new_tr.getData(1,3);
  double dz = new_tr.getData(2,3);
  Matrix rot = new_tr.getMat(0,0,2,2);
                  Quat q = CalculateRotation(rot); 

  double thetay = atan2(new_tr.getData(0,2), sqrt(new_tr.getData(0,0)*new_tr.getData(0,0) + new_tr.getData(0,1)*new_tr.getData(0,1)));

  double thetax = atan2(-new_tr.getData(1,2), new_tr.getData(2,2));

  double thetaz = atan2(-new_tr.getData(0,1), new_tr.getData(0,0));
    //double thetay = atan2(-new_tr.getData(2,0), sqrt(new_tr.getData(2,1)*new_tr.getData(2,1) + new_tr.getData(2,2)*new_tr.getData(2,2)));

      for_g2o_traj << "EDGE_SE2 " << i+1 <<' '<< i+2 <<' '<< -dz <<' '<< -dx <<' '<< thetay <<' '<< "1000 0 0 1000 0 10000" << endl;
      //for_g2o_traj_with_loops << "EDGE_SE2 " << i+1+numImages1 <<' '<< i+2+numImages1 <<' '<< -dz <<' '<< -dx <<' '<< thetay <<' '<< "1000 0 0 1000 0 10000" << endl;
	for_g2o_traj_with_loops << "EDGE3 " << i+2+numImages1 <<' '<< i+3+numImages1 <<' '<< dx <<' '<< dy << ' ' << dz <<' '<< thetax<<' '<<thetay <<' '<<thetaz<<' '<< "1000 0 0 0 0 0 1000 0 0 0 0 1000 0 0 0 1000 0 0 1000 0 1000" << endl;
        for_g2o_traj_with_loops_quat << "EDGE_SE3:QUAT " << i+2+numImages1 <<' '<< i+3+numImages1<<' '<< dx <<' '<< dy << ' ' << dz <<' '<<q.x<<' '<<q.y <<' '<<q.z<<' '<< q.w<<' '<< "1000 0 0 0 0 0 1000 0 0 0 0 1000 0 0 0 1000 0 0 1000 0 1000" << endl;


 
  }

  traj_local.close();
  traj_global.close();
  for_plot.close();
  for_g2o_traj.close();
  
  for_g2o_traj_with_loops << endl;
  for_g2o_traj_with_loops_quat << endl;

  // Detecting Loops
  loops l;
  std::vector<loops> loop;

  my_dloop(loop,VOC_FILE,(IMG_DIR1+"/left"),(IMG_DIR2+"/left"),IMAGE_W,IMAGE_H);


  // loops l1;
  // for (int j = 0; j < loop.size(); j++)
  // {
  //     l1 = loop[j];

  //     cout << "index1: " << l1.idx1 << " index2: " << l1.idx2 << " val1: " << l1.val1 << " val2: " << l1.val2 << endl;

  // }

  // waitKey(0);
  // cout << "loop_size: " << loop.size();

  // Finding merging pairs
  std::vector<Tr_relative> relative;

  // waitKey(0);
  my_libviso2_relative(relative,loop, IMG_DIR1,IMG_DIR2,numImages1);

  //cout << "size Tr: " << relative.size() << endl;

  // traj_local.open("my_complete_data_2/loop1.txt");
  // traj_global.open("my_complete_data_2/edges1.g2o");

  cout << "writing merging loops ..." << endl;
  for (int i = 0; i < relative.size(); i++)
  {
      Tr_relative t = relative[i];
      Matrix m = t.transform;

      	//traj_local << t.frame1 << '\t' << t.frame2 << endl;
    	
    	valx = m.getData(0,3);
    	valz = m.getData(2,3); 

    	new_tr = m;//aboutX*m;

	  double dx = new_tr.getData(0,3);
	  double dy = new_tr.getData(1,3);
	  double dz = new_tr.getData(2,3);
          Matrix rot = new_tr.getMat(0,0,2,2);
          Quat q = CalculateRotation(rot); 

  double thetay = atan2(new_tr.getData(0,2), sqrt(new_tr.getData(0,0)*new_tr.getData(0,0) + new_tr.getData(0,1)*new_tr.getData(0,1)));

  double thetax = atan2(-new_tr.getData(1,2), new_tr.getData(2,2));

  double thetaz = atan2(-new_tr.getData(0,1), new_tr.getData(0,0));
    	//double thetay = atan2(-new_tr.getData(2,0), sqrt(new_tr.getData(2,1)*new_tr.getData(2,1) + new_tr.getData(2,2)*new_tr.getData(2,2)));

    	//if(i<Tr_global.size()-1)
    	//traj_global << "EDGE_SE2 " << t.frame1 <<' '<< t.frame2 <<' '<< -dz <<' '<< -dx <<' '<< thetay <<' '<< "1000 0 0 1000 0 10000" << endl;
    	//for_g2o_traj_with_loops << "EDGE_SE2 " << t.frame1 <<' '<< t.frame2 <<' '<< -dz <<' '<< -dx <<' '<< thetay <<' '<< "1000 0 0 1000 0 10000" << endl;
	for_g2o_traj_with_loops << "EDGE3 " << t.frame1 <<' '<< t.frame2 <<' '<< dx <<' '<< dy << ' ' << dz <<' '<< thetax<<' '<<thetay <<' '<<thetaz<<' '<< "1000 0 0 0 0 0 1000 0 0 0 0 1000 0 0 0 1000 0 0 1000 0 1000" << endl;
        for_g2o_traj_with_loops_quat << "EDGE_SE3:QUAT " << t.frame1 <<' '<< t.frame2<<' '<< dx <<' '<< dy << ' ' << dz <<' '<<q.x<<' '<<q.y <<' '<<q.z<<' '<< q.w<<' '<< "1000 0 0 0 0 0 1000 0 0 0 0 1000 0 0 0 1000 0 0 1000 0 1000" << endl;

      //cout << t.transform << endl;
  }

  // // cout << "size of relative: " << relative.size() << endl;
  // // cout << "size of index1: " << index1.size() << endl;
  // // cout << "size of index2: " << index2.size() << endl;

  // traj_local.close();
  // traj_global.close();
  for_g2o_traj_with_loops << endl << endl << "FIX 1";
  for_g2o_traj_with_loops.close();	
  for_g2o_traj_with_loops_quat << endl << endl << "FIX 1";
  for_g2o_traj_with_loops_quat.close();
	// //Run Dloop here
	//output vec1 vec2 containing loop pairs

	

	

	// traj_local.open("edges.g2o");

	// for (int i = 0; i < relative.size(); i++)
	// {
	// 	Tr_relative tr = relative[i];
	// 	int index1_loop = tr.frame1;
	// 	int index2_loop = tr.frame2;

	// 	Matrix Tr = aboutX*tr.transfom;

	// 	double dx = Tr.getData(0,3);
 //    	double dz = Tr.getData(2,3);
 //    	double thetay = atan2(-Tr.getData(2,0), sqrt(Tr.getData(2,1)*Tr.getData(2,1) + Tr.getData(2,2)*Tr.getData(2,2)));

 //    	traj_local << "EDGE_SE2 " << index1_loop <<' '<< index2 <<' '<< -dz <<' '<< -dx <<' '<< thetay <<' '<< "1000 0 0 1000 0 10000" << endl;
	// }

	// traj_local.close();



	// for (int i = 0; i < vec1.size(); i++)
	// {
		

	// }



	//Read loop.txt pairs
	
	//matlab formatting in cpp for extracting loop pairs
	//run libviso2 on loop pairs
	//store edge transformations and concat with entire trajectory


	//cout << "size " << myvec.size() << endl;
	// //cout << "last " << myvec[numImages1-1] << endl;

	string writtenfile="data/merging/traj1_traj2_with_inliers.g2o";
	batch3d(writtenfile);
	return 0;
}

// ----------------------------------------------------------------------------

void SurfExtractor::operator() (const cv::Mat &im, 
  vector<cv::KeyPoint> &keys, vector<vector<float> > &descriptors) const
{
  // extract surfs with opencv
   static cv::Ptr<cv::xfeatures2d::SURF> surf_detector = 
    cv::xfeatures2d::SURF::create(400);
  
  surf_detector->setExtended(false);
  
  keys.clear(); // opencv 2.4 does not clear the vector
  vector<float> plain;
  surf_detector->detectAndCompute(im, cv::Mat(), keys, plain);
  
  // change descriptor format
  const int L = surf_detector->descriptorSize();
  descriptors.resize(plain.size() / L);

  unsigned int j = 0;
  for(unsigned int i = 0; i < plain.size(); i += L, ++j)
  {
    descriptors[j].resize(L);
    std::copy(plain.begin() + i, plain.begin() + i + L, descriptors[j].begin());
  }
}

// ----------------------------------------------------------------------------

// void my_dloop(std::vector<int> &index1, std::vector<int> &index2,string VOC_FILE1,string IMAGE_DIR1, int IMAGE_W1, int IMAGE_H1)
// {
//    demoDetector<Surf64Vocabulary, Surf64LoopDetector, FSurf64::TDescriptor>
//     demo(VOC_FILE1, IMAGE_DIR1, IMAGE_W1, IMAGE_H1);

//   try 
//   {  
//     // run the demo with the given functor to extract features
//     SurfExtractor extractor;
//     demo.run(index1,index2,"SURF64", extractor);
//   }
//   catch(const std::string &ex)
//   {
//     cout << "Error: " << ex << endl;
//   }
// }

void my_dloop(std::vector<loops> &loop, string VOC_FILE1,string IMAGE_DIR1,string IMAGE_DIR2, int IMAGE_W1, int IMAGE_H1)
{
   demoDetector<Surf64Vocabulary, Surf64LoopDetector, FSurf64::TDescriptor>
    demo(VOC_FILE1, IMAGE_DIR1,IMAGE_DIR2, IMAGE_W1, IMAGE_H1);

  try 
  {  
    // run the demo with the given functor to extract features
    SurfExtractor extractor;
    demo.run(loop,"SURF64", extractor);
  }
  catch(const std::string &ex)
  {
    cout << "Error: " << ex << endl;
  }
}

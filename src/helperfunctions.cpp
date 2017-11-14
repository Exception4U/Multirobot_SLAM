// To find number of images in a directory

#include <sys/types.h>
#include <dirent.h>
#include "helperfunctions.h"
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <stdint.h>

#include <viso_stereo.h>
//#include <png++/png.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>

#define PI 3.14159265
Matrix aboutX1 = Matrix::rotMatX(-PI,4);
Matrix new_tr1;

using namespace std;
using namespace cv;

int listdir(char *dir) {
  struct dirent *dp;
  DIR *fd;
  int counter =0;
 // cout << dir<<endl;

  if ((fd = opendir(dir)) == NULL) {
    fprintf(stderr, "listdir: can't open %s\n", dir);
    return -1;
  }
  while ((dp = readdir(fd)) != NULL) {
  if (!strcmp(dp->d_name, ".") || !strcmp(dp->d_name, ".."))
    continue;    /* skip self and parent */
  // printf("%s/%s\n", dir, dp->d_name);
  	counter++;
  //printf("%s\n", dp->d_name);
  }
  //cout<< counter<<endl;
  closedir(fd);
  return counter;
}


void my_libviso2(std::vector<Matrix> &Tr_local, std::vector<Matrix> &Tr_global, string dir, int numImg) {
  
  // set most important visual odometry parameters
  // for a full parameter list, look at: viso_stereo.h
  VisualOdometryStereo::parameters param;
  
  param.calib.f  = 722.658813477; // focal length in pixels
  param.calib.cu = 619.283325195; // principal point (u-coordinate) in pixels
  param.calib.cv = 360.0; // principal point (v-coordinate) in pixels
  param.base     = 0.12; // baseline in meters

  /**
  // calibration parameters for sequence 2010_03_09_drive_0019 
  param.calib.f  = 711.9212646484375; // focal length in pixels
  param.calib.cu = 647.0408325195312; // principal point (u-coordinate) in pixels
  param.calib.cv = 360.7899169921875; // principal point (v-coordinate) in pixels
  param.base     = 0.12; // baseline in meters
**/

  // calibration parameters for sequence 2010_03_09_drive_0019 
  // param.calib.f  = 645.24; // focal length in pixels
  // param.calib.cu = 635.96; // principal point (u-coordinate) in pixels
  // param.calib.cv = 194.13; // principal point (v-coordinate) in pixels
  // param.base     = 0.5707; // baseline in meters
  
  // init visual odometry
  VisualOdometryStereo viso(param);
  
  // current pose (this matrix transforms a point from the current
  // frame's camera coordinates to the first frame's camera coordinates)
  // std::vector<Matrix> vector_images1;
  // std::vector<Matrix> vector_images2;
  Matrix pose = Matrix::eye(4);
  Tr_global.push_back(pose);
  
  // loop through all frames i=0:372
  for (int32_t i=1; i<=numImg; i++) {

    // input file names
    char base_name[256]; sprintf(base_name,"%04d.jpg",i);
    string left_img_file_name  = dir + "/left/" + base_name;
    string right_img_file_name = dir + "/right/" + base_name;
    
    // cout << left_img_file_name << endl;
    // cout << right_img_file_name << endl;
    // catch image read/write errors here
    try {

      // load left and right input image
      Mat left_img_src, right_img_src;
      left_img_src = imread(left_img_file_name,CV_LOAD_IMAGE_COLOR);
      right_img_src = imread(right_img_file_name,CV_LOAD_IMAGE_COLOR);

      Mat left_img,right_img;
      cvtColor(left_img_src,left_img,CV_BGR2GRAY);
      cvtColor(right_img_src,right_img,CV_BGR2GRAY);
      // png::image< png::gray_pixel > left_img(left_img_file_name);
      // png::image< png::gray_pixel > right_img(right_img_file_name);

      // image dimensions
      // int32_t width  = left_img.get_width();
      // int32_t height = left_img.get_height();

      int32_t width  = left_img.cols;
      int32_t height = left_img.rows;

      // convert input images to uint8_t buffer
      uint8_t* left_img_data  = (uint8_t*)malloc(width*height*sizeof(uint8_t));
      uint8_t* right_img_data = (uint8_t*)malloc(width*height*sizeof(uint8_t));
      int32_t k=0;
      for (int32_t row=0; row < left_img.rows; row++) {
        for (int32_t col=0; col < left_img.cols; col++) {
          // left_img_data[k]  = left_img.get_pixel(u,v);
          // right_img_data[k] = right_img.get_pixel(u,v);
          left_img_data[k]  = left_img.at<uchar>(row,col);
          right_img_data[k] = right_img.at<uchar>(row,col);
          k++;
        }
      }

      // status
      cout << "Processing: Frame: " << i << endl;
      
      // compute visual odometry
      int32_t dims[] = {width,height,width};
      //==if (viso.process(left_img_data,right_img_data,dims)) {
      
        viso.process(left_img_data,right_img_data,dims);
        // on success, update current pose

        Tr_local.push_back(Matrix::inv(viso.getMotion()));

        if(i>1)
        {
          pose = pose * Matrix::inv(viso.getMotion());
          Tr_global.push_back(pose);  
        }
        
        // output some statistics
        double num_matches = viso.getNumberOfMatches();
        double num_inliers = viso.getNumberOfInliers();
        cout << ", Matches: " << num_matches;
        cout << ", Inliers: " << 100.0*num_inliers/num_matches << " %" << ", Current pose: " << endl;
        cout << pose << endl << endl;

      // } else {
      //   cout << " ... failed!" << endl;
      //   cin.get();
      // }

      // release uint8_t buffers
      free(left_img_data);
      free(right_img_data);

    // catch image read errors here
    } catch (...) {
      cerr << "ERROR: Couldn't read input files!" << endl;
      break;
    }
  }

  cout << "Demo complete! Exiting ..." << endl;
}


void my_libviso2_relative(std::vector<Tr_relative> &Tr_final, std::vector<loops> loop, std::string dir1, std::string dir2, int traj1_size)
{
   // set most important visual odometry parameters
  // for a full parameter list, look at: viso_stereo.h
  VisualOdometryStereo::parameters param;
  
  ofstream without_inliers;
  without_inliers.open("data/merging/merging_without_inliers.g2o");
  // calibration parameters for sequence 2010_03_09_drive_0019 
  param.calib.f  = 722.658813477; // focal length in pixels
  param.calib.cu = 619.283325195; // principal point (u-coordinate) in pixels
  param.calib.cv = 360.0; // principal point (v-coordinate) in pixels
  param.base     = 0.12; // baseline in meters

  /**
  param.calib.f  = 711.9212646484375; // focal length in pixels
  param.calib.cu = 647.0408325195312; // principal point (u-coordinate) in pixels
  param.calib.cv = 360.7899169921875; // principal point (v-coordinate) in pixels
  param.base     = 0.12; // baseline in meters
**/

  // calibration parameters for sequence 2010_03_09_drive_0019 
  // param.calib.f  = 645.24; // focal length in pixels
  // param.calib.cu = 635.96; // principal point (u-coordinate) in pixels
  // param.calib.cv = 194.13; // principal point (v-coordinate) in pixels
  // param.base     = 0.5707; // baseline in meters
  
  // init visual odometry
  VisualOdometryStereo viso(param);
  
  // current pose (this matrix transforms a point from the current
  // frame's camera coordinates to the first frame's camera coordinates)
  // std::vector<Matrix> vector_images1;
  // std::vector<Matrix> vector_images2;
  
  // loop through all frames i=0:372
  loops l;
  Tr_relative r;
  string dir_1,dir_2;
  int counter =0;

  for (int i=0; i<loop.size(); i++) 
  {
    l = loop[i];

    if (l.val1==1 && l.val2==1)
    {
      dir_1 = dir1;
      dir_2 = dir1;
      r.frame1 = l.idx1+1;
      r.frame2 = l.idx2+1;
    }
    else if(l.val1==1 && l.val2==2)
    {
      dir_1 = dir1;
      dir_2 = dir2;
      r.frame1 = l.idx1+1;
      r.frame2 = l.idx2 + traj1_size + 2;
    }
    else if(l.val1==2 && l.val2==2)
    {
      dir_1 = dir2;
      dir_2 = dir2;
      r.frame1 = l.idx1 + traj1_size + 2;
      r.frame2 = l.idx2 + traj1_size + 2;
    }

    Matrix pose = Matrix::eye(4);
    Matrix pose1 = Matrix::eye(4);
   
    string left_img_file_name;
    string right_img_file_name;

    for (int j = 1; j < 3; j++)
    {
      if (j==1)
      {
        
        char base_name[256]; sprintf(base_name,"%04d.jpg",l.idx1);
        left_img_file_name  = dir_1 + "/left/" + base_name;
        right_img_file_name = dir_1 + "/right/" + base_name;
        cout << " j==1 " << left_img_file_name << endl;

      }
      else
      {
        char base_name[256]; sprintf(base_name,"%04d.jpg",l.idx2);
        left_img_file_name  = dir_2 + "/left/" + base_name;
        right_img_file_name = dir_2 + "/right/" + base_name;
        cout << " j==2 " << left_img_file_name << endl;
      }
        
      // cout << left_img_file_name << endl;
      // cout << right_img_file_name << endl;
      // catch image read/write errors here
      try {

      // load left and right input image
      Mat left_img_src, right_img_src;
      left_img_src = imread(left_img_file_name,CV_LOAD_IMAGE_COLOR);
      right_img_src = imread(right_img_file_name,CV_LOAD_IMAGE_COLOR);

      Mat left_img,right_img;
      cvtColor(left_img_src,left_img,CV_BGR2GRAY);
      cvtColor(right_img_src,right_img,CV_BGR2GRAY);
      // png::image< png::gray_pixel > left_img(left_img_file_name);
      // png::image< png::gray_pixel > right_img(right_img_file_name);

      // image dimensions
      // int32_t width  = left_img.get_width();
      // int32_t height = left_img.get_height();

      int32_t width  = left_img.cols;
      int32_t height = left_img.rows;

      // convert input images to uint8_t buffer
      uint8_t* left_img_data  = (uint8_t*)malloc(width*height*sizeof(uint8_t));
      uint8_t* right_img_data = (uint8_t*)malloc(width*height*sizeof(uint8_t));
      int32_t k=0;
        for (int32_t row=0; row < left_img.rows; row++) {
          for (int32_t col=0; col < left_img.cols; col++) {
          // left_img_data[k]  = left_img.get_pixel(u,v);
          // right_img_data[k] = right_img.get_pixel(u,v);
            left_img_data[k]  = left_img.at<uchar>(row,col);
            right_img_data[k] = right_img.at<uchar>(row,col);
            k++;
          }
        }

      // status
        cout << "Processing: Frame between: " << r.frame1 << '\t' << l.val1 << '\t' << r.frame2 << '\t' << l.val2 << endl;
      
        // compute visual odometry
      int32_t dims[] = {width,height,width};
      //==if (viso.process(left_img_data,right_img_data,dims)) {
      
        bool pass = viso.process(left_img_data,right_img_data,dims);
        // on success, update current pose

        // output some statistics

        double num_matches = viso.getNumberOfMatches();
        double num_inliers = viso.getNumberOfInliers();
        double inliers_percent = 100.0*num_inliers/num_matches;
        cout << "Matches: " << num_matches;
        cout << ", Inlier percent: " << inliers_percent << " %" << endl;
        //cout << pose << endl << endl;

        

        // <-------------------------without_inliers (remove this afterwards)------------------------------->
      
        
        if (j==2)
        {
          pose1 = pose1 * Matrix::inv(viso.getMotion());
          double valx = pose1.getData(0,3);
          double valz = pose1.getData(2,3); 

          new_tr1 = aboutX1*pose1;

          double dx = new_tr1.getData(0,3);
          double dz = new_tr1.getData(2,3);
          double thetay = atan2(-new_tr1.getData(2,0), sqrt(new_tr1.getData(2,1)*new_tr1.getData(2,1) + new_tr1.getData(2,2)*new_tr1.getData(2,2)));
          
           without_inliers << "EDGE_SE2 " << r.frame1 <<' '<< r.frame2 <<' '<< dx <<' '<< dz <<' '<< thetay <<' '<< "1000 0 0 1000 0 10000" << endl; 
           counter++; 
        }
        

        // <-------------------------------------------------------------------------------------------------->

         // cout << "i: " << i << " result: "<<'\t' << << '\t' << "j: " << j <<endl << endl;
         if(j==2 && inliers_percent >= 10.0 && pass)
        {
          pose = pose * Matrix::inv(viso.getMotion());
          r.transform = pose;
          cout << pose << endl << endl;
          Tr_final.push_back(r);  
        }

      // } else {
      //   cout << " ... failed!" << endl;
      //   cin.get();
      // }

      // release uint8_t buffers
      free(left_img_data);
      free(right_img_data);
      // catch image read errors here
    } 
      catch (...) 
      {
        cerr << "ERROR: Couldn't read input files!" << endl;
        break;
      }
   }
  } 

  without_inliers.close();   

  cout << "without inliers: " << counter << endl;
  cout << "with inliers: " << Tr_final.size() << endl;
}

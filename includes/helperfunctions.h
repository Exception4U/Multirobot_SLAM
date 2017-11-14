
#include <string>
#include <vector>

#include <viso_stereo.h>
#include <vector>
#include "gtsambatch.h"
#ifndef _HELPER_FINCTIONS_h
#define _HELPER_FINCTIONS_h

// To find number of images in a directory
int listdir(char *dir);

// To find Visual Odometry using libviso2
void my_libviso2(std::vector<Matrix> &myvec1, std::vector<Matrix> &myvec2, std::string dir, int numImg);

// Class to hold relative poses

class Tr_relative
{
    public:
      int frame1;
      int frame2;
      Matrix transform;      

};

class loops
{
	public:
		int idx1;
		int idx2;
		int val1;
		int val2;
};

// To find relative transformations between loop closing frames
void my_libviso2_relative(std::vector<Tr_relative> &relative, std::vector<loops> loop, std::string dir1,std::string dir2,int s);

#endif

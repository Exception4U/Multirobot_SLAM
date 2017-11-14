#include "gtsambatch.h"

using namespace std;
using namespace gtsam;

void batch3d(string filename){
      
  //which one!
  std::pair<NonlinearFactorGraph::shared_ptr, Values::shared_ptr> data = readG2o(filename,true);
  //std::pair<NonlinearFactorGraph::shared_ptr, Values::shared_ptr> data = load3D(filename);
  
  Values initial(*data.second);
  NonlinearFactorGraph dataset;
  dataset = *data.first;
  std::cout<<"values, dataset sizes : "<<initial.size()<<", " <<dataset.size()<<endl;
  LevenbergMarquardtParams lmparams;
  LevenbergMarquardtOptimizer lmoptimizer(dataset,initial,lmparams);
  Values result;
  result=lmoptimizer.optimize();
  const string outputfilelm="batch3dMapMerge.g2o";       
  writeG2o(dataset,result,outputfilelm);
    
}

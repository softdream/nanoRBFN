#include "face_orientation_detect.h"

int main()
{
	std::cout<<" ---------------- K MEANS PP TEST -----------------"<<std::endl;

	using FaceOrientationDetectType = typename test::FaceOrientationDetect<double, 2500, 11, 3>;

	// 1. training using training dataset
	std::vector<FaceOrientationDetectType::DataType> training_data;
	std::vector<int> training_labels;

	FaceOrientationDetectType::loadTrainingDataSet( training_data, training_labels, 50 );
	

	FaceOrientationDetectType::faceOrientationTraining( training_data, training_labels );

	// 2. testing using testing dataset
	std::vector<FaceOrientationDetectType::DataType> testing_data;
	std::vector<int> testing_labels;
	
	FaceOrientationDetectType::loadTestingDataSet( testing_data, testing_labels, 30 );

	FaceOrientationDetectType::faceOrientationTesting( testing_data, testing_labels );

	return 0;
}

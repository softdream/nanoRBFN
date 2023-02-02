#ifndef __FACE_ORIENTATION_DETECT_H
#define __FACE_ORIENTATION_DETECT_H

#include <opencv2/opencv.hpp>

#include "rbf.h"


namespace test
{

template<typename T, int InputLayer, int HiddenLayer, int OutputLayer>
class FaceOrientationDetect
{
public:
	using ValueType = T;
	using DataType = typename rbf::RBF<ValueType, InputLayer, HiddenLayer, OutputLayer>::RBFDataType;

	static void loadTrainingDataSet( std::vector<DataType>& training_data, std::vector<int>& training_labels, const int data_num )
	{
		// 1. 
		for( int i = 0; i < data_num; i ++ ) {
			cv::Mat image = cv::imread( training_data_dir + "/Front/" + std::to_string( i + 1 ) + ".png" );
			training_data.push_back( image2Vector( image ) );
			training_labels.push_back( 0 );
		}

		// 2. 
		for( int i = 0; i < data_num; i ++ ) {
			cv::Mat image = cv::imread( training_data_dir + "/Left/" + std::to_string( i + 1 ) + ".png" );
                        training_data.push_back( image2Vector( image ) );
                        training_labels.push_back( 1 );
		}

		// 3. 
		for( int i = 0; i < data_num; i ++ ) {
                        cv::Mat image = cv::imread( training_data_dir + "/Right/" + std::to_string( i + 1 ) + ".png" );
                        training_data.push_back( image2Vector( image ) );
                        training_labels.push_back( 2 );
                }

		// 4. 
		scrambleDataSet( training_data, training_labels );
	}	

	static void loadTestingDataSet( std::vector<DataType>& testing_data, std::vector<int>& testing_labels, const int data_num )
	{
		// 1. 
		for( int i = 0; i < data_num; i ++ ) {
			cv::Mat image = cv::imread( testing_data_dir + "/Front/" + std::to_string( i + 1 ) + ".png" );
		
			testing_data.push_back( image2Vector( image ) );
			testing_labels.push_back( 0 );
		}

		// 2. 
		for( int i = 0; i < data_num; i ++ ) {
                        cv::Mat image = cv::imread( testing_data_dir + "/Left/" + std::to_string( i + 1 ) + ".png" );

                        testing_data.push_back( image2Vector( image ) );
                        testing_labels.push_back( 1 );
                }

		// 3. 
		for( int i = 0; i < data_num; i ++ ) {
                        cv::Mat image = cv::imread( testing_data_dir + "/Right/" + std::to_string( i + 1 ) + ".png" );

                        testing_data.push_back( image2Vector( image ) );
                        testing_labels.push_back( 2 );
                }

	}

	static void faceOrientationTraining( const std::vector<DataType>& training_data, 
					     const std::vector<int>& training_labels,
		       			     const ValueType learning_rate = 0.005,
					     const int iterations = 10 )
	{
		ValueType mse = 0.0;
		rbfn.training( training_data, training_labels, learning_rate, iterations, mse );
	}

	static void faceOrientationTesting( const std::vector<DataType>& testing_data, 
					    const std::vector<int>& testing_labels )
	{
		rbfn.testing( testing_data, testing_labels );
	}
private:
	static const DataType image2Vector( const cv::Mat& image )
	{
		cv::Mat image_gray;
		cv::cvtColor( image, image_gray, CV_RGB2GRAY );
		cv::equalizeHist( image_gray, image_gray );
	
		DataType data = DataType::Zero();
	
		int i = 0;
		for( auto it = image_gray.begin<uchar>(); it != image_gray.end<uchar>(); it ++ ) {
			data[i ++] = static_cast<ValueType>( *it ) / 128.0 - 1;
		}

		return data;
	}

	static void scrambleDataSet( std::vector<DataType>& training_data, std::vector<int>& training_labels )
	{
		std::vector<int> permutaion( training_data.size() );
		std::vector<DataType> data_tmp = training_data;
		std::vector<int> label_tmp = training_labels;

		for( int i = 0; i < training_data.size(); i ++ ) permutaion[i] = i;

		srand( unsigned( time(0) ) );
		random_shuffle( permutaion.begin(), permutaion.end() );

		for( int i = 0; i < training_data.size(); i ++ ) {
			training_data[ permutaion[i] ] = data_tmp[i];
			training_labels[ permutaion[i] ] = label_tmp[i];
		}
	}

private:
	static const std::string training_data_dir;
	static const std::string testing_data_dir;

	static rbf::RBF<T, InputLayer, HiddenLayer, OutputLayer> rbfn;
};

template<typename T, int InputLayer, int HiddenLayer, int OutputLayer>
const std::string FaceOrientationDetect<T, InputLayer, HiddenLayer, OutputLayer>::training_data_dir = "/home/arm/Test/rbf/training";

template<typename T, int InputLayer, int HiddenLayer, int OutputLayer>
const std::string FaceOrientationDetect<T, InputLayer, HiddenLayer, OutputLayer>::testing_data_dir = "/home/arm/Test/rbf/testing";

template<typename T, int InputLayer, int HiddenLayer, int OutputLayer>
rbf::RBF<T, InputLayer, HiddenLayer, OutputLayer> FaceOrientationDetect<T, InputLayer, HiddenLayer, OutputLayer>::rbfn;

}

#endif

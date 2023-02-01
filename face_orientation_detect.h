#ifndef __FACE_ORIENTATION_DETECT_H
#define __FACE_ORIENTATION_DETECT_H

#include <opencv2/opencv.hpp>

#include "rbf.h"


namespace test
{
	
class FaceOrientationDetect
{
public:
	using ValueType = double;
	using DataType = typename rbf::RBF<ValueType, 250, 10, 3>::RBFDataType;

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
};

const std::string FaceOrientationDetect::training_data_dir = "";

const std::string FaceOrientationDetect::testing_data_dir = "";

}

#endif

#ifndef __RBF_H
#define __RBF_H

#include <memory>
#include "k_means_pp.h"

namespace rbf
{

template<typename T, int InputLayer, int HiddenLayer, int OutputLayer>
class RBF
{
public:
	using RBFValueType = typename means::KMeansPP<T, InputLayer>::ValueType;
	using RBFDataType = typename means::KMeansPP<T, InputLayer>::DataType;
	
	using RBFHiddenLayerType = typename Eigen::Matrix<RBFValueType, HiddenLayer, 1>;

	using RBFWeightsType = typename Eigen::Matrix<RBFValueType, HiddenLayer, OutputLayer>;
	
	using RBFOutputLayerType = typename Eigen::Matrix<RBFValueType, 1, OutputLayer>;

	RBF() 
	{
	
	}

	~RBF()
	{
	
	}

	const RBFValueType training( const std::vector<RBFDataType>& training_data, 
				     //const std::vector<RBFOutputLayerType>& traning_labels,
				     const std::vector<int>& traning_labels,
				     const RBFValueType learning_rate,
				     const int num_iterations,
				     RBFValueType& mse	)
	{
		std::cout<<"Starting RBF NetWork Training with "<<HiddenLayer<<" units and learning rate is "<<learning_rate<<std::endl;
		std::cout<<"Getting RBF Centroids using Kmeans ++ ..."<<std::endl;
	
		// 1. caculate the RBF Centroids
		k_means_ptr_ = std::make_unique<means::KMeansPP<T, InputLayer>>( training_data );
		
		k_means_ptr_->runKmeansPP( HiddenLayer, rbf_centroids ); 

		// 2. caculate the parameter of  Gaussian Kernel Function
		kernel_para = caculateGaussianKernelParameter( );

		// 3. caculate the output values of nodes of the hidden layer
		std::vector<RBFHiddenLayerType> hidden_layer_outputs;
		
		caculateHiddenLayerOutputs( training_data, hidden_layer_outputs );

		// 4. caculate the weights from the hidden layer to the output layer
		// 4.1 init the weights matrix 
		//RBFWeightsType weights = RBFWeightsType::Zero();
		weights.assign( OutputLayer, RBFHiddenLayerType::Zero() );
		initWeightsMatrix( weights );

		RBFValueType accuracy = 0.0;
		// 4.2 train the weights matrix
		for( int iter = 0; iter < num_iterations; iter ++ ) {
			for( int i = 0; i < training_data.size(); i ++ ) {
				//RBFOutputLayerType predict = hidden_layer_outputs[i].transpose() * weights; 
				
				//Eigen::Matrix<RBFValueType, 1, OutputLayer> error = traning_labels[i] - predict;
				
				for( int label = 0; label < OutputLayer; label ++ ) {
					RBFValueType predict = hidden_layer_outputs[i].transpose() * weights[label];
					predict = std::max( std::min( predict, 1.0 ), -1.0 );

					RBFValueType truth_base = traning_labels[i] == label ? 1.0 : -1.0;
					RBFValueType error = truth_base - predict;
					
					RBFHiddenLayerType delta = hidden_layer_outputs[i] * error * learning_rate;
					weights[label] += delta;
				}
			}

			// 4.3 statistics information 
			mse = 0;
			accuracy = 0;
			for( int i = 0; i < training_data.size(); i ++ ) {
				RBFValueType error = 0;
				int prediction = predict( training_data[i], error );
				
				if( prediction == traning_labels[i] ) {
					accuracy ++;
				}
	
				mse += error * error;
			}
		
			mse *= 1.0 / static_cast<RBFValueType>( training_data.size() );
			accuracy *= 1.0 / static_cast<RBFValueType>( training_data.size() );

			std::cout<<"Training Iteration : "<<iter<<", MSE = [ "<<mse<<" ], ACC = ["<<accuracy * 100<<" ], Progress : "<<static_cast<RBFValueType>( iter ) / static_cast<RBFValueType>( num_iterations ) * 100<<"------------------------------------------"<<std::endl;
		}

		return accuracy;
	}
	
	const int predict( const RBFDataType& input_data, const RBFValueType& error )
	{
		RBFValueType maxi = std::numeric_limits<RBFValueType>::min();
		
		int best_label = -1;
		
		RBFHiddenLayerType hidden_layer_output;
			
		caculateHiddenLayerOutputs( input_data, hidden_layer_output );

		for( int label = 0; label < OutputLayer; label ++ ) {
			RBFValueType predict = hidden_layer_output * weights[label];
			
			if( maxi < predict ) {
				maxi = predict;
				best_label = label;
			}
		}

		assert( best_label != -1 );
		error = static_cast<RBFValueType>( best_label ) - maxi;

		return best_label;
	}

private:
	void initWeightsMatrix( std::vector<RBFHiddenLayerType>& weights )
	{
		std::random_device rd;
                std::default_random_engine random_engine( rd() );
                std::uniform_real_distribution<RBFValueType> random_real_gen( -1, 1 );

		for( auto& label : weights ) {
			for( int i = 0; i < label.size(); i ++ ) {
				label[i] = random_real_gen( random_engine ) ;
			}
		}
	}

	void initWeightsMatrix( RBFWeightsType& weights )
	{
		std::random_device rd;
		std::default_random_engine random_engine( rd() );
		std::uniform_real_distribution<RBFValueType> random_real_gen( -1, 1 );

		//weights.unaryExpr( []( RBFValueType& dummy ){ return random_real_gen( random_engine ); } );
	
		for( int i = 0; i < weights.rows(); i ++ ){
                        for( int j = 0; j < weights.cols(); j ++ ){
                                weights( i, j ) = random_real_gen( random_engine ) ;
                        }
                }

	}

	void caculateHiddenLayerOutputs( const std::vector<RBFDataType>& training_data,
					 std::vector<RBFHiddenLayerType>& hidden_layer_outputs )
	{
		for( int i = 0; i < training_data.size(); i ++ ) {
			for( int j = 0; j < HiddenLayer; j ++ ) {
				hidden_layer_outputs[i][j] = getGaussianKernelFunctionOutput( training_data[i], rbf_centroids[j] );
			}
		}
	}

	void caculateHiddenLayerOutputs( const RBFDataType& input_data,
					 RBFHiddenLayerType& hidden_layer_output )
	{
		for( int j = 0; j < HiddenLayer; j ++ ) {
                	hidden_layer_output[j] = getGaussianKernelFunctionOutput( input_data, rbf_centroids[j] );
                }

	}

	const RBFValueType getGaussianKernelFunctionOutput( const RBFDataType& data1, const RBFDataType& data2 ) const
	{
		return ::exp( kernel_para * squaredDistance( data1, data2 ) );
	}

	const RBFValueType caculateGaussianKernelParameter( )
	{
		RBFValueType max_covarince = std::numeric_limits<RBFValueType>::min();

		for( int i = 0; i < rbf_centroids.size(); i ++ ) {
			for( int j = i + 1; j < rbf_centroids.size(); j ++ ) {
				max_covarince = std::max( max_covarince, squaredDistance( rbf_centroids[i], rbf_centroids[j] ) );
			}
		}

		max_covarince *= ( 1.0 / static_cast<RBFValueType>( rbf_centroids.size() ) );

		return ( -1.0 / ( 2.0 * max_covarince ) );
	}

	const RBFValueType distance( const RBFDataType& p1, const RBFDataType& p2 ) const
	{
		return ( p1 - p2 ).norm();
	}

	const RBFValueType squaredDistance( const RBFDataType& p1, const RBFDataType& p2 ) const
	{
		return ( p1 - p2 ).squaredNorm();
	}

private:
	std::unique_ptr<means::KMeansPP<T, InputLayer>> k_means_ptr_;

	// centroids of the training data
	std::vector<RBFDataType> rbf_centroids;

	// the kernel function's paramter
	RBFValueType kernel_para = 0;
	
	// the weights from hidden layer to output layer
	std::vector<RBFHiddenLayerType> weights;
};


}

#endif

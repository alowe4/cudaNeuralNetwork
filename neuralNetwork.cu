#include <stdio.h>

__device__ double sigmoid(double x){
	return 1.0f / (1.0f + exp(-x));

}

__device__ double dSigmoid(double x){
	return x * (1.0f - x); 
}


__device__ void matrix_multiply_simple(double *a, double *b, double *ab, size_t width)
{
	size_t n = width; 
        // We take the thread id and the thread id will tell us what the current x & y is 
	int col  = blockIdx.x * blockDim.x + threadIdx.x; 
	int row  = blockIdx.y * blockDim.y + threadIdx.y; 
	
	if((row < n) && (col < n)){
		double pVal = 0; 
		for(int k = 0; k < n; ++k){
			//printf("k: %i, a[k]: %d\n", k, a[k]);
			pVal += a[k] * b[k * n + col]; 
		}
		ab[row * n + col] = pVal; 
	}

}

__global__ void hiddenLayerCompletion(double* hiddenLayer, 


__global__ void forwardFeed(double* inputLayer, double* hiddenWeights, double* hiddenLayer, double* outputLayer, double* outputWeights, double* outputLayerBias, double* hiddenLayerBias, int numHiddenNodes, int numInputs, int numOutputs, int trainingSetIndex){
	int i = trainingSetIndex; 
	for(int j = 0 ; j < numHiddenNodes; j++){
		double activation = hiddenLayerBias[j];
		//for(int k = 0; k < numInputs; k++){
		//	activation += inputLayer[(i * numInputs) + k] * hiddenWeights[(j * numInputs) + k];
		//}
		//multiply matrixes
		double temp[2] = { inputLayer[i * numInputs], inputLayer[(i * numInputs) + 1] };
		double temp2[4];
		matrix_multiply_simple(temp, hiddenWeights,temp2,4); 
		//sum result matrix
		//need new var passed in to ff
		for(int i =0; i<4;i++){
			activation += temp2[i];
		}
		hiddenLayer[j] = sigmoid(activation);
		
	}

	// Compute Output Layer Activation
	for(int j = 0; j < numOutputs; j++){
		double activation = outputLayerBias[j];
		for(int k = 0; k < numHiddenNodes; k++){
			activation += hiddenLayer[k] * outputWeights[j + (k * numOutputs)];
		}
		outputLayer[j] = sigmoid(activation);

	}



}

// Training Outputs Checks If Our Values Are Correct
__global__ void backpropogate(double* trainingInputs, double* hiddenLayer, double* hiddenWeights, double* outputLayer, double* outputWeights, double* trainingOutputs, double* hiddenLayerBias, double* outputLayerBias, int numHiddenNodes, int numInputs, int numOutputs, int trainingSetIndex, double lr){

	int i = trainingSetIndex; 

	double deltaOutput[1];

	// Calcualte Mean Squared Error In Output Weights
	for(int j = 0; j < numOutputs; j++){
		double dError = (trainingOutputs[i * numOutputs + j] - outputLayer[j]);
		deltaOutput[j] = dError * dSigmoid(outputLayer[j]);
	}


	double deltaHidden[4];
	// Calcuate Mean Squared Error in Hidden Weights
	for(int j = 0; j < numHiddenNodes; j++){
		double dError = 0.0f; 
		for(int k = 0; k < numOutputs; k++){
			dError += deltaOutput[k] * outputWeights[(j * 1) + k]; 
		}
		deltaHidden[j] = dError * dSigmoid(hiddenLayer[j]); 
	}

	// Apply Change in Output Weights
	for(int j = 0; j < numOutputs; j++){
		outputLayerBias[j] += deltaOutput[j] * lr;
		for(int k = 0; k < numHiddenNodes; k++){
			outputWeights[(k * numOutputs) + j] += hiddenLayer[k] * deltaOutput[j] * lr;
		}
	}

	// Apply Change in Hidden Weights
	for(int j = 0; j < numHiddenNodes; j++){
		hiddenLayerBias[j] += deltaHidden[j] * lr; 
		for(int k = 0; k < numInputs; k++){
			hiddenWeights[(k * numOutputs) + j] += trainingInputs[(i * numInputs) + k] * deltaHidden[j] * lr; 
		}
	}

}


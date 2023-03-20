#include <stdio.h>

__device__ double sigmoid(double x){
	return 1.0f / (1.0f + exp(-x));

}

__global__ void sigmoidActivationForward(double* src, double* dst,
                                         int Z_x_dim, int Z_y_dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < Z_x_dim * Z_y_dim) {
        dst[index] = sigmoid(src[index]);
    }
}



__device__ double dSigmoid(double x){
	return x * (1.0f - x); 
}


__global__ void forwardFeedP1(double* inputLayer, double* hiddenWeights, double* hiddenLayer, double* outputLayer, double* outputWeights, double* outputLayerBias, double* hiddenLayerBias, int numHiddenNodes, int numInputs, int numOutputs, int trainingSetIndex, double* HiddenLayerAct){
	int i = trainingSetIndex; 
	for(int j = 0 ; j < numHiddenNodes; j++){
		HiddenLayerAct[j] =0;
		double activation = hiddenLayerBias[j];
		for(int k = 0; k < numInputs; k++){
			activation += inputLayer[(i * numInputs) + k] * hiddenWeights[(j * numInputs) + k];
		}
		HiddenLayerAct[j] = activation;
		//hiddenLayer[j] = sigmoid(activation);
	}
}

__global__ void forwardFeedP2(double* inputLayer, double* hiddenWeights, double* hiddenLayer, double* outputLayer, double* outputWeights, double* outputLayerBias, double* hiddenLayerBias, int numHiddenNodes, int numInputs, int numOutputs, int trainingSetIndex, double* OutputLayerAct){

	// Compute Output Layer Activation
	for(int j = 0; j < numOutputs; j++){
		OutputLayerAct[j]=0;
		double activation = outputLayerBias[j];
		for(int k = 0; k < numHiddenNodes; k++){
			activation += hiddenLayer[k] * outputWeights[j + (k * numOutputs)];
		}
		OutputLayerAct[j] = activation;
//		outputLayer[j] = sigmoid(activation);

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


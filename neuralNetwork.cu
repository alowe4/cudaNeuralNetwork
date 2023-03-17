// What will the neural network do?
#include <stdio.h>

__device__ float sigmoid(float x){
	return 1.0f / (1.0f + exp(-x));
}
__global__ void sigmoidActivationForward(float* src, float* dst,
                                         int Z_x_dim, int Z_y_dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < Z_x_dim * Z_y_dim) {
        dst[index] = sigmoid(src[index]);
    }
}

__device__ float dSigmoid(float x){
	return x * (1.0f - x);
}




__global__ void forwardFeedP1(float* inputLayer, float* hiddenWeights, float* hiddenLayer, float* outputLayer, float* outputWeights, float* outputLayerBias, float* hiddenLayerBias, int numHiddenNodes, int numInputs, int numOutputs, int trainingSetIndex, float* HiddenLayerAct){
	int i = trainingSetIndex;
	//float* arr = malloc(sizeof(float) * 4); 
	for(int j = 0 ; j < numHiddenNodes; j++){
		HiddenLayerAct[j] =0;
		float activation = hiddenLayerBias[j];
		for(int k = 0; k < numInputs; k++){
			activation += inputLayer[(i * numInputs) + k] * hiddenWeights[(k * numInputs) + j];
		}
		HiddenLayerAct[j] = activation;
		//printf("hiddenLayer[%d]: %f\n\n",j, hiddenLayer[j]);
		//hiddenLayer[j] = sigmoid(activation);
	}
	
	
}

__global__ void forwardFeedP2(float* inputLayer, float* hiddenWeights, float* hiddenLayer, float* outputLayer, float* outputWeights, float* outputLayerBias, float* hiddenLayerBias, int numHiddenNodes, int numInputs, int numOutputs, int trainingSetIndex, float* OutputLayerAct){
		
	// Compute Output Layer Activation
	for(int j = 0; j < numOutputs; j++){
		OutputLayerAct[j]=0;
		float activation = outputLayerBias[j];
		for(int k = 0; k < numHiddenNodes; k++){
			activation += hiddenLayer[k] * outputWeights[j + (k * numOutputs)];
		}
		OutputLayerAct[j] = activation;
		//outputLayer[j] = sigmoid(activation);
	}




}


__global__ void backpropogate(float* trainingInputs, float* hiddenLayer, float* hiddenWeights, float* outputLayer, float* outputWeights, float* trainingOutputs, float* hiddenLayerBias, float* outputLayerBias, int numHiddenNodes, int numInputs, int numOutputs, int trainingSetIndex, float lr){

	int i = trainingSetIndex;
	float deltaOutput[1];

	// Calcualte Mean Squared Error In Output Weights
	for(int j = 0; j < numOutputs; j++){
		float dError = (trainingOutputs[i * numOutputs + j] - outputLayer[j]);
		deltaOutput[j] = dError * dSigmoid(outputLayer[j]);
	}

	float deltaHidden[4];
	// Calcuate Mean Squared Error in Hidden Weights
	for(int j = 0; j < numHiddenNodes; j++){
		float dError = 0.0f;
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


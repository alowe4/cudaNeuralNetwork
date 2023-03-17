// What will the neural network do? 
#include <stdio.h>

__device__ float sigmoid(float x){
	return 1.0f / (1.0f + exp(-x));

}

__device__ float dSigmoid(float x){
	return x * (1.0f - x); 
}



__global__ void forwardFeed(float* inputLayer, float* hiddenWeights, float* hiddenLayer, float* outputLayer, float* outputWeights, float* outputLayerBias, float* hiddenLayerBias, int numHiddenNodes, int numInputs, int numOutputs, int trainingSetIndex){
	int i = trainingSetIndex; 
	for(int j = 0 ; j < numHiddenNodes; j++){

		float activation = hiddenLayerBias[j];
		for(int k = 0; k < numInputs; k++){
			activation += inputLayer[(i * numInputs) + k] * hiddenWeights[(k * numInputs) + j];
//			printf("%f += %f * %f\n", activation, inputLayer[i * numInputs + k], hiddenWeights[(k * numInputs + j)]);
		}
//		printf("\n\n");
		hiddenLayer[j] = sigmoid(activation);
	}

	// Compute Output Layer Activation
	for(int j = 0; j < numOutputs; j++){
		float activation = outputLayerBias[j];
		for(int k = 0; k < numHiddenNodes; k++){
			activation += hiddenLayer[k] * outputWeights[j + (k * numOutputs)];
		}
		outputLayer[j] = sigmoid(activation);

	}



}

// Training Outputs Checks If Our Values Are Correct
__global__ void backpropogate(float* trainingInputs, float* hiddenLayer, float* hiddenWeights, float* outputLayer, float* outputWeights, float* trainingOutputs, float* hiddenLayerBias, float* outputLayerBias, int numHiddenNodes, int numInputs, int numOutputs, int trainingSetIndex, float lr){


	// Made & Used in Backpropoagate
	int i = trainingSetIndex; 

	float deltaOutput[1];

	//printf("Training Outputs: ");
	// Calcualte Mean Squared Error In Output Weights
	for(int j = 0; j < numOutputs; j++){

	//	printf("%f ", trainingOutputs[i * numOutputs + j]);
		float dError = (trainingOutputs[i * numOutputs + j] - outputLayer[j]);
		deltaOutput[j] = dError * dSigmoid(outputLayer[j]);
	}
	//printf("\nOutput Layer: %f\n", outputLayer[0]);


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
	//		printf("%f += %f * %f * %f\n", outputWeights[(k*numOutputs)+j], hiddenLayer[k], deltaOutput[j], lr); 
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



// Sigmoid function

// Forward Feeding Function

// Backpropogation Function

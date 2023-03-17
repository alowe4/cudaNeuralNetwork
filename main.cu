#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#ifndef RAND_MAX
#define RAND_MAX 32767
#endif

#include "neuralNetwork.h"

float init_weight(){ return ((float) rand())/ ((float)RAND_MAX);}


void shuffle(int* array, size_t n){
	if(n > 1){
		size_t i;
		for(i = 0; i < n - 1; i++){
			size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
			int t = array[j];
			array[j] = array[i];
			array[i] = t;
		}
	}

}


int main(int argc, char** argv){
	if(argc < 2){
		printf("add some params");
		exit(0);
	}

	//int gridSize = 4;
	//int blockSize = 1;
	time_t t;
 	srand((unsigned)time(&t));
	// Set the learning rate & epochs
	int epochs = 10000;
	float lr = 1.0f;

	int numInputs = 2;
	int numHiddenNodes = 4;
	int numOutputs = 1;

	float training_inputs[8] = {0.0f,0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f};
	float training_outputs[4] = {0.0f, 1.0f, 1.0f, 0.0f};
	int trainingSetOrder[] = {0,1,2,3};
	int numTrainingSets = 4;

	// Initialize all the arrays into memory
	float* hiddenLayer = (float*) malloc(numHiddenNodes * sizeof(float));
	float* outputLayer = (float*) malloc(numOutputs * sizeof(float));
	
	float* hiddenLayerAct = (float*) malloc(numHiddenNodes * sizeof(float));
	float* outputLayerAct = (float*) malloc(numOutputs * sizeof(float));

	float* hiddenLayerBias = (float*) malloc(numHiddenNodes * sizeof(float));
	float* outputLayerBias = (float*) malloc(numOutputs * sizeof(float));

	float* hiddenWeights = (float*)malloc(numInputs * numHiddenNodes* sizeof(float));
	float* outputWeights = (float*)malloc(numHiddenNodes * numOutputs * sizeof(float));

	//cuda
	float* cuHiddenLayer;
	float* cuOutputLayer;
	
	float* cuHiddenLayerAct;
	float* cuOutputLayerAct;
	
	float* cuHiddenLayerBias;
	float* cuOutputLayerBias;
	float* cuOutputWeights;
	float* cuHiddenWeights;
	float* cuTrainingInputs;
	float* cuTrainingOutputs;
	int* cuTrainingSetOrder;

	cudaMalloc((void**)&cuHiddenLayer, numHiddenNodes * sizeof(float));
	cudaMalloc((void**)&cuOutputLayer, numOutputs * sizeof(float));
	
	cudaMalloc((void**)&cuHiddenLayerAct, numHiddenNodes * sizeof(float));
	cudaMalloc((void**)&cuOutputLayerAct, numOutputs * sizeof(float));
	
	cudaMalloc((void**)&cuHiddenLayerBias, numHiddenNodes * sizeof(float));
	cudaMalloc((void**)&cuOutputLayerBias, numOutputs * sizeof(float));
	cudaMalloc((void**)&cuHiddenWeights, numInputs * numHiddenNodes * sizeof(float));
	cudaMalloc((void**)&cuTrainingInputs, 8 * sizeof(float));
	cudaMalloc((void**)&cuTrainingOutputs, 4 * sizeof(float));
	cudaMalloc((void**)&cuTrainingSetOrder, 4 * sizeof(int));
	cudaMalloc((void**)&cuOutputWeights, numHiddenNodes * numOutputs *  sizeof(float));

	// Initialize All The Weights
	for(int i = 0; i < numInputs; i++){
		for(int j = 0; j < numHiddenNodes; j++){
			hiddenWeights[(i * 2) + j] = init_weight();
		}
	}
	for(int i=0;i<numHiddenNodes;i++){
		hiddenLayerBias[i] = init_weight();
		for(int j=0; j<numOutputs; j++){
			outputWeights[(2 * i )+ j] = init_weight();
		}
	}

	for(int i = 0; i<numOutputs; i++){
		outputLayerBias[i] = init_weight();
	}


	//cuda memory copy to device
	cudaMemcpy(cuHiddenLayer, hiddenLayer, numHiddenNodes * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cuOutputLayer, outputLayer, numOutputs * sizeof(float), cudaMemcpyHostToDevice);
	
	cudaMemcpy(cuHiddenLayerAct, hiddenLayerAct, numHiddenNodes * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cuOutputLayerAct, outputLayerAct, numOutputs * sizeof(float), cudaMemcpyHostToDevice);

	
	cudaMemcpy(cuHiddenLayerBias, hiddenLayerBias, numHiddenNodes * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cuOutputLayerBias, outputLayerBias, numOutputs * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cuHiddenWeights, hiddenWeights, numInputs * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cuTrainingInputs, training_inputs, 8 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cuTrainingOutputs, training_outputs, 4 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cuTrainingSetOrder, trainingSetOrder, 4 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cuOutputWeights, outputWeights, numHiddenNodes * sizeof(float), cudaMemcpyHostToDevice);



//------------------------------------------------------------------------------------------------------------------
//start epochs

	for(int n = 0; n < epochs; n++){
		shuffle(trainingSetOrder, numTrainingSets);
		for(int x = 0; x < numTrainingSets; x++){
			int i = trainingSetOrder[x];
			forwardFeedP1<<<1, 1>>>(cuTrainingInputs, cuHiddenWeights, cuHiddenLayer, cuOutputLayer, cuOutputWeights, cuOutputLayerBias, cuHiddenLayerBias, numHiddenNodes, numInputs, numOutputs, i, cuHiddenLayerAct);
			cudaDeviceSynchronize();

			sigmoidActivationForward<<<1,4>>>(cuHiddenLayerAct,cuHiddenLayer,4,1);
			cudaDeviceSynchronize();

			forwardFeedP2<<<1, 1>>>(cuTrainingInputs, cuHiddenWeights, cuHiddenLayer, cuOutputLayer, cuOutputWeights, cuOutputLayerBias, cuHiddenLayerBias, numHiddenNodes, numInputs, numOutputs, i, cuOutputLayerAct);
			cudaDeviceSynchronize();
			
			sigmoidActivationForward<<<1,1>>>(cuOutputLayerAct,cuOutputLayer,1,1);
			cudaDeviceSynchronize();

			backpropogate<<<1, 1>>>(cuTrainingInputs, cuHiddenLayer, cuHiddenWeights, cuOutputLayer, cuOutputWeights, cuTrainingOutputs, cuHiddenLayerBias, cuOutputLayerBias, numHiddenNodes, numInputs, numOutputs, i, lr);
			cudaDeviceSynchronize();
		}
	}
	
	// Predict Function

	// Create two pieces of test input
	float test_input[2] ={atof(argv[1]), atof(argv[2])};
	float* cuInputs;
	cudaMalloc((void**)&cuInputs, 2 * sizeof(float));
	cudaMemcpy(cuInputs, test_input, 2 * sizeof(float), cudaMemcpyHostToDevice);
        
        forwardFeedP1<<<1, 1>>>(cuInputs, cuHiddenWeights, cuHiddenLayer, cuOutputLayer, cuOutputWeights, cuOutputLayerBias, cuHiddenLayerBias, numHiddenNodes, numInputs, numOutputs, 0,cuHiddenLayerAct);
			cudaDeviceSynchronize();

	forwardFeedP2<<<1, 1>>>(cuInputs, cuHiddenWeights, cuHiddenLayer, cuOutputLayer, cuOutputWeights, cuOutputLayerBias, cuHiddenLayerBias, numHiddenNodes, numInputs, numOutputs, 0,cuOutputLayerAct);
			cudaDeviceSynchronize();
			


        // Transfer the memory off of the GPU to the CPU
        cudaMemcpy(outputLayer, cuOutputLayer, numOutputs * sizeof(float), cudaMemcpyDeviceToHost);

        printf("%f\n", outputLayer[0]);

}



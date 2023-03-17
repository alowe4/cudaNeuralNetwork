#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#ifndef RAND_MAX
#define RAND_MAX 32767
#endif

#include "neuralNetwork.h"

double init_weight(){ return ((double) rand())/ ((double)RAND_MAX);}


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
	double lr = 1.0f;

	int numInputs = 2;
	int numHiddenNodes = 4;
	int numOutputs = 1;

	double training_inputs[8] = {0.0f,0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f};
	double training_outputs[4] = {0.0f, 1.0f, 1.0f, 0.0f};
	int trainingSetOrder[] = {0,1,2,3};

	// Initialize all the arrays into memory
	double* hiddenLayer = (double*) malloc(numHiddenNodes * sizeof(double));
	double* outputLayer = (double*) malloc(numOutputs * sizeof(double));

	double* hiddenLayerBias = (double*) malloc(numHiddenNodes * sizeof(double));
	double* outputLayerBias = (double*) malloc(numOutputs * sizeof(double));

	double* hiddenWeights = (double*)malloc(numInputs * numHiddenNodes* sizeof(double));
	double* outputWeights = (double*)malloc(numHiddenNodes * numOutputs * sizeof(double));

	//cuda
	double* cuHiddenLayer;
	double* cuOutputLayer;
	double* cuHiddenLayerBias;
	double* cuOutputLayerBias;
	double* cuOutputWeights;
	double* cuHiddenWeights;
	double* cuTrainingInputs;
	double* cuTrainingOutputs;
	int* cuTrainingSetOrder;

	cudaMalloc((void**)&cuHiddenLayer, numHiddenNodes * sizeof(double));
	cudaMalloc((void**)&cuOutputLayer, numOutputs * sizeof(double));
	cudaMalloc((void**)&cuHiddenLayerBias, numHiddenNodes * sizeof(double));
	cudaMalloc((void**)&cuOutputLayerBias, numOutputs * sizeof(double));
	cudaMalloc((void**)&cuHiddenWeights, numInputs * numHiddenNodes * sizeof(double));
	cudaMalloc((void**)&cuTrainingInputs, 8 * sizeof(double));
	cudaMalloc((void**)&cuTrainingOutputs, 4 * sizeof(double));
	cudaMalloc((void**)&cuTrainingSetOrder, 4 * sizeof(int));
	cudaMalloc((void**)&cuOutputWeights, numHiddenNodes * numOutputs *  sizeof(double));

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


	int numTrainingSets = 4; 
	cudaMemcpy(cuHiddenLayer, hiddenLayer, numHiddenNodes * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(cuOutputLayer, outputLayer, numOutputs * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(cuHiddenLayerBias, hiddenLayerBias, numHiddenNodes * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(cuOutputLayerBias, outputLayerBias, numOutputs * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(cuHiddenWeights, hiddenWeights, numInputs * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(cuTrainingInputs, training_inputs, 8 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(cuTrainingOutputs, training_outputs, 4 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(cuTrainingSetOrder, trainingSetOrder, 4 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cuOutputWeights, outputWeights, numHiddenNodes * sizeof(double), cudaMemcpyHostToDevice);









	for(int n = 0; n < epochs; n++){
		
		shuffle(trainingSetOrder, numTrainingSets);
		
		for(int x = 0; x < numTrainingSets; x++){
			int i = trainingSetOrder[x]; 
			//printf("%d\n", n);
			forwardFeed<<<1, 1>>>(cuTrainingInputs, cuHiddenWeights, cuHiddenLayer, cuOutputLayer, cuOutputWeights, cuOutputLayerBias, cuHiddenLayerBias, numHiddenNodes, numInputs, numOutputs, i);	
			cudaDeviceSynchronize(); 

			backpropogate<<<1, 1>>>(cuTrainingInputs, cuHiddenLayer, cuHiddenWeights, cuOutputLayer, cuOutputWeights, cuTrainingOutputs, cuHiddenLayerBias, cuOutputLayerBias, numHiddenNodes, numInputs, numOutputs, i, lr);	

			cudaDeviceSynchronize();
			// Predict Function
				
		}

			
	
	}
	
	
	//  Transfer memory off of GPU to CPU 
	// To run predict we have to:
       	// Specify user inputs	

       // Create two pieces of test input
       double test_input[2] ={atof(argv[1]), atof(argv[2])}; 
       
       double* cuInputs; 
       
       cudaMalloc((void**)&cuInputs, 2 * sizeof(double));
       
       cudaMemcpy(cuInputs, test_input, 2 * sizeof(double), cudaMemcpyHostToDevice);
       
       forwardFeed<<<1, 1>>>(cuInputs, cuHiddenWeights, cuHiddenLayer, cuOutputLayer, cuOutputWeights, cuOutputLayerBias, cuHiddenLayerBias, numHiddenNodes, numInputs, numOutputs, 0);	
       cudaDeviceSynchronize();

       // Transfer the memory off of the GPU to the CPU 
       //outputLayer[0] = 15.00f;
       cudaMemcpy(outputLayer, cuOutputLayer, numOutputs * sizeof(double), cudaMemcpyDeviceToHost);

       // Run the forward feed function given the Hidden Weights

       // Then we output 

       printf("%f\n", outputLayer[0]);	


}



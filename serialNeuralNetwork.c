#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#ifndef RAND_MAX
#define RAND_MAX 32767 
#endif
// Goal: Create a C Implementation of a Xor Neural Network

// Activation Function and Its Derivative
double sigmoid(double x){ return 1 / (1 + exp(-x)); }
double dSigmoid(double x) { return x * (1 - x); }

// Init all weights and biases between 0 & 1
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

// Forward Feed function
// Back propogate
// Fit 
// Update Weights
// Predict
// Calcuate MSE
//

int main(int argc, char** argv){
	time_t t; 
	srand((unsigned)time(&t)); 
	if(argc < 2){
		printf("add some params");
		exit(0);
	}
	// The number of inputs
	static const int numInputs = 2; 
	static const int numHiddenNodes = 4;
	static const int numOutputs = 1; 


	// Set Up The Neural Network
	// Define the Dimensions of the Hidden Layers
	double hiddenLayer[numHiddenNodes]; 
	double outputLayer[numOutputs]; 

	// Define the Bias of the Hidden Layers
	double hiddenLayerBias[numHiddenNodes];
	double outputLayerBias[numOutputs];

	// Define the Hidden Weights
	double hiddenWeights[numInputs][numHiddenNodes];
	double outputWeights[numHiddenNodes][numOutputs];

	// Set up the Training Input / Output 
	static const int numTrainingSets = 4;
	// Define the Training Inputs
	double training_inputs[][2] = { {0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f} }; 

	double training_outputs[][1] = { {0.0f}, {1.0f}, {1.0f}, {0.0f} }; 

	for(int i = 0; i < numInputs; i++){
		for(int j = 0; j < numHiddenNodes; j++){
			hiddenWeights[i][j] = init_weight(); 
		}
	}
	for(int i = 0; i < numHiddenNodes; i++){
		hiddenLayerBias[i] = init_weight(); 
		for(int j = 0; j < numOutputs; j++){
			outputWeights[i][j] = init_weight(); 
		}
	}

	for(int i = 0; i < numOutputs; i++){
		outputLayerBias[i] = init_weight(); 
	}

	
	int trainingSetOrder[] = {0, 1, 2, 3};
	// Iterate over a number of epochs and foreach epoch pick one pair of inputs and its expected output 
	const double lr = 15.2f; 
	int epochs = 1000000; 
	for(int n = 0; n < epochs; n++){
		// As per SGD, shuffle hte order of the training set 	
		
		shuffle(trainingSetOrder, numTrainingSets);

		// Cycle through each of hte training set elements
		for(int x = 0; x < numTrainingSets; x++){

			//
			// FORWARD FEED IS HERE
			// Calculate the output of the network given the current weights according ot this formula sigmoid(hiddenLayerBias + Sum(trainingInput_k * hiddenWeight))
			int i = trainingSetOrder[x]; 


			// Compute Hidden Layer Activation
			for(int j = 0; j < numHiddenNodes; j++){
				double activation = hiddenLayerBias[j]; 
				for(int k = 0; k < numInputs; k++){
					activation += training_inputs[i][k] * hiddenWeights[k][j]; 
				}
				hiddenLayer[j] = sigmoid(activation); 
			}

			
			// Compute output layer activation
			for(int j = 0; j < numOutputs; j++){
				double activation = outputLayerBias[j]; 
				for(int k = 0; k < numHiddenNodes; k++){
					activation += hiddenLayer[k] * outputWeights[k][j]; 
				}
				outputLayer[j] = sigmoid(activation);

			}



		
			
			// Backpropogation begins here

			// Calculate Mean Squared Error In output Weights
			double deltaOutput[numOutputs];
			for(int j = 0; j < numOutputs; j++){
				double dError = (training_outputs[i][j] - outputLayer[j]); 
				deltaOutput[j] = dError * dSigmoid(outputLayer[j]); 
			}
			

			// Calcuate Mean Squared Error in Hidden Weights
			double deltaHidden[numHiddenNodes]; 
			for(int j = 0; j < numHiddenNodes; j++){
				double dError = 0.0f; 
				for(int k = 0; k < numOutputs; k++){
					dError += deltaOutput[k] * outputWeights[j][k]; 
				}
			deltaHidden[j] = dError * dSigmoid(hiddenLayer[j]); 


			}

			// Apply change in output weights
			for(int j = 0; j < numOutputs; j++){
				outputLayerBias[j] += deltaOutput[j] * lr;	
				for(int k = 0; k < numHiddenNodes; k++){
					outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * lr; 
				}
			}
		
			// Apply change in hidden weights
			for(int j = 0; j < numHiddenNodes; j++){
				hiddenLayerBias[j] += deltaHidden[j] * lr; 
				for(int k = 0; k < numInputs; k++){
					hiddenWeights[k][j] += training_inputs[i][k] * deltaHidden[j] * lr; 
				
				}
			}

			// Backpropogation ends here

		}
	}


	// PREDICT FUNCTION 
	double test_input[][2] = { {atof(argv[1]), atof(argv[2])} }; 

	// Compute hidden outer layer activation
	for(int j = 0; j < numHiddenNodes; j++){
		double activation = hiddenLayerBias[j];
		for(int k = 0; k < numInputs; k++){
			activation += test_input[0][k]*hiddenWeights[k][j]; 
		}
		hiddenLayer[j] = sigmoid(activation);
	}


	// Compute output layer activation
	for(int j = 0; j < numOutputs; j++){
		double activation = outputLayerBias[j]; 
		for(int k = 0; k < numHiddenNodes; k++){
			activation += hiddenLayer[k] * outputWeights[k][j]; 
		}
		outputLayer[j] = sigmoid(activation);

	}

	printf("%f", outputLayer[0]);	
			
	
}

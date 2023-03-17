__global__ void forwardFeed(double* inputLayer, double* hiddenWeights, double* hiddenLayer, double* outputLayer, double* outputWeights, double* outputLayerBias, double* hiddenLayerBias, int numHiddenNodes, int numInputs, int numOutputs, int trainingSetIndex);


__global__ void backpropogate(double* trainingInputs, double* hiddenLayer, double* hiddenWeights, double* outputLayer, double* outputWeights, double* trainingOutputs, double* hiddenLayerBias, double* outputLayerBias, int numHiddenNodes, int numInputs, int numOutputs, int trainingSetIndex, double lr);

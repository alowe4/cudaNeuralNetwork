__global__ void forwardFeed(float* inputLayer, float* hiddenWeights, float* hiddenLayer, float* outputLayer, float* outputWeights, float* outputLayerBias, float* hiddenLayerBias, int numHiddenNodes, int numInputs, int numOutputs, int trainingSetIndex);


__global__ void backpropogate(float* trainingInputs, float* hiddenLayer, float* hiddenWeights, float* outputLayer, float* outputWeights, float* trainingOutputs, float* hiddenLayerBias, float* outputLayerBias, int numHiddenNodes, int numInputs, int numOutputs, int trainingSetIndex, float lr);

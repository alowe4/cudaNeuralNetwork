__global__ void sigmoidActivationForward(float* src, float* dst, int Z_x_dim, int Z_y_dim);

__global__ void forwardFeedP1(float* inputLayer, float* hiddenWeights, float* hiddenLayer, float* outputLayer, float* outputWeights, float* outputLayerBias, float* hiddenLayerBias, int numHiddenNodes, int numInputs, int numOutputs, int trainingSetIndex,float* HiddenLayerAct);

__global__ void forwardFeedP2(float* inputLayer, float* hiddenWeights, float* hiddenLayer, float* outputLayer, float* outputWeights, float* outputLayerBias, float* hiddenLayerBias, int numHiddenNodes, int numInputs, int numOutputs, int trainingSetIndex,float* OutputLayerAct);



__global__ void backpropogate(float* trainingInputs, float* hiddenLayer, float* hiddenWeights, float* outputLayer, float* outputWeights, float* trainingOutputs, float* hiddenLayerBias, float* outputLayerBias, int numHiddenNodes, int numInputs, int numOutputs, int trainingSetIndex, float lr);

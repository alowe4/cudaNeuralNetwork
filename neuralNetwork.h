__global__ void sigmoidActivationForward(double* src, double* dst, int Z_x_dim, int Z_y_dim);

__global__ void forwardFeedP1(double* inputLayer, double* hiddenWeights, double* hiddenLayer, double* outputLayer, double* outputWeights, double* outputLayerBias, double* hiddenLayerBias, int numHiddenNodes, int numInputs, int numOutputs, int trainingSetIndex,double* HiddenLayerAct);

__global__ void forwardFeedP2(double* inputLayer, double* hiddenWeights, double* hiddenLayer, double* outputLayer, double* outputWeights, double* outputLayerBias, double* hiddenLayerBias, int numHiddenNodes, int numInputs, int numOutputs, int trainingSetIndex,double* OutputLayerAct);

__global__ void backpropogate(double* trainingInputs, double* hiddenLayer, double* hiddenWeights, double* outputLayer, double* outputWeights, double* trainingOutputs, double* hiddenLayerBias, double* outputLayerBias, int numHiddenNodes, int numInputs, int numOutputs, int trainingSetIndex, double lr);

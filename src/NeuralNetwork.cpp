#include <NeuralNetwork.hpp>
#include <System/GPUMath.hpp>

#include <SFML/System/Clock.hpp>
#include <SFML/System/Time.hpp>

#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <chrono>


static std::vector<float> ParseFloatVecFromString(const std::string& _str) {
  std::vector<float> vec;

  std::string buffer;
  for(char c : _str) {
    if(std::isdigit(c) || c == '-' || c == '.') {
      buffer.push_back(c);
      continue;
    }

    if(buffer.size() == 0) continue;

    vec.push_back(std::stof(buffer));
    buffer.clear();
  }

  if(buffer.size() != 0) vec.push_back(std::stof(buffer));

  return vec;
}
static std::ostream& operator<<(std::ostream& os, const std::vector<float>& floatVec) {
  const size_t vecSize = floatVec.size();

  if(vecSize == 0) return (os << "{}");

  os << "{";
  for(int i = 0; i < vecSize - 1; i++) os << floatVec[i] << ", ";
  return (os << floatVec.back() << "}");
}


float WeightLayer::ActivationFunction(float _x) {
  return 1.0f / (1.0f + std::exp(-_x));
}
float WeightLayer::ActivationDerivative(float _x) {
  float activation = ActivationFunction(_x);
  return activation * (1 - activation);
}

float WeightLayer::NodeCost(float outputActivation, float expectedActivation) {
  float error = outputActivation - expectedActivation;
  return error * error;
}
float WeightLayer::NodeCostDerivative(float outputActivation, float expectedActivation) {
  return 2 * (outputActivation - expectedActivation);
} 


std::vector<float> WeightLayer::CalculateOutputs(const std::vector<float>& _inputs) {
  std::vector<float> return_activations(outputCount);

  for(int nodeOut = 0; nodeOut < outputCount; nodeOut++) {
    float weightedInput = biases[nodeOut];
    for(int nodeIn = 0; nodeIn < inputCount; nodeIn++) {
      weightedInput += _inputs[nodeIn] * weights(nodeIn, nodeOut);
    }

    weightedInputs[nodeOut] = weightedInput;
    return_activations[nodeOut] = ActivationFunction(weightedInput);
  }

  inputs = _inputs;
  activations = return_activations;

  return return_activations;
}
std::vector<float> WeightLayer::CalculateOutputLayerNodeValues(const std::vector<float>& expectedOutputs) const {
  std::vector<float> nodeValues(expectedOutputs.size());

  for(int i = 0; i < nodeValues.size(); i++) {
    float costDerivative = NodeCostDerivative(activations[i], expectedOutputs[i]);
    float activationDerivative = ActivationDerivative(weightedInputs[i]);
    nodeValues[i] = activationDerivative * costDerivative;
  }

  return nodeValues;
}
std::vector<float> WeightLayer::CalculateHiddenLayerNodeValues(const WeightLayer& oldLayer, const std::vector<float>& oldNodeValues) const {
  std::vector<float> newNodeValues(outputCount);

  for(int newNodeIndex = 0; newNodeIndex < newNodeValues.size(); newNodeIndex++) {
    float newNodeValue = 0;
    for(int oldNodeIndex = 0; oldNodeIndex < oldNodeValues.size(); oldNodeIndex++) {
      float weightedInputDerivative = oldLayer.weights(newNodeIndex, oldNodeIndex);
      newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex];
    }

    newNodeValue *= ActivationDerivative(weightedInputs[newNodeIndex]);
    newNodeValues[newNodeIndex] = newNodeValue;
  }

  return newNodeValues;
}

void WeightLayer::ApplyGradients(float learnRate) {
  for(int nodeOut = 0; nodeOut < outputCount; nodeOut++) {
    biases[nodeOut] -= costGradientB[nodeOut] * learnRate;
    for(int nodeIn = 0; nodeIn < inputCount; nodeIn++) {
      weights(nodeIn, nodeOut) -= costGradientW(nodeIn, nodeOut) * learnRate;
    }
  }

  // weights = GPUMath::ASubtractBMulScalar(weights, costGradientW, learnRate);
}
void WeightLayer::UpdateGradients(const std::vector<float>& nodeValues) {
  for(int nodeOut = 0; nodeOut < outputCount; nodeOut++) {
    for(int nodeIn = 0; nodeIn < inputCount; nodeIn++) {
      float derivativeCostWrtWeight = inputs[nodeIn] * nodeValues[nodeOut];
      costGradientW(nodeIn, nodeOut) += derivativeCostWrtWeight;
    }

    float derivateCostWrtBias = nodeValues[nodeOut];
    costGradientB[nodeOut] += derivateCostWrtBias;
  }
}
void WeightLayer::ClearGradients() {
  for(int nodeOut = 0; nodeOut < outputCount; nodeOut++) {
    for(int nodeIn = 0; nodeIn < inputCount; nodeIn++) {
      costGradientW(nodeIn, nodeOut) = 0;
    }
    costGradientB[nodeOut] = 0;
  }
}

void WeightLayer::InitializeRandomWeights() {
  std::mt19937 gen(std::chrono::high_resolution_clock().now().time_since_epoch().count());
  std::uniform_real_distribution<float> rand(-1.0f, 1.0f);

  for(int i = 0; i < inputCount; i++) {
    for(int j = 0; j < outputCount; j++) {
      weights(i, j) = rand(gen);
    }
  }
}
void WeightLayer::Resize(unsigned int _in, unsigned int _out) {
  inputCount = _in;
  outputCount = _out;

  weights.Resize(_in, _out);
  costGradientW.Resize(_in, _out);

  biases.resize(outputCount);
  costGradientB.resize(outputCount);

  weightedInputs.resize(outputCount);
  activations.resize(outputCount);

  inputs.resize(inputCount);
}



NeuralNetwork::NeuralNetwork(const std::vector<unsigned int>& layerSizes) {
  layers.resize(layerSizes.size() - 1);
  for(int i = 0; i < layers.size(); i++) {
    layers[i] = WeightLayer(layerSizes[i], layerSizes[i + 1]);
    layers[i].InitializeRandomWeights();
  }
}

static std::string GetQuotedLabel(const std::string& _str) {
  unsigned int _strSize = _str.size();

  for(int i = 0; i < _strSize; i++) {
    if(_str[i] != '\"') continue;

    for(int j = i + 1; j < _strSize; j++) {
      if(_str[j] != '\"') continue;

      return _str.substr(i + 1, j - i - 1);
    }
  }

  return "";
}
static std::ostream& operator<<(std::ostream& os, const WeightLayer& layer) {
  os << "\t{\n";

  layer.weights.Print(os, "\t\t");
  os << '\n';

  os << "\t\t\"biases\": " << layer.biases << ",\n";
  os << "\t\t\"weighted inputs\": " << layer.weightedInputs << ",\n";
  os << "\t\t\"activations\": " << layer.activations << '\n';
  os << "\t}";

  return os;
}
void NeuralNetwork::SaveToFile(const std::string& _path) const {
  std::ofstream file(_path);

  {
    file << "\"layers\": {\n";

    for(int i = 0; i < static_cast<int>(layers.size()) - 1; i++) file << layers[i] << ",\n";
    file << layers.back() << '\n';

    file << "}";
  }

  file.close();
}
bool NeuralNetwork::LoadFromFile(const std::string& _path) {
  std::ifstream file(_path);

  if(file.is_open()) {
    std::vector<std::string> lines;

    {
      std::string str;
      while(std::getline(file, str)) lines.push_back(str);

      file.close();
    }

    layers.clear();

    WeightLayer currentLayer = WeightLayer(0,0);
    for(int i = 0; i < lines.size(); i++) {
      if(lines[i].substr(0, 4) == "\t\t\t{") {
        const std::vector<float> floatVec = ParseFloatVecFromString(lines[i]);
        currentLayer.weights.Resize(floatVec.size(), currentLayer.weights.height);
        currentLayer.weights.PushBack(floatVec);

      } else if(lines[i].substr(0, 2) == "\t}") {
        currentLayer.Resize(currentLayer.weights.width, currentLayer.weights.height);
        layers.push_back(currentLayer);

        currentLayer = WeightLayer(0,0);

      } else {
        const std::string label = GetQuotedLabel(lines[i]);

            if(label == "biases") currentLayer.biases = ParseFloatVecFromString(lines[i]);
        else if(label == "weighted inputs") currentLayer.weightedInputs = ParseFloatVecFromString(lines[i]);
        else if(label == "activations") currentLayer.activations = ParseFloatVecFromString(lines[i]);
      }
    }

    return true;
  }

  return false;
}

std::vector<float> NeuralNetwork::CalculateOutputs(std::vector<float> inputs) {
  for(WeightLayer& layer : layers) inputs = layer.CalculateOutputs(inputs);
  return inputs;
}

void NeuralNetwork::ApplyAllGradients(float learnRate) {
  for(WeightLayer& layer : layers) layer.ApplyGradients(learnRate);
}
void NeuralNetwork::UpdateAllGradients(const DataPoint& dataPoint) {
  CalculateOutputs(dataPoint.inputs);

  WeightLayer& outputLayer = layers.back();
  std::vector<float> nodeValues = outputLayer.CalculateOutputLayerNodeValues(dataPoint.expectedOutputs);
  outputLayer.UpdateGradients(nodeValues);

  for(int hiddenLayerIndex = layers.size() - 2; hiddenLayerIndex >= 0; hiddenLayerIndex--) {
    WeightLayer& hiddenLayer = layers[hiddenLayerIndex];
    nodeValues = hiddenLayer.CalculateHiddenLayerNodeValues(layers[hiddenLayerIndex + 1], nodeValues);
    hiddenLayer.UpdateGradients(nodeValues);
  }
}
void NeuralNetwork::ClearAllGradients() {
  for(WeightLayer& layer : layers) layer.ClearGradients();
}

void NeuralNetwork::LoadNextTrainingBatch() {
  if(trainingBatch.size() != trainingBatchSize) trainingBatch.resize(trainingBatchSize);

  static unsigned int batchStartOffset = 0;
  const int totalDataSize = trainingData.size();

  if(batchStartOffset + trainingBatchSize <= totalDataSize) {
    // FAST PATH: The entire batch fits sequentially in one clean block
    std::copy(trainingData.begin() + batchStartOffset, trainingData.begin() + batchStartOffset + trainingBatchSize, trainingBatch.begin());

    batchStartOffset += trainingBatchSize;
    if(batchStartOffset >= totalDataSize) batchStartOffset = 0;
  } 
  else {
    // SPLIT PATH: The batch wraps around the end of the array. Copy in 2 fast chunks.
    int firstChunkSize = totalDataSize - batchStartOffset;
    int secondChunkSize = trainingBatchSize - firstChunkSize;

    // Copy chunk 1: From current offset up to the very end of the array
    std::copy(trainingData.begin() + batchStartOffset, trainingData.end(), trainingBatch.begin());

    // Copy chunk 2: Wrap around and copy the remainder from the beginning of the array
    std::copy(trainingData.begin(), trainingData.begin() + secondChunkSize, trainingBatch.begin() + firstChunkSize);

    // Update the offset to point cleanly into the start of the next iteration segment
    batchStartOffset = secondChunkSize;
  }
}
float NeuralNetwork::Learn() {
  LoadNextTrainingBatch();
  
  float learnTime; sf::Clock timeClock;
  for(const DataPoint& dataPoint : trainingBatch) UpdateAllGradients(dataPoint);
  learnTime = timeClock.restart().asSeconds();

  ApplyAllGradients(learnRate / static_cast<float>(trainingBatchSize));
  ClearAllGradients();

  prevCost = Cost(trainingBatch);
  trainingIterations++;

  return learnTime;
}

float NeuralNetwork::Cost(DataPoint dataPoint) {
  std::vector<float> outputs = CalculateOutputs(dataPoint.inputs);
  WeightLayer outputLayer = layers.back();
  float cost = 0;

  for(int nodeOut = 0; nodeOut < outputs.size(); nodeOut++) {
    cost += outputLayer.NodeCost(outputs[nodeOut], dataPoint.expectedOutputs[nodeOut]);
  }

  return cost;
}
float NeuralNetwork::Cost(const std::vector<DataPoint>& data) {
  float totalCost = 0;

  for(const DataPoint& dataPoint : data) totalCost += Cost(dataPoint);

  return totalCost / static_cast<float>(data.size());
}

int NeuralNetwork::Classify(const std::vector<float>& inputs) {
  const std::vector<float> outputs = CalculateOutputs(inputs);
  return IndexOfMaxValue(outputs);
}


int NeuralNetwork::IndexOfMaxValue(const std::vector<float>& outputs) {
  float max = -MAX_FLOAT;
  int index = -1;
  for(int i = 0; i < outputs.size(); i++) {
    if(outputs[i] > max) {
      max = outputs[i];
      index = i;
    }
  }

  return index;
}
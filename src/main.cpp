#include <iostream>
#include <sstream>

#include <SFML/System/Clock.hpp>
#include <SFML/System/Time.hpp>

#include <NeuralNet/Tensor.hpp>
#include <NeuralNet/FNN.hpp>
#include <NeuralNet/DataPoint.hpp>
using namespace NeuralNetwork;


static std::string ConsoleInput(const std::string &_msg) {
  std::cout << _msg << ": ";
  std::string str;
  std::getline(std::cin, str);
  return str;
}
static void log(int32_t _ms, float _loss, int _iteration) {
  std::stringstream ss;
  ss << _ms << "ms | " <<
  "Loss: " << _loss << " | " <<
  "Iterations: " << _iteration << '\n';
  std::cout << ss.str();  
}
int main(int argc, char* argv[]) {
  FNN fnn;
  fnn.AddLayer(new Dense<TensorInit::Random, TensorInit::Constant>(2,2, {-.5f, .5f, 0}));

  std::vector<DataPoint> trainingData;
  trainingData.push_back(DataPoint({0,0}, {0,0}));
  trainingData.push_back(DataPoint({0,1}, {0,1}));
  trainingData.push_back(DataPoint({1,0}, {1,0}));
  trainingData.push_back(DataPoint({1,1}, {1,1}));
  
  for(const DataPoint &dataPoint : trainingData) {    
    std::cout << "Input: " << dataPoint.input.ToString() << '\n' <<
                 "Output: " << fnn.Forward(Tensor<float>::Copy(dataPoint.input)).ToString() << '\n';
    std::cout << "Loss: " << fnn.MSELoss(dataPoint) << "\n\n";
  }

  uint64_t trainingIterations = std::stoull(ConsoleInput("Enter Training Iterrations"));
  uint64_t iterationInterval = trainingIterations / 10;
  
  std::cout << "training for " << trainingIterations << " iterations...\n";
  const float nudge = .0001f;
  const float learnRate = .01f;
  sf::Clock timeClock;
  for(uint64_t i = 0; i < trainingIterations; i++) {
    if((i % iterationInterval) == 0) {
      log(timeClock.restart().asMilliseconds(), fnn.MSELoss(trainingData[0]), i);
    }
      
    for(const DataPoint &dataPoint : trainingData) {
      for(Layer *layer : fnn.m_layers) {
        std::vector<Tensor<float>*> trainables = layer->RetreiveTrainables();
        for(Tensor<float> *trainable : trainables) {
          const uint32_t neuronCount = trainable->m_width * trainable->m_height;
          for(int i = 0; i < neuronCount; i++) {
            const float preLoss = fnn.MSELoss(dataPoint);
            *(trainable->m_data + i) += nudge;
            const float postLoss = fnn.MSELoss(dataPoint);
            *(trainable->m_data + i) -= nudge;

            float slope = (postLoss - preLoss) / nudge;
            *(trainable->m_data + i) -= learnRate * slope;
          }
        }
      }
    }
  }
  log(timeClock.restart().asMilliseconds(), fnn.MSELoss(trainingData[0]), trainingIterations);
  std::cout << "training complete.\n";
  
  std::cin.get();
  system("cls");

  for(const DataPoint &dataPoint : trainingData) {
    std::cout << "Total Loss: " << fnn.MSELoss(dataPoint) << '\n';
  }

  const float inputA = std::stof(ConsoleInput("Input A"));
  const float inputB = std::stof(ConsoleInput("Input B"));
  system("cls");

  Tensor<float> input(1,2);
  input.set({inputA, inputB});
  std::cout << "Input:\n" << input.ToString() << "\n\n" <<
               "Output:\n" << fnn.Forward(Tensor<float>::Copy(input)).ToString();

  if constexpr(false) {
    NeuralNetwork::Tensor<float> tensorA(2,2);
    tensorA.set({1,2,3,4});

    std::cout << tensorA.ToString() << '\n';
  }
  if constexpr(false) {
    NeuralNetwork::Tensor<float> tensorA(3,2);
    tensorA.set({1,2,3,4,5,6});

    NeuralNetwork::Tensor<float> tensorB(2,3);
    tensorB.set({7,8,9,1,2,3});

    std::cout << tensorB.dot(tensorA).ToString() << '\n';
  }

  std::cin.get();
  return 0;
}
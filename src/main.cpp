#include <windows.h>
#include <filesystem>
#include <iostream>
#include <fstream>

#include <NeuralNetwork.hpp>


static std::filesystem::path PATH;


int main(int argc, char* argv[]) {
  char buffer[MAX_PATH]; GetModuleFileNameA(NULL, buffer, MAX_PATH);
  PATH = std::filesystem::path(buffer).parent_path().parent_path().string();


  const std::vector<DataPoint> trainingData = {
    DataPoint{
      Matrix(MatrixData{{0,1}}),
      Matrix(MatrixData{{0,1}})
    },
    DataPoint{
      Matrix(MatrixData{{1,0}}),
      Matrix(MatrixData{{1,0}})
    },
    DataPoint{
      Matrix(MatrixData{{0,0}}),
      Matrix(MatrixData{{0,0}})
    },
  };

  FNN neuralNetwork = FNN({2,2});
  neuralNetwork.SaveToFile(PATH/"assets/neural_network_pre.txt", trainingData);

  { //Pre Training Results
    for(const DataPoint dataPoint : trainingData) {
      std::cout << "Input: {" << join_string(dataPoint.inputs.m_data[0], ", ")  << "}\n" <<
                  "Output: " << std::string(neuralNetwork.CalculateOutputs(dataPoint)) << '\n' <<
                  "Loss: " << neuralNetwork.CalculateLoss(dataPoint) << "\n\n";
    }

    std::cin.get();
    system("cls");
  } //Pre Training Results

  { //Training
    std::cout << "Enter Iteration Count: ";
    std::string str;
    std::getline(std::cin, str);
    system("cls");

    const int iterationCount = std::stoi(str);

    std::cout << "training for " << iterationCount << " iterations...\n";

    for(int iteration = 0; iteration < iterationCount; iteration++) {
      for(const DataPoint &_dataPoint : trainingData) {
        static constexpr NNValueType learnRate = .01;
        static constexpr NNValueType nudge = .0001;


        for(int i = 0; i < neuralNetwork.m_layers.back().m_outputCount; i++) {
          for(int j = 0; j < neuralNetwork.m_layers.back().m_inputCount; j++) {
            const NNValueType preLoss = neuralNetwork.CalculateLoss(_dataPoint);

            NNValueType &weightRef = neuralNetwork.m_layers.back().m_weights[j][i];

            weightRef += nudge;
            const NNValueType postLoss = neuralNetwork.CalculateLoss(_dataPoint);
            weightRef -= nudge;

            weightRef -= learnRate * ((postLoss - preLoss) / nudge);
          }
        }
        
        for(int i = 0; i < neuralNetwork.m_layers.back().m_outputCount; i++) {
          const NNValueType preLoss = neuralNetwork.CalculateLoss(_dataPoint);

          NNValueType &biasRef = neuralNetwork.m_layers.back().m_biases[0][i];

          biasRef += nudge;
          const NNValueType postLoss = neuralNetwork.CalculateLoss(_dataPoint);
          biasRef -= nudge;

          biasRef -= learnRate * ((postLoss - preLoss) / nudge);
        }
      }
    }

    std::cout << "training finished.\n";
    std::cin.get();
    system("cls");
  } //Training

  { //Post Training Results
    for(const DataPoint dataPoint : trainingData) {
      std::cout << "Input: {" << join_string(dataPoint.inputs.m_data[0], ", ")  << "}\n" <<
                  "Output: " << std::string(neuralNetwork.CalculateOutputs(dataPoint)) << '\n' <<
                  "Loss: " << neuralNetwork.CalculateLoss(dataPoint) << "\n\n";
    }
  } //Post Training Results
  
  neuralNetwork.SaveToFile(PATH/"assets/neural_network_post.txt", trainingData);

  std::cin.get();
  return 0;
}
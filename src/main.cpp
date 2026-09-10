#include <windows.h>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <functional>

#include <NeuralNetwork.hpp>
#define CPU_MODE
// #define OLD_TRAINING_METHOD1

#include <SFML/Graphics.hpp>


static std::filesystem::path PATH;


int main(int argc, char* argv[]) {
  char buffer[MAX_PATH]; GetModuleFileNameA(NULL, buffer, MAX_PATH);
  PATH = std::filesystem::path(buffer).parent_path().parent_path().string();


  std::vector<float> accuracy;
  std::vector<float> loss;
  float lossMax = 0;

  { //Neural Network
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
        Matrix(MatrixData{{1,1}}),
        Matrix(MatrixData{{1,1}})
      },
      DataPoint{
        Matrix(MatrixData{{0,0}}),
        Matrix(MatrixData{{0,0}})
      },
    };

    std::function<void(FNN&)> PrintTrainingResults = [&](FNN &network) {
      for(const DataPoint dataPoint : trainingData) {
        std::cout << "Input: {" << join_string(dataPoint.inputs.m_data[0], ", ")  << "}\n" <<
                    "Output: " << std::string(network.CalculateOutputs(dataPoint.inputs)) << '\n' <<
                    "Loss: " << network.CalculateLoss(dataPoint) << "\n\n";
      }
    };

    FNN neuralNetwork = FNN({2,2});
    neuralNetwork.m_learnRate = .01;
    neuralNetwork.m_loss = LossType::CategoricalCrossEntropy;

    neuralNetwork.SaveToFile(PATH/"assets/neural_network_pre.txt", trainingData);

    { //Pre Training Results
      PrintTrainingResults(neuralNetwork);
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

      loss.resize(iterationCount);
      for(int iteration = 0; iteration < iterationCount; iteration++) {
      #ifdef OLD_TRAINING_METHOD
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
      #else
        neuralNetwork.Learn(trainingData);
        loss[iteration] = neuralNetwork.CalculateLoss(trainingData);
        lossMax = std::max<float>(lossMax, loss[iteration]);
      #endif
      }

      std::cout << "training finished.\n";
      std::cin.get();
      system("cls");
    } //Training
    { //Post Training Results
      PrintTrainingResults(neuralNetwork);
      std::cin.get();
    } //Post Training Results
    
    neuralNetwork.SaveToFile(PATH/"assets/neural_network_post.txt", trainingData);
  }

  sf::RenderWindow window = sf::RenderWindow(sf::VideoMode{{800,600}}, "Neural Network Graph");
  window.setVerticalSyncEnabled(true);

  while(window.isOpen()) {
    while(const auto &eventOpt = window.pollEvent()) {
      const auto &event = *eventOpt;

      if(event.is<sf::Event::Closed>()) window.close();
      else if(const auto *resized = event.getIf<sf::Event::Resized>()) {
        window.setView(sf::View{sf::FloatRect{{0,0}, sf::Vector2f{resized->size}}});
      }
      else if(const auto *keyPressed = event.getIf<sf::Event::KeyPressed>()) {
        if(keyPressed->code == sf::Keyboard::Key::Escape) window.close();
      }
    }


    window.clear(sf::Color::Black);

    const sf::FloatRect area = sf::FloatRect({15,15}, sf::Vector2f{window.getSize()} - sf::Vector2f{30,30});

    std::function<sf::Vector2f(int, float, bool)> NormalizePoint = [&](int _index, float _val, bool flipped) {
      return sf::Vector2f{
        _index / static_cast<float>(loss.size()),
        static_cast<int>(flipped) * (1 - 2 * (_val / lossMax)) + (_val / lossMax)
      };
    };

    std::vector<sf::Vector2f> points(loss.size());
    for(int i = 0; i < loss.size(); i++) {
      points[i] = area.position + NormalizePoint(i, loss[i], true).componentWiseMul(area.size);
    }

    sf::VertexArray shape(sf::PrimitiveType::Lines);
    for(int i = 1; i < loss.size(); i++) {
      shape.append(sf::Vertex{points[i-1], sf::Color::Blue});
      shape.append(sf::Vertex{points[i], sf::Color::Blue});
    }
    window.draw(shape);

    window.display();
  }

  return 0;
}
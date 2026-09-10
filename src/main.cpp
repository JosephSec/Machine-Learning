#include <windows.h>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <functional>

#include <NeuralNetwork.hpp>
#define CPU_MODE

#include <SFML/Graphics.hpp>

#include <random>
#include <chrono>
static int RandomInt(int _min, int _max) {
  std::mt19937 gen(std::chrono::high_resolution_clock().now().time_since_epoch().count());
  std::uniform_int_distribution<int> rand(_min, _max);
  return rand(gen);
}

static void PrintTrainingSteps(int32_t _ms, NNValueType _loss, int _iteration) {
  std::stringstream ss;
  ss << _ms << "ms | " <<
  // "Average Loss: " << (intervalTotalLoss / static_cast<double>(iterationInterval)) << " | " <<
  "Loss: " << _loss << " | " <<
  "Iterations: " << _iteration << '\n';
  std::cout << ss.str();  
}
static void PrintTrainingResults(FNN &_network, const std::vector<DataPoint> &_trainingData) {
  for(const DataPoint dataPoint : _trainingData) {
    std::cout << "Input: {" << join_string(dataPoint.inputs.m_data[0])  << "}\n" <<
    "Output: {" << join_string(_network.CalculateOutputs(dataPoint.inputs).m_data[0]) << "}\n" <<
    "Loss: " << _network.CalculateLoss(dataPoint) << "\n\n";
  }
}

static NNValueType GetAccuracy(const DataPoint &_dataPoint, const Matrix &_output) {
  if(_output.m_size.y <= 0) return 0.0;

  int correctNodes = 0;

  for(int i = 0; i < _output.m_size.y; i++) {
    // Convert continuous prediction to binary 0 or 1
    int predictedState = (_output[0][i] >= 0.5) ? 1 : 0;
    int expectedState  = (_dataPoint.expectedOutputs[0][i] >= 0.5) ? 1 : 0;

    if(predictedState == expectedState) correctNodes++;
  }

  // Returns a clean fraction (e.g., 0.75 if 3 out of 4 nodes were guessed correctly)
  return static_cast<NNValueType>(correctNodes) / static_cast<NNValueType>(_output.m_size.y);
}


static constexpr uint64_t MAX_GRAPH_RESOLUTION = 1000;
static sf::VertexArray GetGraphShape(const sf::FloatRect &_area, const std::vector<float> &_data, const sf::Color &_color) {
  float dataMax = 0;
  for(const float &_val : _data) dataMax = std::max<float>(_val, dataMax);

  const uint32_t pointCount = _data.size();
  const uint32_t graphPointCount = std::min<uint32_t>(MAX_GRAPH_RESOLUTION, pointCount);
  const uint32_t pointInterval = pointCount / graphPointCount;

  std::vector<sf::Vector2f> points(graphPointCount);
  for(int i = 0; i < graphPointCount; i++) {
    const uint32_t dataIndex = i * pointInterval;

    const sf::Vector2f normalized = sf::Vector2f{
      i / static_cast<float>(graphPointCount),
      1 - (_data[dataIndex] / dataMax)
      // static_cast<int>(_flip) * (1 - 2 * (_data[dataIndex] / dataMax)) + (_data[dataIndex] / dataMax)
    };

    points[i] = _area.position + normalized.componentWiseMul(_area.size);
  }

  sf::VertexArray shape(sf::PrimitiveType::Lines);
  for(int i = 1; i < points.size(); i++) {
    shape.append(sf::Vertex{points[i-1], _color});
    shape.append(sf::Vertex{points[i], _color});
  }
  return shape;
}


static std::filesystem::path PATH;

int main(int argc, char* argv[]) {
  char buffer[MAX_PATH]; GetModuleFileNameA(NULL, buffer, MAX_PATH);
  PATH = std::filesystem::path(buffer).parent_path().parent_path().string();


  std::vector<NNValueType> accuracyPoints;
  std::vector<NNValueType> lossPoints;
  NNValueType lossPointsMax = 0;

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

    FNN neuralNetwork = FNN({2,4,8,2});
    neuralNetwork.m_learnRate = .01;
    neuralNetwork.m_loss = LossType::MeanSquaredError;

    neuralNetwork.SaveToFile(PATH/"assets/neural_network_pre.txt", trainingData);

    { //Pre Training Results
      PrintTrainingResults(neuralNetwork, trainingData);
      std::cin.get();
      system("cls");
    } //Pre Training Results
    { //Training
      std::cout << "Enter Iteration Count: ";
      std::string str;
      std::getline(std::cin, str);
      system("cls");

      const int iterationCount = std::stoi(str);
      const int iterationInterval = iterationCount / 10;

      std::cout << "training for " << iterationCount << " iterations...\n\n";

      std::mt19937 gen(std::chrono::high_resolution_clock().now().time_since_epoch().count());
      std::uniform_int_distribution<int> rand(0, static_cast<int>(trainingData.size()) - 1);
      const DataPoint &accuracySample = trainingData[rand(gen)];

      accuracyPoints.resize(iterationCount);
      lossPoints.resize(iterationCount);
      sf::Clock timeClock;
      for(int iteration = 0; iteration < iterationCount; iteration++) {
        neuralNetwork.Learn(trainingData);
        const NNValueType loss = neuralNetwork.CalculateLoss(trainingData);

        if((iteration % iterationInterval) == 0) {
          PrintTrainingSteps(timeClock.restart().asMilliseconds(), loss, iteration);
        }

        accuracyPoints[iteration] = GetAccuracy(accuracySample, neuralNetwork.CalculateOutputs(accuracySample.inputs));
        lossPoints[iteration] = loss;
        lossPointsMax = std::max<NNValueType>(lossPointsMax, loss);
      }      
      PrintTrainingSteps(timeClock.restart().asMilliseconds(), neuralNetwork.CalculateLoss(trainingData), iterationCount);

      std::cout << "\ntraining finished.\n";
      std::cin.get();
      system("cls");
    } //Training
  
    PrintTrainingResults(neuralNetwork, trainingData);
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
    window.draw(GetGraphShape(area, accuracyPoints, sf::Color(255,125,0)));
    window.draw(GetGraphShape(area, lossPoints, sf::Color::Blue));

    window.display();
  }

  return 0;
}
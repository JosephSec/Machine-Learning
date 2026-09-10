#include <ModelTesting.hpp>
#include <Renderer.hpp>

#include <windows.h>
#include <iostream>
#include <random>
#include <chrono>


FNN ModelTesting::neuralNetwork({2,4,8,2});
std::vector<DataPoint> ModelTesting::trainingData;

GraphLine ModelTesting::lossGraph;
GraphLine ModelTesting::accuracyGraph;


void ModelTesting::init() {
  trainingData = {
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
  
  neuralNetwork.m_learnRate = .01;
  neuralNetwork.m_loss = LossType::MeanSquaredError;

  lossGraph.m_color = sf::Color::Blue;
  accuracyGraph.m_color = sf::Color(255,125,0);
}
void ModelTesting::learn() {
  const int iterationCount = ModelTesting::GetIterationCount("Enter Iteration Count");
  const int iterationInterval = iterationCount / 10;

  accuracyGraph.setSize(iterationCount);
  lossGraph.setSize(iterationCount);

  const DataPoint &accuracySample = ModelTesting::GetRandomDataPoint();


  sf::Clock timeClock;
  std::cout << "training for " << iterationCount << " iterations...\n\n";
  for(int iteration = 0; iteration < iterationCount; iteration++) {
    neuralNetwork.Learn(trainingData);
    const NNValueType loss = neuralNetwork.CalculateLoss(trainingData);

    if((iteration % iterationInterval) == 0) {
      LogTrainingProgress(timeClock.restart().asMilliseconds(), loss, iteration);
    }

    accuracyGraph.setIndex(iteration, GetAccuracy(accuracySample));
    lossGraph.setIndex(iteration, loss);
  }
  LogTrainingProgress(timeClock.restart().asMilliseconds(), neuralNetwork.CalculateLoss(trainingData), iterationCount);

  std::cout << "\ntraining finished.\n";
}
void ModelTesting::draw() {
  const float pad = 25;
  const sf::FloatRect area = sf::FloatRect(
    {15,15},
    sf::Vector2f{Renderer::window.getSize()} - sf::Vector2f{30,30}
  );
  const sf::FloatRect innerArea = sf::FloatRect(
    area.position + sf::Vector2f{1,1} * pad,
    area.size - sf::Vector2f{1,1} * (pad * 2)
  );

  
  // { //Background
  //   const sf::Color a = sf::Color(255,255,255, 200);
  //   const sf::Color b = sf::Color(255,255,255, 100);
  //
  //   sf::VertexArray background(sf::PrimitiveType::Lines);
  //   background.append(sf::Vertex{sf::Vector2f{innerArea.position.x, area.position.y}, a});
  //   background.append(sf::Vertex{sf::Vector2f{innerArea.position.x, innerArea.position.y + innerArea.size.y}, a});
  //   background.append(sf::Vertex{sf::Vector2f{area.position.x, innerArea.position.y}, a});
  //   background.append(sf::Vertex{sf::Vector2f{innerArea.position.x + innerArea.size.x, innerArea.position.y}, a});
  //   window.draw(background);
  // }

  Renderer::window.draw(GetGraphLineShape(innerArea, lossGraph));
  Renderer::window.draw(GetGraphLineShape(innerArea, accuracyGraph));
}

void ModelTesting::LogTrainingProgress(int32_t _ms, NNValueType _loss, int _iteration) {
  std::stringstream ss;
  ss << _ms << "ms | " <<
  "Loss: " << _loss << " | " <<
  "Iterations: " << _iteration << '\n';
  std::cout << ss.str();  
}
void ModelTesting::LogDataSetOutputs() {
  for(const DataPoint dataPoint : trainingData) {
    std::cout << "Input: {" << join_string(dataPoint.inputs.m_data[0])  << "}\n" <<
    "Output: {" << join_string(neuralNetwork.CalculateOutputs(dataPoint.inputs).m_data[0]) << "}\n" <<
    "Loss: " << neuralNetwork.CalculateLoss(dataPoint) << "\n\n";
  }
}

const DataPoint &ModelTesting::GetRandomDataPoint() {
  std::mt19937 gen(std::chrono::high_resolution_clock().now().time_since_epoch().count());
  std::uniform_int_distribution<int> rand(0, static_cast<int>(trainingData.size()) - 1);
  return trainingData[rand(gen)];
}
NNValueType ModelTesting::GetAccuracy(const DataPoint &_dataPoint) {
  const Matrix output = neuralNetwork.CalculateOutputs(_dataPoint.inputs);

  //AI
  if(output.m_size.y <= 0) return 0.0;

  int correctNodes = 0;

  for(int i = 0; i < output.m_size.y; i++) {
    int predictedState = (output[0][i] >= 0.5) ? 1 : 0;
    int expectedState  = (_dataPoint.expectedOutputs[0][i] >= 0.5) ? 1 : 0;

    if(predictedState == expectedState) correctNodes++;
  }

  return static_cast<NNValueType>(correctNodes) / static_cast<NNValueType>(output.m_size.y);
  //AI
}
int ModelTesting::GetIterationCount(const std::string &_msg) {
  std::cout << _msg << ": ";
  std::string str;
  std::getline(std::cin, str);
  
  system("cls");
  return std::stoi(str);
}

sf::VertexArray ModelTesting::GetGraphLineShape(const sf::FloatRect &_area, const GraphLine &_line) {
  const uint32_t pointCount = _line.m_data.size();
  const uint32_t graphPointCount = std::min<uint32_t>(MAX_GRAPH_RESOLUTION, pointCount);
  const uint32_t pointInterval = pointCount / graphPointCount;

  std::vector<sf::Vector2f> points(graphPointCount);
  for(int i = 0; i < graphPointCount; i++) {
    const uint32_t dataIndex = i * pointInterval;

    const sf::Vector2f normalized = sf::Vector2f{
      i / static_cast<float>(graphPointCount),
      1 - (_line.m_data[dataIndex] / _line.m_dataMax)
    };

    points[i] = _area.position + normalized.componentWiseMul(_area.size);
  }

  sf::VertexArray shape(sf::PrimitiveType::Lines);
  for(int i = 1; i < points.size(); i++) {
    shape.append(sf::Vertex{points[i-1], _line.m_color});
    shape.append(sf::Vertex{points[i], _line.m_color});
  }
  return shape;
}
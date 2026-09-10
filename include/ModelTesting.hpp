#pragma once

#include <string>
#include <SFML/Graphics/Rect.hpp>
#include <SFML/Graphics/VertexArray.hpp>

#include <NeuralNetwork.hpp>


static constexpr uint64_t MAX_GRAPH_RESOLUTION = 2000;
struct GraphLine {
public:
  std::vector<NNValueType> m_data;
  sf::Color m_color;
  NNValueType m_dataMax = 0;

  void setSize(uint32_t _size) {
    m_data.resize(_size);
  }
  void setIndex(uint32_t _index, NNValueType _val) {
    m_data[_index] = _val;
    m_dataMax = std::max<NNValueType>(m_dataMax, _val);
  }
};
class ModelTesting {
public:
  static FNN neuralNetwork;
  static std::vector<DataPoint> trainingData;

  static GraphLine lossGraph;
  static GraphLine accuracyGraph;


  static void init();
  static void learn();
  static void draw();

  static void LogTrainingProgress(int32_t _ms, NNValueType _loss, int _iteration);
  static void LogDataSetOutputs();

  static const DataPoint &GetRandomDataPoint();
  static NNValueType GetAccuracy(const DataPoint &_dataPoint);
  static int GetIterationCount(const std::string &_msg);
  
  static sf::VertexArray GetGraphLineShape(const sf::FloatRect &_area, const GraphLine &_line);
};
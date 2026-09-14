#include <iostream>
#include <sstream>
#include <filesystem>
#include <windows.h>

#include <random>
#include <chrono>

#include <SFML/Graphics.hpp>

#include <NeuralNet/Tensor.hpp>
#include <NeuralNet/FNN.hpp>
#include <NeuralNet/DataPoint.hpp>
using namespace NeuralNetwork;


static std::filesystem::path PATH;


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
  char buffer[MAX_PATH];
  GetModuleFileNameA(NULL, buffer, MAX_PATH);
  PATH = std::filesystem::path(buffer).parent_path().parent_path();

  FNN fnn;
  fnn.AddLayer(new Dense<TensorInit::Random, TensorInit::Constant>(4,6, {-.5f, .5f, 0}));
  fnn.AddLayer(new Dense<TensorInit::Random, TensorInit::Constant>(6,2, {-.5f, .5f, 0}));

  std::vector<DataPoint> trainingData;
  std::mt19937 gen(std::chrono::high_resolution_clock().now().time_since_epoch().count());
  std::uniform_real_distribution<float> rand(0,1);
  for(int i = 0; i < 20; i++) {
    const sf::Vector2f from = sf::Vector2f{rand(gen), rand(gen)};
    const sf::Vector2f to = sf::Vector2f{rand(gen), rand(gen)};
    const sf::Vector2f dir = sf::Vector2f{to - from};

    trainingData.push_back(DataPoint({from.x,from.y, to.x,to.y}, {dir.x,dir.y}));
  }

  { //Training
    uint64_t trainingIterations = std::stoull(ConsoleInput("Enter Training Iterrations"));
    uint64_t iterationInterval = trainingIterations / 10;
    
    std::cout << "training for " << trainingIterations << " iterations...\n";
    const float nudge = .000001f;
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
  } //Training
  fnn.SaveToFile(PATH/"assets/PostTrainModel.txt");
  std::cin.get();
  system("cls");


  sf::RenderWindow window = sf::RenderWindow(sf::VideoMode{{800,600}}, "Neural Network Library");
  sf::Vector2f windowSize = sf::Vector2f{window.getSize()};
  std::vector<sf::Vector2f> modelPositions;
  for(int i = 0; i < 100; i++) {
    modelPositions.push_back(sf::Vector2f{rand(gen), rand(gen)});
  }

  sf::Clock timeClock;

  while(window.isOpen()) {
    while(const auto &eventOpt = window.pollEvent()) {
      const auto &event = *eventOpt;

      if(event.is<sf::Event::Closed>()) window.close();
      else if(const auto *resized = event.getIf<sf::Event::Resized>()) {
        windowSize = sf::Vector2f{resized->size};
        window.setView(sf::View{sf::FloatRect{{0,0}, windowSize}});
      }
    }

    const sf::Vector2f normMouse = sf::Vector2f{sf::Mouse::getPosition(window)}.componentWiseDiv(windowSize);
    { //Update
      const float deltaTime = timeClock.restart().asSeconds();

      Tensor<float> input(1,4);
      input.set({0,0, normMouse.x,normMouse.y});
      for(sf::Vector2f &_modelPos : modelPositions) {
        input.m_data[0] = _modelPos.x;
        input.m_data[1] = _modelPos.y;

        Tensor<float> output = fnn.Forward(Tensor<float>::Copy(input));
        _modelPos += sf::Vector2f{output.m_data[0], output.m_data[1]}.normalized() * deltaTime * 1.0f;
      }
    } //Update

    { //Render
      window.clear(sf::Color::Black);

      { //Points
        sf::CircleShape circle(25, 12);
        circle.setOrigin(sf::Vector2f{1,1} * circle.getRadius());

        circle.setFillColor(sf::Color::Red);
        circle.setPosition(normMouse.componentWiseMul(windowSize));
        window.draw(circle);

        circle.setFillColor(sf::Color::Green);
        for(const sf::Vector2f &_modelPos : modelPositions) {
          circle.setPosition(_modelPos.componentWiseMul(windowSize));
          window.draw(circle);
        }
      } //Points

      { //Network
        const sf::Vector2f start = sf::Vector2f{100,100};
        const sf::Vector2f size = sf::Vector2f{500,300};
        const float radius = 15;
        const float xPad = radius * 5;
        const float yPad = radius * 2.5f;

        std::vector<sf::Vector2f> neurons;
        const uint32_t layerCount = fnn.m_layers.size();
        for(int i = 0; i < layerCount; i++) {
          const Layer *layer = fnn.m_layers[i];
          for(int j = 0; j < layer->m_inCount; j++) {
            neurons.push_back(start + sf::Vector2f{size.x / (layerCount + 2) * (i + 1), size.y / (layer->m_inCount + 1) * (j + 1)});
          }

          if(i == (layerCount - 1)) {
            for(int j = 0; j < layer->m_outCount; j++) {
              neurons.push_back(start + sf::Vector2f{size.x / (layerCount + 2) * (i + 2), size.y / (layer->m_outCount + 1) * (j + 1)});
            }
          }
        }

        { //Weights
          sf::VertexArray line(sf::PrimitiveType::Lines, 2);
          line[0].color = sf::Color(125,125,125,255);
          line[1].color = sf::Color(125,125,125,255);

          int neuronIndex = 0;
          for(int i = 0; i < fnn.m_layers[0]->m_inCount; i++) {
            line[0].position = neurons[neuronIndex + i];
            for(int j = 0; j < fnn.m_layers[0]->m_outCount; j++) {
              line[1].position = neurons[neuronIndex + fnn.m_layers[0]->m_inCount + j];
              window.draw(line);
            }
          }
        } //Weights
        { //Neurons
          sf::CircleShape circle(radius, 12);
          circle.setOrigin(sf::Vector2f{1,1} * circle.getRadius());
          circle.setFillColor(sf::Color(150,150,150, 255));

          for(const sf::Vector2f point : neurons) {
            circle.setPosition(point);
            window.draw(circle);
          }
        } //Neurons
      } //Network

      window.display();
    } //Render

  }


  return 0;
}
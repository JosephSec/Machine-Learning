#include <Renderer.hpp>
#include <System.hpp>
#include <User.hpp>

#include <GL/glew.h>


sf::RenderWindow Renderer::window;
sf::Vector2f Renderer::windowSize;
sf::Vector2f Renderer::windowCenter;

sf::Font Renderer::font;
sf::Text Renderer::TextPrefab(font);


void Renderer::init() {
  window = sf::RenderWindow(sf::VideoMode(sf::Vector2u(900,300)), "Window");
  window.setFramerateLimit(300);
  window.requestFocus();
  update();

  if(!font.openFromFile(System::PATH+"/assets/Roboto.ttf"))
    Debug::error("Roboto.ttf", Debug::Error::OpenFile);
  else System::assets.push_back("Roboto.ttf");
  
  TextPrefab.setFont(font);
  TextPrefab.setString("[EMPTY STRING]");
}
void Renderer::update() {
  windowSize = sf::Vector2f(window.getSize());
  windowCenter = windowSize / 2.0f;
}
static void DrawNeuralNetwork(float _radius) {
  std::vector<unsigned int> layers;
  layers.push_back(System::network.layers[0].inputCount);
  for(const WeightLayer& layer : System::network.layers) layers.push_back(layer.outputCount);

  std::vector<std::vector<sf::Vector2f>> points;

  { //Generate Points
    const float xDif = Renderer::windowSize.x / (layers.size() + 1);
    const float yDif = _radius * 2 + 4;

    for(int i = 0; i < layers.size(); i++) {
      points.push_back({});

      const sf::Vector2 start = sf::Vector2f(xDif * (i + 1), Renderer::windowCenter.y - yDif * (layers[i] / 2.0f));

      for(int j = 0; j < layers[i]; j++) {
        points[i].push_back(start + sf::Vector2f(0, yDif * j));
      }
    }
  }

  { //Drawing
    sf::VertexArray lines(sf::PrimitiveType::Lines);

    for(int i = 0; i < layers.size() - 1; i++) {      
      for(int j = 0; j < points[i].size(); j++) {
        for(int k = 0; k < points[i + 1].size(); k++) {
          lines.append({points[i][j], sf::Color(sf::Color(50,50,50))});
          lines.append({points[i + 1][k], sf::Color(sf::Color(50,50,50))});
        }
      }
    }

    Renderer::draw(lines);


    sf::CircleShape circle(_radius,20);
    circle.setOrigin(sf::Vector2f(_radius, _radius));

    std::vector<float> colorDifs = {
      255.0f / 5.0f * System::network.trainingBatch.back().inputs[0],
      255.0f / 10.0f * System::network.trainingBatch.back().inputs[1]
    };

    for(int i = 0; i < 2; i++) {
      circle.setPosition(points[0][i]);
      circle.setFillColor(sf::Color(colorDifs[i], colorDifs[i], colorDifs[i]));
      Renderer::draw(circle);
    }

    circle.setFillColor(sf::Color(150,150,150));
    for(int i = 1; i < points.size(); i++) {
      const std::vector<float>& weightedInputs = System::network.layers[i - 1].weightedInputs;
      float min = MAX_FLOAT;
      float max = -MAX_FLOAT;

      for(float val : weightedInputs) {
        if(val < min) min = val;
        if(val > max) max = val;
      }

      float range = max - min;
      if(range == 0) range = 1.0f;

      for(int j = 0; j < weightedInputs.size(); j++) {
        circle.setPosition(points[i][j]);

        float normalized = (weightedInputs[j] - min) / range;
        std::uint8_t colorVal = static_cast<std::uint8_t>(normalized * 255.0f);

        circle.setFillColor(sf::Color(colorVal, colorVal, colorVal));
        Renderer::draw(circle);
      }
    }
  }
}
void Renderer::draw() {
  window.clear();

  { //Training Data Visual
    sf::CircleShape circle(15,10);
    circle.setOrigin(sf::Vector2f(1,1) * circle.getRadius());

    for(const DataPoint& dataPoint : System::network.trainingData) {
      circle.setPosition(sf::Vector2f(windowSize.x / 5.0f * dataPoint.inputs[0], windowSize.y / 10.0f * dataPoint.inputs[1]));

      circle.setFillColor(
        (dataPoint.expectedOutputs[0] > dataPoint.expectedOutputs[1]) ? sf::Color(255,0,0, 50) : sf::Color(0,255,0, 50)
      );

      draw(circle);
    }
  }

  {  //Text Objects
    sf::Text text = TextPrefab;
    text.setPosition(sf::Vector2f(15,15));

    std::stringstream ss;
    ss << System::network.prevCost;
    text.setString(ss.str());
    draw(text);

    
    text.setPosition(text.getPosition() + sf::Vector2f(0, text.getLocalBounds().size.y + text.getLocalBounds().position.y));

    ss = std::stringstream();
    ss << System::network.trainingIterations;
    text.setString(ss.str());
    draw(text);

    
    text.setPosition(text.getPosition() + sf::Vector2f(0, text.getLocalBounds().size.y + text.getLocalBounds().position.y));

    ss = std::stringstream();
    ss << System::learnTime;
    text.setString(ss.str());
    draw(text);
  }


  DrawNeuralNetwork(10);

  window.display();
}
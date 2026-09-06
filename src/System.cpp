#include <System.hpp>
#include <User.hpp>
#include <Renderer.hpp>

#include <System/GPUMath.hpp>

#include <SFML/System/Time.hpp>
#include <filesystem>
#include <windows.h>


string System::PATH;

sf::Clock System::timeClock;
float System::deltaTime;

std::vector<string> System::assets;


NeuralNetwork System::network({2, 8, 8, 2});
float System::learnTime;


void System::init() {
  char buffer[MAX_PATH]; GetModuleFileNameA(NULL, buffer, MAX_PATH);
  PATH = std::filesystem::path(buffer).parent_path().parent_path().string();

  Renderer::init();
  GPUMath::Init();


  { //Initialize Neural Network
    LoadTrainingData();
    network.learnRate = std::stof(GetConsoleInput("Enter Learn Rate"));
    network.trainingBatchSize = std::stof(GetConsoleInput("Enter Batch Size"));
    std::cout << "Frame Rate: 300\n";

    if(network.LoadFromFile(PATH+"/assets/NeuralNetworkData.txt") == false) {
      Debug::error("assets/NeuralNetworkData.txt", Debug::Error::OpenFile);
    }
  }

  sf::Clock timeClock;
  for(int i = 0; i < 5000; i++) network.Learn();
  Debug::log("10000 training iterations time", timeClock.restart().asSeconds());

  GenerateReleaseAssets();
}
void System::update() {
  deltaTime = timeClock.restart().asMilliseconds() * .001f;

  User::update();
  Renderer::update();

  learnTime = network.Learn();

  static constexpr float delayTime = .1f;
  static float delay = delayTime;
  delay -= deltaTime;

  if(Renderer::window.hasFocus()) {
    if(User::GetKeyDown(Keyboard::Key::R)) RandomizeTrainingData();
    else if(User::GetKeyDown(Keyboard::Key::Enter)) {
      const unsigned int frameRate = std::stoul(GetConsoleInput("\nEnter Frame Rate"));
      Renderer::window.setFramerateLimit(frameRate);

      system("cls");
      std::stringstream ss;
      ss << "Learning Rate: " << network.learnRate << '\n' <<
            "Batch Size: " << network.trainingBatchSize << '\n' <<
            "Frame Rate: " << frameRate << '\n';
      
      std::cout << ss.str();

    } else if(User::GetKey(Keyboard::Key::S)) {
      if(delay <= 0) {
        network.SaveToFile(PATH+"/assets/NeuralNetworkData.txt");
        delay = delayTime;
      }

    } else if(User::GetKeyDown(Keyboard::Key::X)) Debug::log("Window Size", Renderer::windowSize);
  }
}
int System::end() {
  GPUMath::ClearGPUMemory();
  return 0;
}


static std::vector<float> ParseFloatVectorFromString(const std::string& _str) {
  std::vector<float> vec;

  std::string buffer;
  for(char c : _str) {
    if(std::isdigit(c) || c == '.' || c == '-') {
      buffer.push_back(c);
      continue;
    }

    if(buffer.size() == 0) continue;

    vec.push_back(std::stof(buffer));
    buffer.clear();
  }

  return vec;
}
void System::RandomizeTrainingData() {
  static constexpr unsigned int dataPointCount = 1000;
  const float w1 = Random::RangeFloat(1.0f, 4.0f);
  const float t1 = Random::RangeFloat(.1f, 2.0f);

  const float w2 = Random::RangeFloat(1.0f, 9.0f);
  const float t2 = Random::RangeFloat(.1f, 3.0f);


  network.trainingData.resize(dataPointCount);

  for(int i = 0; i < dataPointCount; i++) {
    float v1 = Random::RangeFloat(0.0f, 5.0f);
    float v2 = Random::RangeFloat(0.0f, 10.0f);

    float o1 = std::abs(w1 - v1) / t1;
    float o2 = std::abs(w2 - v2) / t2;


    network.trainingData[i] = DataPoint {
      {v1,v2},
      {(o1 > o2? 1.0f : 0.0f), (o2 > o1? 1.0f : 0.0f)}
    };
  }
}
void System::LoadTrainingData() {
  std::ifstream file(System::PATH+"/assets/TrainingData.txt");

  if(file.is_open()) {
    std::string str;
    while(std::getline(file, str)) {
      for(int i = 0; i < str.size(); i++) {
        if(i == 0 || str[i] != '{') continue;

        network.trainingData.push_back(DataPoint{
          ParseFloatVectorFromString(str.substr(0, i)),
          ParseFloatVectorFromString(str.substr(i))
        });

        break;
      }
    }

    file.close();
  }

  if(network.trainingData.size() == 0) {
    RandomizeTrainingData();

    std::ofstream file(System::PATH+"/assets/TrainingData.txt");

    for(const DataPoint& dataPoint : network.trainingData) {
      file << "{";
      for(int i = 0; i < dataPoint.inputs.size() - 1; i++) file << dataPoint.inputs[i] << " ";
      file << dataPoint.inputs[dataPoint.inputs.size() - 1] << "} ";

      file << "{";
      for(int i = 0; i < dataPoint.expectedOutputs.size() - 1; i++) file << dataPoint.expectedOutputs[i] << " ";
      file << dataPoint.expectedOutputs[dataPoint.expectedOutputs.size() - 1] << "}\n";
    }

    file.close();
  }
}


#include <random>
#include <chrono>
int Random::RangeInt(int min, int max) {
  std::mt19937 gen(std::chrono::high_resolution_clock().now().time_since_epoch().count());
  std::uniform_int_distribution rand(min, max);
  return rand(gen);
}
float Random::RangeFloat(float min, float max) {
  std::mt19937 gen(std::chrono::high_resolution_clock().now().time_since_epoch().count());
  std::uniform_real_distribution<float> rand(min, max);
  return rand(gen);
}
#pragma once

#include <vector>
#include <fstream>
#include <System/Debug.hpp>
#include <SFML/System/Clock.hpp>

#include <NeuralNetwork.hpp>


class System {
public:
  static string PATH;

  static sf::Clock timeClock;
  static float deltaTime;

  static std::vector<string> assets;


  static NeuralNetwork network;
  static float learnTime;


  static void init();
  static void update();
  static int end();


  static void RandomizeTrainingData();
  static void LoadTrainingData();


private:
  static void GenerateReleaseAssets() {
    std::ofstream file(PATH+"/assets/Release.txt");
    if(!file.is_open()) {
      Debug::error("Couldnt generate Release assets file");
      return;
    }

    for(int i = 0; i < assets.size(); i++) file << assets[i] << '\n';

    file.close();
  }
};


class Random {
public:
  static int RangeInt(int min, int max);
  static float RangeFloat(float min, float max);
};


static std::string GetConsoleInput(const std::string& _str) {
  std::cout << _str << ": ";
  
  std::string str;
  std::getline(std::cin, str);

  return str;
}
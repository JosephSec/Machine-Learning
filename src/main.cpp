#include <windows.h>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <functional>
#include <random>
#include <chrono>

#include <Renderer.hpp>
#include <ModelTesting.hpp>

#include <NeuralNetwork.hpp>
#define CPU_MODE


static std::filesystem::path PATH;

int main(int argc, char* argv[]) {
  char buffer[MAX_PATH]; GetModuleFileNameA(NULL, buffer, MAX_PATH);
  PATH = std::filesystem::path(buffer).parent_path().parent_path().string();

  
  ModelTesting::init();
  ModelTesting::learn();

  Renderer::init();
  while(Renderer::window.isOpen()) {
    Renderer::events();
    Renderer::draw();
  }

  return 0;
}
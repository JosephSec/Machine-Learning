#include <windows.h>
#include <filesystem>
#include <iostream>

#include <SFML/Graphics.hpp>

#include <WeightLayer.hpp>


static std::filesystem::path PATH;


int main(int argc, char* argv[]) {
  char buffer[MAX_PATH]; GetModuleFileNameA(NULL, buffer, MAX_PATH);
  PATH = std::filesystem::path(buffer).parent_path().parent_path().string();


  WeightLayer output()

  return 0;
}
#include <windows.h>
#include <filesystem>

#include <SFML/Graphics.hpp>


static sf::RenderWindow window;

static std::filesystem::path PATH;


int main(int argc, char* argv[]) {
  char buffer[MAX_PATH]; GetModuleFileNameA(NULL, buffer, MAX_PATH);
  PATH = std::filesystem::path(buffer).parent_path().parent_path().string();

  window = sf::RenderWindow(sf::VideoMode{{800,600}}, "Window");


  while(window.isOpen()) {
    while(std::optional<sf::Event> eventOpt = window.pollEvent()) {
      const auto& event = *eventOpt;

      if(event.is<sf::Event::Closed>()) window.close();
      else if(const auto* resized = event.getIf<sf::Event::Resized>())
        window.setView(sf::View(sf::FloatRect({}, sf::Vector2f{resized->size})));
    }

    window.clear(sf::Color::Black);
    window.display();
  }

  return 0;
}
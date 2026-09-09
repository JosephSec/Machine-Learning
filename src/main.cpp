#include <NeuralNetwork.hpp>

#include <System.hpp>
#include <User.hpp>
#include <Renderer.hpp>

// #define CPU_MODE
#define GPU_MODE


int main(int argc, char* argv[]) {
  System::init();
  sf::RenderWindow* window = &Renderer::window;


  while(window->isOpen()) {
    while(std::optional<Event> eventOpt = window->pollEvent()) {
      const auto& event = *eventOpt;
      User::handle(event);

      if(event.is<Event::Closed>()) window->close();
      else if(const auto* resized = event.getIf<Event::Resized>())
        window->setView(sf::View(sf::FloatRect({}, sf::Vector2f(resized->size))));
    }

    System::update();
    Renderer::draw();
  }

  return 0;
}
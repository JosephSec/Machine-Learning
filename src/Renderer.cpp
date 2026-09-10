#include <Renderer.hpp>
#include <ModelTesting.hpp>


sf::RenderWindow Renderer::window;


void Renderer::init() {
  window = sf::RenderWindow(sf::VideoMode{{800,600}}, "Model Testing");
  window.setFramerateLimit(30);
}
void Renderer::update() {}
void Renderer::draw() {
  window.clear(sf::Color::Black);
  ModelTesting::draw();
  window.display();
}
void Renderer::events() {
  while(const auto &eventOpt = window.pollEvent()) {
    const auto &event = *eventOpt;

    if(event.is<sf::Event::Closed>()) window.close();
    else if(const auto *resized = event.getIf<sf::Event::Resized>()) {
      window.setView(sf::View{sf::FloatRect{{0,0}, sf::Vector2f{resized->size}}});
    }
    else if(const auto *keyPressed = event.getIf<sf::Event::KeyPressed>()) {
      if(keyPressed->code == sf::Keyboard::Key::Escape) window.close();
    }
  }
}
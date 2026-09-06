#pragma once

#include <SFML/Graphics.hpp>

#include <NeuralNetwork.hpp>


class Renderer {
public:
  static sf::RenderWindow window;
  static sf::Vector2f windowSize;
  static sf::Vector2f windowCenter;

  static sf::Font font;
  static sf::Text TextPrefab;


  static void init();
  static void update();
  static void draw();
  static inline void draw(const sf::Drawable& drawable) {
    window.draw(drawable);
  }
};
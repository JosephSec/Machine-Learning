#pragma once

#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/VertexArray.hpp>

#include <Matrix.hpp>


class Renderer {
public:
  static sf::RenderWindow window;

  static void init();
  static void update();
  static void draw();
  static void events();
};
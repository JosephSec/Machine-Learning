#pragma once

#include <iostream>
#include <sstream>
#include <string>
using string = std::string;

#include <SFML/System/Vector2.hpp>
#include <SFML/System/Vector3.hpp>
#include <SFML/Graphics/Rect.hpp>


class Debug {
public:
  static void log(const string& debug, bool line = true) {
    std::cout << debug;
    if(line) std::cout << '\n';
  }


  template <typename T>
  static void log(const string& debug, const sf::Vector2<T>& val, bool line = true) {
    std::ostringstream out;
    out << debug << ": (" << val.x << ", " << val.y << ")";
    if(line) out << '\n';
    std::cout << out.str();
  }

  template <typename T>
  static void log(const string& debug, const sf::Vector3<T>& val, bool line = true) {
    std::ostringstream out;
    out << debug << ": (" << val.x << ", " << val.y << ", " << val.z << ")";
    if(line) out << '\n';
    std::cout << out.str();
  }

  template <typename T>
  static void log(const string& debug, const sf::Rect<T>& val, bool line = true) {
    std::ostringstream out;
    out << debug << ": (" << val.position.x << ", " << val.position.y << ", " << val.size.x << ", " << val.size.y << ")";
    if(line) out << '\n';
    std::cout << out.str();
  }

  template <typename T>
  static void log(const string& debug, const T& val, bool line = true) {
    std::ostringstream out;
    out << debug << ": " << val;
    if(line) out << '\n';
    std::cout << out.str();
  }


  template <typename T>
  static void logbin(const string& debug, const T& val, bool line = true) {
    std::ostringstream out;
    out << debug << ": ";
    const unsigned int bitCount = sizeof(T) * 8;
    for(int i = 0; i < bitCount; i++) {
      out << (char)('0' + ((val >> (bitCount - 1 - i)) & 1ULL));
    }
    if(line) out << '\n';
    std::cout << out.str();
  }

  template <typename T>
  static string getBinaryStr(const T& val) {
    string str;

    const unsigned int bitCount = sizeof(T) * 8;
    for(int i = 0; i < bitCount; i++) {
      str += (char)('0' + ((val >> (bitCount - 1 - i)) & 1ULL));
    }
    
    return str;
  }


  static void logbool(const string& debug, bool val, bool line = true) {
    std::stringstream ss;
    ss << debug << ": " << (val? "True" : "False");
    if(line) ss << '\n';
    std::cout << ss.str();
  }


  enum Error {
    Default,
    OpenFile,
    SaveFile
  };

  static void error(const string& debug, Error type = Default) {
    string str = "[Error]: ";

    switch(type) {
      case OpenFile:
        str += "File missing or can't be opened (" + debug + ")";
        break;
      
      case SaveFile:
        str += "File is being used by another program or can't be found (" + debug + ")";
        break;


      default:
      case Default:
        str += debug;
        break;
    }
    
    str += '\n';
    std::cout << str;
  }
};
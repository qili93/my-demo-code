#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

std::string readTextFile(const char* filename) {
  std::fstream shaderFile(filename, std::ios::in);

  std::stringstream buffer;
  buffer << shaderFile.rdbuf();

  return buffer.str();
}


int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "[ERROR] usage: ./" << argv[0] << "file_path\n";
    exit(1);
  }
  std::string file_path = argv[1];
  std::cout << "File Path is <" << file_path << ">" << std::endl;

  std::string model_buffer = readTextFile(file_path.c_str());

  // std::ifstream ifs(file_path.c_str());
  // std::string content((std::istreambuf_iterator<char>(ifs)),
  //                     (std::istreambuf_iterator<char>()));

  std::cout << "model_buffer length is " << model_buffer.length() << std::endl;

  return 0;
}
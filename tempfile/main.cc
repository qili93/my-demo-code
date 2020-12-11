#include <iostream>
#include <fstream>
#include <string>

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "[ERROR] usage: ./" << argv[0] << "file_path\n";
    exit(1);
  }
  std::string file_path = argv[1];
  std::cout << "File Path is <" << file_path << ">" << std::endl;

  std::ifstream ifs(file_path.c_str());
  std::string content((std::istreambuf_iterator<char>(ifs)),
                      (std::istreambuf_iterator<char>()));

  std::cout << "content length is " << content.length() << std::endl;

  return 0;
}
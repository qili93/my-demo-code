#include <iostream>
#include <fstream>
#include <string>

int main(int argc, char** argv)
{

  std::ifstream ifs("myfile.txt");
  std::string content((std::istreambuf_iterator<char>(ifs)),
                      (std::istreambuf_iterator<char>()));

  std::cout << "content length is " << content.length() << std::endl;
  std::cout << "content string is " << std::endl << content << std::endl;
  
  return 0;
}
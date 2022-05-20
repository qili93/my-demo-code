#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

// std::string readTextFile(const char* filename) {
//   std::fstream shaderFile(filename, std::ios::in);

//   std::stringstream buffer;
//   buffer << shaderFile.rdbuf();

//   return buffer.str();
// }

static std::string ReadFileToBuff(std::string filename) {
  FILE *file = fopen(filename.c_str(), "rb");
  if (file == nullptr) {
    std::cout << "Failed to open file: " << filename << std::endl;
    return nullptr;
  }
  fseek(file, 0, SEEK_END);
  int64_t size = ftell(file);
  if (size == 0) {
    std::cout << "File should not be empty: " << size << std::endl;
    return nullptr;
  }
  rewind(file);
  char * data = new char[size];
  size_t bytes_read = fread(data, 1, size, file);
  if (bytes_read != size) {
    std::cout << "Read binary file bytes do not match with fseek: " << bytes_read << std::endl;
    return nullptr;
  }
  fclose(file);
  std::string file_data(data, size);
  return file_data;
}


int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "[ERROR] usage: ./" << argv[0] << "file_path\n";
    exit(1);
  }
  std::string file_path = argv[1];
  std::cout << "File Path is <" << file_path << ">" << std::endl;

  // int64_t file_size;
  // char * file_data = ReadFileToBuff(file_path, file_size);

  // std::cout << "file size " << file_size << std::endl;

  // std::string mystring(file_data, file_size);

  // std::string model_buffer = readTextFile(file_path.c_str());

  // std::ifstream ifs(file_path.c_str());
  // std::string content((std::istreambuf_iterator<char>(ifs)),
  //                     (std::istreambuf_iterator<char>()));

  std::string mystring = ReadFileToBuff(file_path);

  std::cout << "string length is " << mystring.length() << std::endl;

  return 0;
}
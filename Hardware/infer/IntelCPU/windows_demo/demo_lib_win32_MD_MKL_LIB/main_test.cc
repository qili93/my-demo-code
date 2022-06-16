#include <iostream>
#include <string>

#ifdef USE_FULL_API
extern void SaveModel(const std::string model_path, const int model_type);
extern void RunFullModel(const std::string model_path);
#endif

extern void RunLiteModel(const std::string model_path);


int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "[ERROR] usage: ./" << argv[0] << "model_path\n";
    exit(1);
  }
  std::string model_path = argv[1];
  std::cout << "Model Path is <" << model_path << ">" << std::endl;

  // 0 for uncombined, 1 for combined model
  int model_type = 1;

#ifdef USE_FULL_API
  SaveModel(model_path, model_type);
  RunFullModel(model_path);
#endif

  RunLiteModel(model_path);

  return 0;
}
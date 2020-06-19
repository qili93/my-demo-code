#include "paddle_api.h"
#include <iostream>

const int CPU_THREAD_NUM = 1;
const paddle::lite_api::PowerMode CPU_POWER_MODE =  paddle::lite_api::PowerMode::LITE_POWER_HIGH;

int main(int argc, char **argv) {
    if (argc < 1) {
    printf("Usage: \n"
           "./image_classification_demo model_dir");
    return -1;
    }
    std::string model_dir = argv[1];
    std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor = nullptr;

    paddle::lite_api::CxxConfig cxx_config;
    cxx_config.set_model_dir(model_dir);
    //cxx_config.set_threads(CPU_THREAD_NUM);
    //cxx_config.set_power_mode(CPU_POWER_MODE);
    cxx_config.set_valid_places({.set_valid_places({
        Place{TARGET(kX86), PRECISION(kFloat)},
        Place{TARGET(kHost), PRECISION(kFloat)}
    });
    cxx_config.set_subgraph_model_cache_dir(model_dir.substr(0, model_dir.find_last_of("/")));
    try {
        predictor = paddle::lite_api::CreatePaddlePredictor(cxx_config);
        std::cout << "PaddlePredictor Version: " << predictor->GetVersion() << std::endl;
        // delete old optimized model
        int ret = system(paddle::lite::string_format("rm -rf %s", (model_dir+".nb").c_str());
        if (ret == 0) { std::cout << "delete old optimized model " << model_dir << std::endl; }
        // save new model
        predictor->SaveOptimizedModel(model_dir, paddle::lite_api::LiteModelType::kNaiveBuffer);
        std::cout << "Load model from " << model_dir << std::endl;
        std::cout << "Save optimized model to " << (model_dir+".nb") << std::endl;
    } catch (std::exception e) {
        std::cout << "An internal error occurred in PaddleLite(cxx config)." << std::endl;
    }
    return 0;
}


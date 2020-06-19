#include "paddle_api.h"
#include <iostream>

const int CPU_THREAD_NUM = 1;
const paddle::lite_api::PowerMode CPU_POWER_MODE =  paddle::lite_api::PowerMode::LITE_POWER_HIGH;

struct RESULT {
  int class_id;
  float score;
};

bool topk_compare_func(std::pair<float, int> a, std::pair<float, int> b) {
  return (a.first > b.first);
}

std::vector<RESULT> postprocess(const float *output_data, int64_t output_size) {
    const int TOPK = 3;
    std::vector<std::pair<float, int>> vec;
    for (int i = 0; i < output_size; i++) {
        vec.push_back(std::make_pair(output_data[i], i));
    }
    std::partial_sort(vec.begin(), vec.begin() + TOPK, vec.end(), topk_compare_func);
    std::vector<RESULT> results(TOPK);
    for (int i = 0; i < TOPK; i++) {
        results[i].score = vec[i].first;
        results[i].class_id = vec[i].second;
    }
    return results;
}

int main(int argc, char **argv) {
    if (argc < 1) {
    printf("Usage: \n"
           "./image_classification_demo model_dir");
    return -1;
    }
    std::string model_file = argv[1];
    std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor = nullptr;

    paddle::lite_api::MobileConfig mobile_config;
    mobile_config.set_model_from_file(model_file);
    mobile_config.set_threads(CPU_THREAD_NUM);
    mobile_config.set_power_mode(CPU_POWER_MODE);
    try {
        predictor = paddle::lite_api::CreatePaddlePredictor(mobile_config);
        std::cout << "PaddlePredictor Version: " << predictor->GetVersion() << std::endl;
        std::unique_ptr<paddle::lite_api::Tensor> input_tensor(std::move(predictor->GetInput(0)));
        input_tensor->Resize({1, 1, 28, 28});
        auto* data = input_tensor->mutable_data<float>();
        for (int i = 0; i < 28*28; ++i) {
            data[i] = 1;
        }
        predictor->Run();
        std::unique_ptr<const paddle::lite_api::Tensor> output_tensor(std::move(predictor->GetOutput(0)));
        const float *output_data = output_tensor->data<float>();
        int64_t output_size = 1;
        for (auto dim : output_tensor->shape()) {
            output_size *= dim;
        }
        // auto output_data=output_tensor->data<float>();
        // printf("Output Data is: %f", output_data);
        std::vector<RESULT> results = postprocess(output_data, output_size);
        printf("results: %lu\n", results.size());
        for (int i = 0; i < results.size(); i++) {
            printf("Top%d: %d - %f\n", i, results[i].class_id, results[i].score);
        }
    } catch (std::exception e) {
        std::cout << "An internal error occurred in PaddleLite(cxx config)." << std::endl;
    }
    return 0;
}


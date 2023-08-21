#include <fstream>
#include <sstream>
#include <chrono>
#include <filesystem>
#include "Model/inc/Model.hxx"
#include <torch/script.h>
#include <torchscatter/scatter.h>
#include <torchsparse/sparse.h>
#include <iostream>

int main() {
    // Read data.
    std::filesystem::path current_path = std::filesystem::path(__FILE__).parent_path();
    std::vector<float> X;
    
    std::string line, word;


    std::ifstream X_f = std::ifstream(current_path.string() + "/data/X.csv", std::ios::in);
    while(getline(X_f, line)) {
			std::stringstream str(line);
 
			while(getline(str, word, ',')) {
				X.push_back(std::stof(word));
		  }
    }
    X_f.close();

    // Generate TorchGNN model.
    Model torchGNN_model = Model();

    // Load PyTorch script.
    torch::Device device(torch::kCPU);
    torch::jit::script::Module torch_model = torch::jit::load(current_path.string() + "/model_script.pt", device);

    int batch_size = 64;
    int in_features = 3072;
    int n_classes = 10;

    std::vector<float> out;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    std::chrono::duration<double> torch_time;
    std::chrono::duration<double> torchGNN_time;

    std::filesystem::remove(current_path.string() + "/torch_result.csv");
    std::filesystem::remove(current_path.string() + "/torchGNN_result.csv");
    std::ofstream out_f;
    
    // Evaluate all batches using TorchScript.
    for (int i = 0; i < 10000; i += batch_size) {
      std::vector<float> X_batch(X.begin() + in_features * i, X.begin() + in_features * (i + batch_size));
      std::vector<torch::jit::IValue> inputs;
      inputs.push_back(torch::from_blob(X_batch.data(), {batch_size, in_features}));
      at::Tensor out_batch = torch_model.forward(inputs).toTensor();

      float* out_batch_arr = out_batch.data_ptr<float>();
      for (int j = 0; j < batch_size * n_classes; j++) {
        out.push_back(*out_batch_arr++);
      }
    }

    // Write result to file.
    out_f.open(current_path.string() + "/torch_result.csv", std::ios::app);
    for (std::size_t i = 0; i < out.size(); i += n_classes) {
      bool first_feat = true;
      for (int j = 0; j < n_classes; j++) {
        if (!first_feat) {
          out_f << ",";
        } else {
          first_feat = false;
        }
        out_f << out[i + j];
      }
      out_f << "\n";
    }
    out_f.close();

    // Evaluate all batches using TorchGNN.
    out.clear();
    for (int i = 0; i < 10000; i += batch_size) {
      std::vector<float> X_batch(X.begin() + in_features * i, X.begin() + in_features * (i + batch_size));
      std::vector<float> out_batch = torchGNN_model.Forward(X_batch);

      for (float e: out_batch) {
        out.push_back(e);
      }
    }

    // Write result to file.
    out_f.open(current_path.string() + "/torchGNN_result.csv", std::ios::app);
    for (std::size_t i = 0; i < out.size(); i += n_classes) {
      bool first_feat = true;
      for (int j = 0; j < n_classes; j++) {
        if (!first_feat) {
          out_f << ",";
        } else {
          first_feat = false;
        }
        out_f << out[i + j];
      }
      out_f << "\n";
    }
    out_f.close();

    // Collect timing data.
    for (int round = 0; round < 100; round++) {
      for (int i = 0; i < 10000; i += batch_size) {
        std::vector<float> X_batch(X.begin() + in_features * i, X.begin() + in_features * (i + batch_size));
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(torch::from_blob(X_batch.data(), {batch_size, in_features}));

        start = std::chrono::high_resolution_clock::now();
        at::Tensor out_batch = torch_model.forward(inputs).toTensor();
        end = std::chrono::high_resolution_clock::now();
        torch_time += end - start;
      }

      for (int i = 0; i < 10000; i += batch_size) {
        std::vector<float> X_batch(X.begin() + in_features * i, X.begin() + in_features * (i + batch_size));

        start = std::chrono::high_resolution_clock::now();
        std::vector<float> out_batch = torchGNN_model.Forward(X_batch);
        end = std::chrono::high_resolution_clock::now();
        torchGNN_time += end - start;
      }
    }
    
    // Write timings.
    std::ofstream time_f;
    time_f.open(current_path.string() + "/timings.csv", std::ios::trunc);
    time_f << "PyTorch," << std::chrono::duration_cast<std::chrono::milliseconds>(torch_time).count() << std::endl;
    time_f << "TorchGNN," << std::chrono::duration_cast<std::chrono::milliseconds>(torchGNN_time).count() << std::endl;
    time_f.close();

    return 0;
}
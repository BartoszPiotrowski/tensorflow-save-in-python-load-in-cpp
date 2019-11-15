#include <vector>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/cc/saved_model/loader.h"


// based on https://stackoverflow.com/questions/46098863/how-to-import-an-saved-tensorflow-model-train-using-tf-estimator-and-predict-on
int main(int argc, char** argv) {

	std::string export_dir = argv[1];
	std::cout << "Reading saved model from " << export_dir << std::endl;

	// initial declaration
	tensorflow::Session* session;
	tensorflow::Status status;

	status = tensorflow::NewSession(tensorflow::SessionOptions(), &session);

	if (!status.ok()) {
   		std::cout << status.ToString() << std::endl;
    	return 1;
    }

	tensorflow::SavedModelBundle bundle;
    tensorflow::SessionOptions session_options;
    tensorflow::RunOptions run_options;

	tensorflow::LoadSavedModel(session_options, run_options,
							   export_dir, {"serve"}, &bundle);
	// "serve" is a default tag

	if (!status.ok()) {
    	std::cout << status.ToString() << std::endl;
     	return 1;
   	}

	// We should get the signature (names of input/output) out of MetaGraphDef,
	// but that's a bit involved.
    const std::string input_name = "in:0";
    const std::string output_name = "out:0";

    auto input_value = tensorflow::Tensor(
			tensorflow::DT_FLOAT, tensorflow::TensorShape({10, 784}));
	// TODO read input from a file or initialize it randomly

	std::vector<std::pair<std::string, tensorflow::Tensor>> input =
		{{input_name, input_value}};
    std::vector<tensorflow::Tensor> output;

    status = bundle.session->Run(input, {output_name}, {}, &output);

	if (!status.ok()) {
    	std::cout << status.ToString() << std::endl;
     	return 1;
   	}

    for (const auto& tensor : output) {
		std::cout << tensor.matrix<float>() << std::endl;
    }
}

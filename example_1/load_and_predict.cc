#include <vector>

#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>

// based on https://stackoverflow.com/questions/35508866/tensorflow-different-ways-to-export-and-run-graph-in-cint main(int argc, char** argv) {

int main(int argc, char** argv) {

	std::string meta_graph_path = argv[1]; // path/to/model.meta
	std::string checkpoint_path = argv[2]; // path/to/model
	std::cout << "Reading saved meta graph model from " <<
		meta_graph_path << std::endl;
	std::cout << "Reading saved checkpoint from " <<
		checkpoint_path << std::endl;

	// initial declaration
	tensorflow::Session* session;
	tensorflow::Status status;

	status = tensorflow::NewSession(tensorflow::SessionOptions(), &session);

	if (!status.ok()) {
   		std::cout << status.ToString() << std::endl;
    	return 1;
    }

	// Read in the exported meta-graph
	tensorflow::MetaGraphDef graph_def;
	status = ReadBinaryProto(
			tensorflow::Env::Default(), meta_graph_path, &graph_def);

	if (!status.ok()) {
    	std::cout << status.ToString() << std::endl;
     	return 1;
   	}

	// Add the graph to the session
	status = session->Create(graph_def.graph_def());

	if (!status.ok()) {
    	std::cout << status.ToString() << std::endl;
     	return 1;
   	}

	// Read weights from the saved checkpoint
	tensorflow::Tensor checkpointPathTensor(
			tensorflow::DT_STRING, tensorflow::TensorShape());
	checkpointPathTensor.scalar<std::string>()() = checkpoint_path;
	status = session->Run(
        {{graph_def.saver_def().filename_tensor_name(), checkpointPathTensor }},
        {},
        {graph_def.saver_def().restore_op_name()},
        nullptr);

	std::cout << graph_def.saver_def().filename_tensor_name() << std::endl;
	std::cout << graph_def.saver_def().restore_op_name() << std::endl;

	// We should get the signature (names of input/output) out of MetaGraphDef,
	// but that's a bit involved.
    const std::string input_name = "in:0";
    const std::string output_name = "out:0";

	// prepare input for the graph
    auto input_value = tensorflow::Tensor(
			tensorflow::DT_FLOAT, tensorflow::TensorShape({10, 784}));
	// TODO read input from a file or initialize it randomly

	std::vector<std::pair<std::string, tensorflow::Tensor>> input =
		{{input_name, input_value}};
    std::vector<tensorflow::Tensor> output;

    status = session->Run(input, {output_name}, {}, &output);

	if (!status.ok()) {
    	std::cout << status.ToString() << std::endl;
     	return 1;
   	}

    for (const auto& tensor : output) {
		std::cout << tensor.matrix<float>() << std::endl;
    }
}

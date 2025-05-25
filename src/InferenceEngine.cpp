#include "InferenceEngine.h"
#include "GraphUtils.h"
#include "operators.h"
#include <iostream>
#include <queue>
#include <set>

// === Graph Building ===
void InferenceEngine::build_graph(const onnx::ModelProto& model) {
    const auto& g = model.graph();

    // Parse initializers (constants)
    for (const auto& initializer : g.initializer()) {
        std::vector<int64_t> shape(initializer.dims().begin(), initializer.dims().end());
        Tensor<float> tensor(shape);

        const float* raw_data = nullptr;

        if (initializer.data_type() == onnx::TensorProto::FLOAT) {
            if (initializer.has_raw_data()) {
                raw_data = reinterpret_cast<const float*>(initializer.raw_data().data());
            } else {
                for (int i = 0; i < initializer.float_data_size(); ++i) {
                    tensor.data[i] = initializer.float_data(i);
                }
                graph_.tensors[initializer.name()] = std::move(tensor);
                continue;
            }

            size_t size = tensor.data.size();
            std::memcpy(tensor.data.data(), raw_data, size * sizeof(float));
            graph_.tensors[initializer.name()] = std::move(tensor);
        } else {
            throw std::runtime_error("Unsupported tensor data type.");
        }
    }

    // Parse nodes
    for (const auto& n : g.node()) {
        Node node;
        node.op_type = n.op_type();
        node.input_names.assign(n.input().begin(), n.input().end());
        node.output_names.assign(n.output().begin(), n.output().end());
        for (const auto& attr : n.attribute()) {
            node.attrs[attr.name()] = attr;
        }
        graph_.add_node(std::move(node));
    }

    // Setup operator registry
    op_registry["Relu"] = [](const std::vector<Tensor<float>>& inputs, const Node&) {
        return op_relu(inputs[0]);
    };
    op_registry["Add"] = [](const std::vector<Tensor<float>>& inputs, const Node&) {
        return op_add(inputs[0], inputs[1]);
    };
    op_registry["Flatten"] = [](const std::vector<Tensor<float>>& inputs, const Node&) {
        return op_flatten(inputs[0]);
    };
    op_registry["Gemm"] = [](const std::vector<Tensor<float>>& inputs, const Node& node) {
        bool transA = false, transB = false;
        float alpha = 1.0f, beta = 1.0f;

        if (node.attrs.count("transA"))
            transA = static_cast<bool>(node.attrs.at("transA").i());
        if (node.attrs.count("transB"))
            transB = static_cast<bool>(node.attrs.at("transB").i());
        if (node.attrs.count("alpha"))
            alpha = node.attrs.at("alpha").f();
        if (node.attrs.count("beta"))
            beta = node.attrs.at("beta").f();

        return op_gemm(inputs[0], inputs[1], inputs[2], transA, transB, alpha, beta);
    };

    // Compute execution order
    topo_order_ = topo_sort(graph_);
}

// === Inference Execution ===
Tensor<float> InferenceEngine::infer(const std::unordered_map<std::string, Tensor<float>>& inputs) {
    for (const auto& [name, tensor] : inputs) {
        graph_.tensors[name] = tensor;
    }

    for (size_t idx : topo_order_) {
        const Node& node = graph_.nodes[idx];
        if (op_registry.count(node.op_type) == 0) {
            throw std::runtime_error("Unimplemented operator: " + node.op_type);
        }

        std::vector<Tensor<float>> input_tensors;
        for (const auto& name : node.input_names) {
            if (!graph_.tensors.count(name)) {
                throw std::runtime_error("Missing input tensor: " + name);
            }
            input_tensors.push_back(graph_.tensors[name]);
        }

        Tensor<float> result = op_registry[node.op_type](input_tensors, node);

        // Output: assume single output tensor for now
        graph_.tensors[node.output_names[0]] = std::move(result);
    }

    // Return final output tensor (assume last node’s output)
    const std::string& out_name = graph_.nodes.back().output_names[0];
    return graph_.tensors.at(out_name);
}

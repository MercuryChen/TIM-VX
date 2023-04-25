/****************************************************************************
*
*    Copyright (c) 2020 Vivante Corporation
*
*    Permission is hereby granted, free of charge, to any person obtaining a
*    copy of this software and associated documentation files (the "Software"),
*    to deal in the Software without restriction, including without limitation
*    the rights to use, copy, modify, merge, publish, distribute, sublicense,
*    and/or sell copies of the Software, and to permit persons to whom the
*    Software is furnished to do so, subject to the following conditions:
*
*    The above copyright notice and this permission notice shall be included in
*    all copies or substantial portions of the Software.
*
*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
*    DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/ops/elementwise.h"
#include "tim/vx/ops/nbg.h"

#include "gtest/gtest.h"

#include "tim/utils/trace_utils.h"

#include <vector>

namespace tvx = trace;

TEST(graph, trace_test) {
    auto ctx = tvx::Context::Create();
    auto graph = ctx->CreateGraph();

    tvx::ShapeType io_shape({1,2,2,1});
    tvx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, io_shape, tim::vx::TensorAttribute::INPUT);
    tvx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, io_shape, tim::vx::TensorAttribute::OUTPUT);
    auto input_t0 = graph->CreateTensor(input_spec, nullptr);
    auto input_t1 = graph->CreateTensor(input_spec, nullptr);
    auto output_t = graph->CreateTensor(output_spec, nullptr);

    auto add = graph->CreateOperation<tvx::ops::Add>();
    (*add).BindInputs({input_t0, input_t1}).BindOutputs({output_t});

    size_t bin_size = -1;
    EXPECT_TRUE(graph->CompileToBinary(nullptr, &bin_size));
    EXPECT_NE(bin_size, -1);
    std::vector<char> nbg_buf(bin_size);

    // generate binary graph does't require input data
    EXPECT_TRUE(graph->CompileToBinary(nbg_buf.data(), &bin_size));

    // binary graph compilation doesn't impact current graph's execution
    std::vector<float> in = {1.1f, 2.2f, 3.3f, 4.4f};
    std::vector<float> expected_out = {2.2f, 4.4f, 6.6f, 8.8f};;
    EXPECT_TRUE(input_t0->CopyDataToTensor(in.data(), sizeof(float) * in.size()));
    EXPECT_TRUE(input_t1->CopyDataToTensor(in.data(), sizeof(float) * in.size()));

    EXPECT_TRUE(graph->Run());
    std::vector<float> output(in.size());
    EXPECT_TRUE(output_t->CopyDataFromTensor(output.data()));
    EXPECT_EQ(output, expected_out);

    auto nbg_graph = ctx->CreateGraph();
    auto nbg_in0 = nbg_graph->CreateTensor(input_spec, nullptr);
    auto nbg_in1 = nbg_graph->CreateTensor(input_spec, nullptr);
    auto nbg_out = nbg_graph->CreateTensor(output_spec, nullptr);

    EXPECT_TRUE(nbg_in0->CopyDataToTensor(in.data(), sizeof(float) * in.size()));
    EXPECT_TRUE(nbg_in1->CopyDataToTensor(in.data(), sizeof(float) * in.size()));

    auto nbg_node = nbg_graph->CreateOperation<tvx::ops::NBG, const char*, size_t, size_t>(
        (nbg_buf.data()), /*num_of_input*/ 2,
        /*num_of_output*/ 1);
    (*nbg_node).BindInputs({nbg_in0, nbg_in1}).BindOutputs({nbg_out});
    EXPECT_TRUE(nbg_graph->Compile());
    EXPECT_TRUE(nbg_graph->Run());

    EXPECT_TRUE(nbg_out->CopyDataFromTensor(output.data()));
    EXPECT_EQ(output, expected_out);
}

TEST(graph, replay_test) {
    auto ctx_0 = tim::vx::Context::Create();
    auto graph_0 = ctx_0->CreateGraph();
    auto spec_0 = tim::vx::TensorSpec((tim::vx::DataType)9, trace::_VSI_Replayer::get_vector<unsigned int>(0, 4), (tim::vx::TensorAttribute)8);
    auto spec_1 = tim::vx::TensorSpec((tim::vx::DataType)9, trace::_VSI_Replayer::get_vector<unsigned int>(16, 4), (tim::vx::TensorAttribute)16);
    auto tensor_0 = graph_0->CreateTensor(spec_0, nullptr);
    auto tensor_1 = graph_0->CreateTensor(spec_0, nullptr);
    auto tensor_2 = graph_0->CreateTensor(spec_1, nullptr);
    auto add_0 = graph_0->CreateOperation<tim::vx::ops::Add>();
    add_0->BindInputs({tensor_0,tensor_1});
    add_0->BindOutputs({tensor_2});
    size_t nbg_size_0 = -1;
    graph_0->CompileToBinary(nullptr, &nbg_size_0);
    graph_0->CompileToBinary(trace::_VSI_Replayer::get_vector<char>(32, 5212).data(), &nbg_size_0);
    tensor_0->CopyDataToTensor(trace::_VSI_Replayer::get_vector<char>(5244, 16).data(), 16);
    tensor_1->CopyDataToTensor(trace::_VSI_Replayer::get_vector<char>(5260, 16).data(), 16);
    graph_0->Run();
    tensor_2->CopyDataFromTensor(trace::_VSI_Replayer::get_vector<char>(5276, 16).data());
    auto graph_1 = ctx_0->CreateGraph();
    auto tensor_3 = graph_1->CreateTensor(spec_0, nullptr);
    auto tensor_4 = graph_1->CreateTensor(spec_0, nullptr);
    auto tensor_5 = graph_1->CreateTensor(spec_1, nullptr);
    tensor_3->CopyDataToTensor(trace::_VSI_Replayer::get_vector<char>(5292, 16).data(), 16);
    tensor_4->CopyDataToTensor(trace::_VSI_Replayer::get_vector<char>(5308, 16).data(), 16);
    std::vector<char> nbg_buf_vec_0 = trace::_VSI_Replayer::get_vector<char>(5324, 5212);
    auto nbg_0 = graph_1->CreateOperation<tim::vx::ops::NBG>(nbg_buf_vec_0.data(), 2, 1);
    nbg_0->BindInputs({tensor_3,tensor_4});
    nbg_0->BindOutputs({tensor_5});
    graph_1->Compile();
    graph_1->Run();
    tensor_5->CopyDataFromTensor(trace::_VSI_Replayer::get_vector<char>(10536, 16).data());
}

TEST(graph, gen_binary_graph_with_empty_graph) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    size_t bin_size = -1;
    EXPECT_FALSE(graph->CompileToBinary(nullptr, &bin_size)) << "Can not generate binary graph if it is empty";
}

TEST(graph, gen_binary_graph_with_simple_add) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({1,1,1,1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, io_shape, tim::vx::TensorAttribute::OUTPUT);
    auto input_t0 = graph->CreateTensor(input_spec);
    auto input_t1 = graph->CreateTensor(input_spec);
    auto output_t = graph->CreateTensor(output_spec);

    auto add = graph->CreateOperation<tim::vx::ops::Add>();
    (*add).BindInputs({input_t0, input_t1}).BindOutputs({output_t});

    size_t bin_size = -1;
    EXPECT_TRUE(graph->CompileToBinary(nullptr, &bin_size));
    EXPECT_NE(bin_size, -1);
    std::vector<char> nbg_buf(bin_size);

    // generate binary graph does't require input data
    EXPECT_TRUE(graph->CompileToBinary(nbg_buf.data(), &bin_size));

    // binary graph compilation doesn't impact current graph's execution
    float in = 1.0f;
    float expected_out = 2.0f;
    EXPECT_TRUE(input_t0->CopyDataToTensor(&in, sizeof(in)));
    EXPECT_TRUE(input_t1->CopyDataToTensor(&in, sizeof(in)));

    EXPECT_TRUE(graph->Run());
    float output = 0.0f;
    EXPECT_TRUE(output_t->CopyDataFromTensor(&output));
    EXPECT_EQ(output, expected_out);

    auto nbg_graph = ctx->CreateGraph();
    auto nbg_in0 = nbg_graph->CreateTensor(input_spec);
    auto nbg_in1 = nbg_graph->CreateTensor(input_spec);
    auto nbg_out = nbg_graph->CreateTensor(output_spec);

    EXPECT_TRUE(nbg_in0->CopyDataToTensor(&in, sizeof(in)));
    EXPECT_TRUE(nbg_in1->CopyDataToTensor(&in, sizeof(in)));

    auto nbg_node = nbg_graph->CreateOperation<tim::vx::ops::NBG>(
        (nbg_buf.data()), /*num_of_input*/ 2,
        /*num_of_output*/ 1);
    (*nbg_node).BindInputs({nbg_in0, nbg_in1}).BindOutputs({nbg_out});
    EXPECT_TRUE(nbg_graph->Compile());
    EXPECT_TRUE(nbg_graph->Run());

    output=0.0f;
    EXPECT_TRUE(nbg_out->CopyDataFromTensor(&output));
    EXPECT_EQ(output, expected_out);
}

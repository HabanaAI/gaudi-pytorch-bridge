#include <ATen/ATen.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/inline_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/lower_grad_of.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/specialize_autogradzero.h>
#include "test_base.h"
#include "test_utils.h"
#include "torch/csrc/autograd/engine.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/frontend/tracer.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/passes/constant_propagation.h"
#include "torch/csrc/jit/passes/create_autodiff_subgraphs.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/graph_fuser.h"
#include "torch/csrc/jit/passes/lower_grad_of.h"
#include "torch/csrc/jit/passes/requires_grad_analysis.h"
#include "torch/csrc/jit/passes/shape_analysis.h"
#include "torch/csrc/jit/passes/utils/subgraph_utils.h"
#include "torch/csrc/jit/runtime/argument_spec.h"
#include "torch/csrc/jit/runtime/autodiff.h"

namespace torch {
namespace jit {

using namespace torch::autograd;

using var_meta_type = std::vector<int64_t>;
using var_meta_list = std::vector<var_meta_type>;
using test_fn_type = std::function<variable_list(const variable_list&)>;

struct ADTestSpec {
  ADTestSpec(const char* name, var_meta_list input_meta, test_fn_type test_fn)
      : name(name), input_meta(input_meta), test_fn(test_fn) {}

  variable_list operator()(const variable_list& inputs) const {
    return test_fn(inputs);
  };

  // struct Net : torch::nn::Module {
  // Net()
  //    : conv1(torch::nn::Conv2dOptions(1, 20, /*kernel_size=*/5)),
  //     conv2(torch::nn::Conv2dOptions(20, 50, /*kernel_size=*/5)),
  /*     fc1(450, 500),
       fc2(500, 10) {
   register_module("conv1", conv1);
   register_module("conv2", conv2);
   register_module("conv2_drop", conv2_drop);
   register_module("fc1", fc1);
   register_module("fc2", fc2);
 }

 torch::Tensor forward(torch::Tensor x) {
   x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
   x = torch::relu(
       torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
   x = x.view({-1, 450});
   x = torch::relu(fc1->forward(x));
   x = fc2->forward(x);*/
  //    return torch::log_softmax(x, /*dim=*/1);
  /*  }

    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
  };*/

  std::vector<Variable> make_vars() const {
    std::vector<Variable> out;
    for (const auto& m : input_meta) {
      out.push_back(torch::randn(m, at::requires_grad(true)));
    }
    return out;
  }

  const char* name;
  var_meta_list input_meta;
  test_fn_type test_fn;
};

variable_list get_grad_outputs(const variable_list& vars) {
  return fmap(vars, [](const Variable& v) -> Variable {
    return at::randn(v.sizes(), v.options());
  });
}

variable_list grad(
    const variable_list& outputs,
    const variable_list& inputs,
    const variable_list& grad_outputs) {
  const auto get_edge = [](const Variable& v) {
    return torch::autograd::impl::gradient_edge(v);
  };
  auto& engine = torch::autograd::Engine::get_default_engine();
  return engine.execute(
      fmap(outputs, get_edge),
      grad_outputs,
      true,
      false,
      fmap(inputs, get_edge));
}

/*void testADFormulas() {
  const auto cast = [](const Variable& v) { return static_cast<at::Tensor>(v);
};

  using VL = variable_list;
  const var_meta_list binary_pointwise = {{2, 3, 4, 5}, {2, 3, 4, 5}};
  const var_meta_list unary_pointwise = {{2, 3, 4, 5}};
  const var_meta_list unary_pointwise_2d = {{2, 3}};
  const std::vector<ADTestSpec> ad_tests = {
      {"add",
       binary_pointwise,
       [](const VL& v) -> VL { return {v[0] + v[1]}; }},
      {"sub",
       binary_pointwise,
       [](const VL& v) -> VL { return {v[0] - v[1]}; }},
      {"mul",
       binary_pointwise,
       [](const VL& v) -> VL { return {v[0] * v[1]}; }},
      {"sigmoid",
       unary_pointwise,
       [](const VL& v) -> VL { return {v[0].sigmoid()}; }},
      {"tanh",
       unary_pointwise,
       [](const VL& v) -> VL { return {v[0].tanh()}; }},
      {"t", unary_pointwise_2d, [](const VL& v) -> VL { return {v[0].t()}; }},
      {"view",
       unary_pointwise_2d,
       [](const VL& v) -> VL {
         return {v[0].view({3, 2})};
       }},
      {"expand",
       {{2, 1}},
       [](const VL& v) -> VL {
         return {v[0].expand({2, 3})};
       }},
      {"mm",
       {{10, 12}, {12, 15}},
       [](const VL& v) -> VL { return {v[0].mm(v[1])}; }},
      // TODO: enable once we'll be able to capture lists across
      // forward-backward
      //{"chunk",   {{10, 12, 15}}, [](const VL& v) -> VL { return
      // fmap<Variable>(v[0].chunk(4, 1)); }},
      //{"chunk",   {{10, 12, 15}}, [](const VL& v) -> VL { return
      // fmap<Variable>(v[0].chunk(3, 2)); }},
      //{"split",   {{10, 12, 15}}, [](const VL& v) -> VL { return
      // fmap<Variable>(v[0].split(4, 1)); }},
      //{"split",   {{10, 12, 15}}, [](const VL& v) -> VL { return
      // fmap<Variable>(v[0].split(3, 2)); }},
  };

  for (const auto& test : ad_tests) {
    // Get reference values form autograd
    auto vars_in = test.make_vars();
    auto vars_out = test(vars_in);
    auto var_grads_in = get_grad_outputs(vars_out);
    auto var_grads_out = grad(vars_out, vars_in, var_grads_in);

    // Trace and differentiate the op
    auto graph = tracer::trace(
      fmap<IValue>(vars_in),
      [&test](Stack in) -> Stack {
        auto ivalue_inps = fmap(in, [](const IValue& v){
          return Variable(v.toTensor());
        });
        return fmap<IValue>(test(ivalue_inps));
      },
      [](const Variable& var) { return "";}
    ).first->graph;
    EliminateDeadCode(graph); // Tracing of some ops depends on the DCE trick
    ConstantPropagation(graph);
    auto grad_spec = differentiate(graph);
    LowerGradOf(*grad_spec.df);
    // Get outputs from the interpreter
    auto tensors_in = fmap(vars_in, cast);
    auto tensor_grads_in = fmap(var_grads_in, cast);
    tensor_list tensors_out, tensor_grads_out;
    std::tie(tensors_out, tensor_grads_out) =
        runGradient(grad_spec, tensors_in, tensor_grads_in);

    // Compare results
    auto expected_tensors_out = fmap(vars_out, cast);
    auto expected_tensor_grads_out = fmap(var_grads_out, cast);
    assertAllClose(tensors_out, expected_tensors_out);
    assertAllClose(tensor_grads_out, expected_tensor_grads_out);
  }
}*/

void testDifferentiate() {
  // Note: can't use IRParser for this test due to issue #23989
  auto graph = std::make_shared<Graph>();
  const auto type = TensorType::create(
      at::ScalarType::Float, at::kCPU, {2, 3, 4}, {12, 4, 1});

  // Builds graph a * b * a + b
  auto* a = graph->addInput()->setType(type);
  auto* b = graph->addInput()->setType(type);
  auto* cOne = graph->insertConstant(1);

  auto* ab = graph->insertNode(graph->create(aten::mul, /*num_outputs =*/1));
  ab->addInput(a);
  ab->addInput(b);

  auto* aba = graph->insertNode(graph->create(aten::mul, /*num_outputs =*/1));
  aba->addInput(ab->output());
  aba->addInput(a);

  auto* abaplusb =
      graph->insertNode(graph->create(aten::add, /*num_outputs =*/1));
  abaplusb->addInput(aba->output());
  abaplusb->addInput(b);
  abaplusb->addInput(cOne);

  graph->registerOutput(abaplusb->output());

  auto grad_spec = differentiate(graph);
  std::vector<size_t> expected_captured_inputs = {0, 1};
  std::vector<size_t> expected_captured_outputs = {1, 2, 3, 4, 5, 6, 7};
  std::vector<size_t> expected_input_vjps = {0, 1};
  std::vector<size_t> expected_output_vjps = {0, 1};
  ASSERT_EQ(grad_spec.f_real_outputs, 1);
  ASSERT_EQ(grad_spec.df_input_captured_inputs, expected_captured_inputs);
  ASSERT_EQ(grad_spec.df_input_captured_outputs, expected_captured_outputs);
  ASSERT_EQ(grad_spec.df_input_vjps, expected_input_vjps);
  ASSERT_EQ(grad_spec.df_output_vjps, expected_output_vjps);
  testing::FileCheck()
      .check_count("= aten::mul", 2)
      ->check("= aten::size")
      ->check("= aten::add")
      ->run(*grad_spec.f);
  testing::FileCheck()
      .check("= prim::GradOf[name=\"aten::add\"]")
      ->check_count("= prim::GradOf[name=\"aten::mul\"]", 2)
      ->check_count("= AutogradAdd", 2)
      ->run(*grad_spec.df);
}

void testDifferentiateWithRequiresGrad(std::string graph_string) {
  /*const auto graph_string = R"IR(
    graph(%0 : Tensor,
          %1 : Tensor):
      %2 : int = prim::Constant[value=1]()
      %3 : Tensor = aten::mul(%1, %1)
      %4 : Tensor = aten::add(%3, %1, %2)
      %5 : Tensor = aten::add(%4, %0, %2)
      %6 : Tensor = aten::mul(%5, %0)
      %7 : Tensor = aten::add(%6, %1, %2)
      return (%4, %7))IR";*/

  auto g = std::make_shared<Graph>();

  torch::jit::parseIR(graph_string, g.get());

  /*auto a_var = autograd::make_variable(
      at::empty_strided(2, 2, at::CPU(at::kFloat).options()), true);
  auto b_var = autograd::make_variable(
      at::empty_strided(2, 2, at::CPU(at::kFloat).options()), false);
  auto c_var = autograd::make_variable(
      at::empty_strided(2, 2, at::CPU(at::kFloat).options()), false);

  ArgumentSpecCreator asc(*g);
  asc.specializeTypes(*g, asc.create(true, {a_var, b_var, c_var}));*/

  Inline(*g);
  LowerGradOf(*g);
  specializeAutogradZero(*g);
  LowerSimpleTuples(g);
  ConstantPooling(g);
  runRequiredPasses(g);
  ConstantPropagation(g);
  PropagateInputShapes(g);
  PropagateRequiresGrad(g);
  auto grad_spec = differentiate(g);
  std::cout << "Before lowering tuples in forward graph" << std::endl;
  std::cout << grad_spec.f->toString() << std::endl;
  LowerAllTuples(grad_spec.f);
  std::cout << "After lowering tuples in forward graph" << std::endl;
  std::cout << grad_spec.f->toString() << std::endl;
  std::cout << "Before lowering tuples in backward graph" << std::endl;
  std::cout << grad_spec.df->toString() << std::endl;
  LowerAllTuples(grad_spec.df);
  std::cout << "After lowering tuples in backward graph" << std::endl;
  std::cout << grad_spec.df->toString() << std::endl;
  std::cout << "Before HabanaFuse Op in forward graph" << std::endl;
  std::cout << grad_spec.f->toString() << std::endl;
  FuseGraph(grad_spec.f);
  std::cout << grad_spec.f->toString() << std::endl;
  HabanaFuseGraph(grad_spec.f);
  std::cout << "After HabanaFuse Op in forward graph" << std::endl;
  std::cout << grad_spec.f->toString() << std::endl;
  std::cout << "Before HabanaFuse Op in backward graph" << std::endl;
  std::cout << grad_spec.df->toString() << std::endl;
  HabanaFuseGraph(grad_spec.df);
  std::cout << "After HabanaFuse Op in backward graph" << std::endl;
  std::cout << grad_spec.df->toString() << std::endl;

  std::vector<size_t> expected_input_vjps = {1, 2}; // for e and %4 = (d + a)
  std::vector<size_t> expected_output_vjps = {0}; // only a requires grad
  ASSERT_EQ(grad_spec.f_real_outputs, 2);
  ASSERT_EQ(grad_spec.df_input_captured_inputs, std::vector<size_t>({0}));
  ASSERT_EQ(
      grad_spec.df_input_captured_outputs,
      std::vector<size_t>({2, 3, 4, 5, 6}));
  ASSERT_EQ(grad_spec.df_input_vjps, expected_input_vjps);
  ASSERT_EQ(grad_spec.df_output_vjps, expected_output_vjps);
  testing::FileCheck()
      .check("= aten::mul")
      ->check_count("= aten::add", 2)
      ->check("= aten::mul")
      ->check("= aten::size")
      ->check("= aten::add")
      ->run(*grad_spec.f);

  testing::FileCheck()
      .check_count("= prim::GradOf[name=\"aten::mul\"]", 1, /*exactly*/ true)
      ->run(*grad_spec.df);
}

/*void mnist_model()
{

  auto graph = std::make_shared<Graph>();
  auto a_var = autograd::make_variable(
      at::empty_strided(5, 1, 28, 28, at::CPU(at::kFloat).options()), true);

  const auto type = TensorType::create(Float, at::kCPU, {2, 3, 4}, {12, 4, 1});

  auto* a  = graph->addInput()->setType(type);
  auto* b  = graph->addInput()->setType(type);
  auto* c  = graph->addInput()->setType(type);

  auto conv_node = graph->insertNode(graph->create(aten::conv2d()))




}*/

} // namespace jit
} // namespace torch

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "Invalid Arguments!" << std::endl;
    std::cout << "Usage: ./test_plugin --file <path to ir string file>"
              << std::endl;
    return 0;
  }

  /*torch::jit::script::Module module = torch::jit::load(argv[2]);
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({ 5, 1, 28, 28 }));
  at::Tensor output = module.forward(inputs).toTensor();*/

  std::ifstream input_file_handle(argv[2]);
  std::string graph_str(
      (std::istreambuf_iterator<char>(input_file_handle)),
      (std::istreambuf_iterator<char>()));
  // std::stringstream file_buffer;
  // file_buffer << input_file_handle.rdbuf();
  // auto graph_string = file_buffer.str();
  torch::jit::testDifferentiateWithRequiresGrad(graph_str);

  return 0;
}
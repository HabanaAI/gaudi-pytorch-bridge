/**
 * Copyright (c) 2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdint>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include "sl_report_generator/src/stack_generator.h"
#include "sl_report_generator/src/utils/shared_structures.h"
#include "sl_report_generator/src/utils/tensor_helpers.h"

#undef CHECK

#include <doctest/doctest.h>

TEST_SUITE("Stack Generator tests") {
  TEST_CASE("Main tests") {
    std::unique_ptr<slrg::StackGenerator> stack_gen = nullptr;
    SUBCASE("Only ranks") {
      std::vector<std::int64_t> rank{};
      std::vector<at::Stack> expected_stacks{};

      SUBCASE("No rank") {}

      SUBCASE("Rank -1") {
        rank.emplace_back(-1);
        expected_stacks.emplace_back(at::Stack{});
      }

      SUBCASE("Rank 0") {
        rank.emplace_back(0);
        expected_stacks.emplace_back(at::Stack{});
      }

      SUBCASE("Rank 1") {
        rank.emplace_back(1);
        expected_stacks.emplace_back(at::Stack{});
      }

      SUBCASE("Rank 2") {
        rank.emplace_back(2);
        expected_stacks.emplace_back(at::Stack{});
      }

      SUBCASE("2 Ranks") {
        rank.emplace_back(2);
        rank.emplace_back(1);
        expected_stacks.emplace_back(at::Stack{});
        expected_stacks.emplace_back(at::Stack{});
      }

      stack_gen = std::make_unique<slrg::StackGenerator>(rank);

      CHECK(stack_gen->getStacks(at::ScalarType::Float) == expected_stacks);
      CHECK(stack_gen->getOpAndOverloadName() == "");
      CHECK_FALSE(stack_gen->isBlacklistedPrecisionType(at::ScalarType::Int));
      CHECK_FALSE(stack_gen->isWhitelistedPrecisionType(at::ScalarType::Bool));
      CHECK(stack_gen->DebugString() == "");
    }

    SUBCASE("Full") {
      std::vector<slrg::InputDescriptor> inputs{};
      std::vector<at::ScalarType> blacklisted_types{};
      std::vector<at::ScalarType> whitelisted_types{};
      std::string op_name = "Dummy";
      std::string op_overload_name;

      std::vector<at::Stack> expected_stacks{};

      SUBCASE("Default") {
        expected_stacks.push_back(at::Stack{});
      }

      SUBCASE("Single Input") {
        slrg::InputDescriptor input{};

        SUBCASE("Optional") {
          input.is_optional = true;

          SUBCASE("Only none") {
            input.allow_only_none = true;
            expected_stacks.push_back(at::Stack{c10::IValue()});
          }
        }

        SUBCASE("PT_TENSOR") {
          input.type = slrg::InputType::PT_TENSOR;

          SUBCASE("Single rank and dtype") {
            input.dtypes = {at::ScalarType::Float};
            input.ranks = {1};
            input.match_rank = true;
            input.match_precision_type = true;

            expected_stacks.push_back(
                at::Stack{slrg::createTensor(1, at::ScalarType::Float)});
          }

          SUBCASE("Multiple rank and dtype") {
            input.dtypes = {at::ScalarType::Float, at::ScalarType::Int};
            input.ranks = {1, 2};

            SUBCASE("Match rank and precision type") {
              input.match_rank = true;
              input.match_precision_type = true;

              expected_stacks.push_back(
                  at::Stack{slrg::createTensor(1, at::ScalarType::Float)});
            }

            SUBCASE("Don't match rank and precision type") {
              input.match_rank = false;
              input.match_precision_type = false;

              expected_stacks.push_back(
                  at::Stack{slrg::createTensor(1, at::ScalarType::Float)});
              expected_stacks.push_back(
                  at::Stack{slrg::createTensor(2, at::ScalarType::Float)});
              expected_stacks.push_back(
                  at::Stack{slrg::createTensor(1, at::ScalarType::Int)});
              expected_stacks.push_back(
                  at::Stack{slrg::createTensor(2, at::ScalarType::Int)});
            }
          }
        }

        SUBCASE("PT_SCALAR") {
          input.type = slrg::InputType::PT_SCALAR;

          input.values = std::vector<std::any>{0.0, 1};

          expected_stacks.push_back(at::Stack{0.0});
          expected_stacks.push_back(at::Stack{1});
        }

        SUBCASE("DTYPE") {
          input.type = slrg::InputType::PT_SCALAR;

          SUBCASE("Match precision_type") {
            input.match_precision_type = true;
            input.values = {0.0};

            expected_stacks.push_back(at::Stack{0.0});
          }

          SUBCASE("Multiple values") {
            input.values = {0.0, 1};

            expected_stacks.push_back(at::Stack{0.0});
            expected_stacks.push_back(at::Stack{1});
          }
        }

        SUBCASE("NATIVE") {
          input.type = slrg::InputType::NATIVE_FLOAT;

          SUBCASE("Match precision_type") {
            input.match_precision_type = true;
            input.values = {0.0F};

            expected_stacks.push_back(at::Stack{0.0F});
          }

          SUBCASE("Multiple values") {
            input.values = {0.0F, 1.0F};

            expected_stacks.push_back(at::Stack{0.0F});
            expected_stacks.push_back(at::Stack{1.0F});
          }
        }

        inputs.push_back(input);
      }

      stack_gen = std::make_unique<slrg::StackGenerator>(
          inputs,
          blacklisted_types,
          whitelisted_types,
          op_name,
          op_overload_name);

      CHECK(stack_gen->getStacks(at::ScalarType::Float) == expected_stacks);
    }
  }

  TEST_CASE("Exceptions") {
    std::vector<slrg::InputDescriptor> inputs{};
    std::vector<at::ScalarType> blacklisted_types{};
    std::vector<at::ScalarType> whitelisted_types{};
    std::string op_name = "Dummy";
    std::string op_overload_name;

    std::vector<at::Stack> expected_stacks{};

    SUBCASE("Single Input") {
      slrg::InputDescriptor input{};
      doctest::String exception_string{};

      SUBCASE("Optional") {
        input.is_optional = true;

        SUBCASE("Not definied allow_none") {
          exception_string =
              "StackGenerator cannot process InputDescriptor: for optional inputs at least one of the parameters 'allow_none' must be specified or 'allow_only_none' must be TRUE";
        }
      }

      SUBCASE("PT_TENSOR") {
        input.type = slrg::InputType::PT_TENSOR;

        SUBCASE("Default") {
          exception_string =
              "StackGenerator::generateIValues<InputType::PT_TENSOR> cannot process InputDescriptor for op 'Dummy': for tensors the 'match_rank' parameters must be specified";
        }

        SUBCASE("Don't Match_rank") {
          input.match_rank = false;
          exception_string =
              "StackGenerator::generateIValues<InputType::PT_TENSOR> cannot process InputDescriptor for op 'Dummy': for tensors with 'match_rank'=false, the 'ranks' parameters must be specified";
        }

        SUBCASE("Not matching precision types") {
          input.match_rank = true;
          exception_string =
              "StackGenerator::generateIValues<InputType::PT_TENSOR> cannot process InputDescriptor for op 'Dummy': for tensors the 'match_precision_type' parameter must be specified";
        }

        SUBCASE("Missing explicit dtypes") {
          input.match_rank = true;
          input.match_precision_type = false;
          exception_string =
              "StackGenerator::generateIValues<InputType::PT_TENSOR> cannot process InputDescriptor for op 'Dummy': for tensors with 'match_precision_type'=false, the 'dtypes' parameter must be specified";
        }
      }

      SUBCASE("PT_SCALAR") {
        input.type = slrg::InputType::PT_SCALAR;

        SUBCASE("default") {
          exception_string =
              "StackGenerator::generateIValues<InputType::PT_SCALAR> cannot process InputDescriptor for op 'Dummy': the 'values' parameter must be specified";
        }

        SUBCASE("Empty vector as value") {
          input.values = std::vector<std::any>{};

          exception_string =
              "StackGenerator::generateIValues<InputType::PT_SCALAR> cannot process InputDescriptor for op 'Dummy': the 'values' parameter cannot be empty";
        }

        SUBCASE("Invalid type") {
          input.values = std::vector<std::any>{"nope"};

          exception_string =
              "StackGenerator::generateIValues<InputType::PT_SCALAR> cannot process InputDescriptor for op 'Dummy': unknown value type 'PKc' for param ''";
        }
      }

      SUBCASE("DTYPE") {
        input.type = slrg::InputType::DTYPE;

        SUBCASE("default") {
          SUBCASE("No values") {}

          SUBCASE("Empty vector as value") {
            input.values = std::vector<std::any>{};
          }

          exception_string =
              "StackGenerator::generateIValues<InputType::DTYPE> cannot process InputDescriptor for op 'Dummy': for input with 'match_precision_type'=false, the 'values' parameter cannot be empty (param '')";
        }

        SUBCASE("Not all values are scalars") {
          input.values = std::vector<std::any>{0.0};

          exception_string =
              "StackGenerator::generateIValues<InputType::DTYPE> cannot process InputDescriptor for op 'Dummy': the 'values' parameter must contain only ScalarTypes (param '')";
        }
      }

      SUBCASE("Native") {
        input.type = slrg::InputType::NATIVE_INT;

        SUBCASE("No values") {
          exception_string =
              "StackGenerator::generateIValuesBasicTypes<T, InputType::NATIVE_INT> cannot process InputDescriptor for op 'Dummy': the 'values' parameter must be specified (param '')";
        }

        SUBCASE("Empty vector") {
          input.values = std::vector<std::any>{};

          exception_string =
              "StackGenerator::generateIValuesBasicTypes<T, InputType::NATIVE_INT> cannot process InputDescriptor for op 'Dummy': the 'values' parameter cannot be empty (param '')";
        }

        SUBCASE("Invalid type") {
          input.values = std::vector<std::any>{0.0};

          exception_string =
              "StackGenerator::generateIValuesBasicTypes<T, InputType::NATIVE_INT> cannot process InputDescriptor for op 'Dummy': the 'values' parameter have values of type corresponding to InputType (param ''), got: 'd'";
        }
      }

      SUBCASE("Generator") {
        input.type = slrg::InputType::GENERATOR;

        exception_string =
            "StackGenerator::getInputVariants cannot process the input type: GENERATOR. Generator should be replaced with 'Tensor seed'.";
      }

      slrg::StackGenerator stack_gen(
          std::vector<slrg::InputDescriptor>{input},
          blacklisted_types,
          whitelisted_types,
          op_name,
          op_overload_name);

      CHECK_THROWS_WITH_AS(
          stack_gen.getStacks(at::ScalarType::Int),
          exception_string,
          std::invalid_argument);
    }
  }

  TEST_CASE("Op name and overload") {
    std::string op_overload_name;
    std::string expected_name;

    std::unique_ptr<slrg::StackGenerator> stack_gen = nullptr;

    SUBCASE("empty overload") {
      expected_name = "dummy";
    }

    SUBCASE("named overload") {
      op_overload_name = "dummy.test";
      expected_name = "dummy.test";
    }

    stack_gen = std::make_unique<slrg::StackGenerator>(
        std::vector<slrg::InputDescriptor>{},
        std::vector<at::ScalarType>{},
        std::vector<at::ScalarType>{},
        "dummy",
        op_overload_name);

    CHECK(stack_gen->getOpAndOverloadName() == expected_name);
  }
}

TEST_SUITE("Schema Stack Generator tests") {
  TEST_CASE("Main test") {
    c10::List<c10::IValue> expected_list(c10::IntType::get());
    for (auto i{0}; i < 4; ++i)
      expected_list.push_back(1);

    std::vector<at::Stack> expected_stacks{at::Stack{
        slrg::createTensor(1, at::ScalarType::Int),
        c10::IValue(expected_list),
        c10::IValue()}};

    slrg::SchemaStackGenerator generator(
        "Generator wat, int[4] values, bool? foo", "dummy");
    auto generated_stacks = generator.getStacks(at::ScalarType::Float);

    CHECK(expected_stacks == generated_stacks);
  }

  TEST_CASE("Invalid type") {
    CHECK_THROWS_WITH_AS(
        slrg::SchemaStackGenerator generator("invalid param", "dummy"),
        "SchemaStackGenerator::generateInputs cannot process the input schema for 'dummy' op. Unknown input type: 'invalid' for param 'invalid param'",
        std::invalid_argument);
  }

  TEST_CASE("Op name and overload") {
    std::string op_overload_name;
    std::string expected_name;

    std::unique_ptr<slrg::SchemaStackGenerator> stack_gen = nullptr;

    SUBCASE("empty overload") {
      expected_name = "dummy";
    }

    SUBCASE("named overload") {
      op_overload_name = "dummy.test";
      expected_name = "dummy.test";
    }

    stack_gen = std::make_unique<slrg::SchemaStackGenerator>(
        "",
        "dummy",
        op_overload_name);

    CHECK(stack_gen->getOpAndOverloadName() == expected_name);
  }
}

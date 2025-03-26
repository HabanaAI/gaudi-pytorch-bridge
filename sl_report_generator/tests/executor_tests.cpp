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

#include <exception>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "sl_report_generator/src/executor.h"
#include "sl_report_generator/src/utils/shared_structures.h"

#include "mocks/check_node_with_shared_layer_validator.h"
#include "mocks/i_stack_generator.h"

// name collision with doctest
#undef CHECK

#include <doctest/doctest.h>

using SelectedValidator =
    slrg::unit_tests::CheckNodeWithSharedLayerValidatorMock;

TEST_SUITE("Executor tests") {
  TEST_CASE("Main tests") {
    std::unique_ptr<slrg::SharedLayerExecutor<SelectedValidator>> executor;
    slrg::Report expected_report;

    SUBCASE("Static Executor") {
      slrg::Report supported_types;

      SUBCASE("Nothing is supported") {}

      SUBCASE("Float is supported") {
        supported_types[at::ScalarType::Float] = true;
        expected_report[at::ScalarType::Float] = true;
      }

      SUBCASE("Float and int is supported") {
        supported_types[at::ScalarType::Float] = true;
        supported_types[at::ScalarType::Int] = true;

        expected_report[at::ScalarType::Float] = true;
        expected_report[at::ScalarType::Int] = true;
      }

      executor =
          std::make_unique<slrg::StaticSharedLayerExecutor<SelectedValidator>>(
              supported_types);
    }

    SUBCASE("Generic Executor") {
      slrg::unit_tests::IStackGeneratorMock generator{};
      generator.configure().enable_function("isBlacklistedPrecisionType");

      SelectedValidator validator{};

      for (const auto& type : slrg::report_precision_types)
        expected_report[type] = false;

      executor =
          std::make_unique<slrg::GenericSharedLayerExecutor<SelectedValidator>>(
              &generator, &validator);

      SUBCASE("Executor is validated once, All types are unsupported") {
        generator.configure()
            .enable_outputs_cycling("isBlacklistedPrecisionType")
            .register_output("isBlacklistedPrecisionType", true);
        executor->validate();
        executor->validate();
        CHECK(
            slrg::report_precision_types.size() ==
            generator.getFunctionCallCounter("isBlacklistedPrecisionType"));
      }

      SUBCASE("Only Float is supported") {
        expected_report[at::ScalarType::Float] = true;

        generator.configure()
            .enable_function("isWhitelistedPrecisionType")
            .register_outputs(
                "isBlacklistedPrecisionType",
                {false,
                 true,
                 true,
                 true,
                 true,
                 true,
                 true,
                 true,
                 true,
                 true,
                 true})
            .register_output("isWhitelistedPrecisionType", true);
      }

      SUBCASE("Stacks tests") {
        generator.configure()
            .enable_function("isWhitelistedPrecisionType")
            .enable_function("getStacks");

        SUBCASE("No stacks") {
          for (const auto& type : slrg::report_precision_types)
            expected_report[type] = true;
        }

        SUBCASE("Validate INT dtype") {
          for (const auto& type : slrg::report_precision_types)
            expected_report[type] = true;

          generator.configure()
              .register_output("getStacks", std::vector<at::Stack>{{0.0}})
              .enable_outputs_cycling("getStacks");

          validator.configure()
              .enable_function("Validate")
              .register_outputs(
                  "Validate",
                  {true,
                   true,
                   true,
                   true,
                   true,
                   true,
                   false,
                   true,
                   true,
                   true});

          expected_report[at::ScalarType::Int] = false;
        }
      }
      executor->validate();
    }

    SUBCASE("Custom Executor") {
      slrg::unit_tests::IStackGeneratorMock generator{};
      generator.configure().enable_function("isBlacklistedPrecisionType");

      SelectedValidator validator{};

      for (const auto& type : slrg::report_precision_types)
        expected_report[type] = false;

      executor =
          std::make_unique<slrg::CustomSharedLayerExecutor<SelectedValidator>>(
              &generator, &validator);

      SUBCASE("Executor is validated once, All types are unsupported") {
        generator.configure()
            .enable_outputs_cycling("isBlacklistedPrecisionType")
            .register_output("isBlacklistedPrecisionType", true);
        executor->validate();
        executor->validate();
        CHECK(
            slrg::report_precision_types.size() ==
            generator.getFunctionCallCounter("isBlacklistedPrecisionType"));
      }

      SUBCASE("Only Float is supported") {
        expected_report[at::ScalarType::Float] = true;

        generator.configure()
            .enable_function("isWhitelistedPrecisionType")
            .register_outputs(
                "isBlacklistedPrecisionType",
                {false,
                 true,
                 true,
                 true,
                 true,
                 true,
                 true,
                 true,
                 true,
                 true,
                 true})
            .register_output("isWhitelistedPrecisionType", true);
      }

      SUBCASE("Stacks tests") {
        generator.configure()
            .enable_function("isWhitelistedPrecisionType")
            .enable_function("getStacks");

        SUBCASE("No stacks") {
          for (const auto& type : slrg::report_precision_types)
            expected_report[type] = true;
        }

        SUBCASE("Validate INT dtype") {
          for (const auto& type : slrg::report_precision_types)
            expected_report[type] = true;

          generator.configure()
              .register_output("getStacks", std::vector<at::Stack>{{0.0}})
              .enable_outputs_cycling("getStacks");

          validator.configure()
              .enable_function("ValidateCustom")
              .register_outputs(
                  "ValidateCustom",
                  {true,
                   true,
                   true,
                   true,
                   true,
                   true,
                   false,
                   true,
                   true,
                   true});

          expected_report[at::ScalarType::Int] = false;
        }
      }
      executor->validate();
    }
    const auto report = executor->getReport();
    CHECK(expected_report == report);
  }

  TEST_CASE("Exceptions") {
    SUBCASE("Generic Executor validate exception catch") {
      slrg::unit_tests::IStackGeneratorMock generator{};
      SelectedValidator validator{};

      slrg::GenericSharedLayerExecutor<SelectedValidator> executor(
          &generator, &validator);

      generator.configure()
          .enable_function("isBlacklistedPrecisionType")
          .enable_function("isWhitelistedPrecisionType")
          .enable_function("getStacks")
          .register_output("getStacks", std::vector<at::Stack>{{0.0}})
          .enable_outputs_cycling("getStacks")
          .enable_function("getOpAndOverloadName")
          .register_output("getOpAndOverloadName", "Dummy::TestOp");

      CHECK_THROWS_WITH_AS(
          executor.validate(),
          "SharedLayerExecutor::validate failed for op 'Dummy::TestOp'. Possibly incorrect input quantity, signature or TORCH_CHECK inside output_meta.\nOriginal exception: Unexpected CheckNodeWithSharedLayerValidator::Validate function call.",
          std::logic_error);
    }

    SUBCASE("Custom Executor validate exception catch") {
      slrg::unit_tests::IStackGeneratorMock generator{};

      SelectedValidator validator{};

      slrg::CustomSharedLayerExecutor<SelectedValidator> executor(
          &generator, &validator);

      generator.configure()
          .enable_function("isBlacklistedPrecisionType")
          .enable_function("isWhitelistedPrecisionType")
          .enable_function("getStacks")
          .register_output("getStacks", std::vector<at::Stack>{{0.0}})
          .enable_outputs_cycling("getStacks")
          .enable_function("getOpAndOverloadName")
          .register_output("getOpAndOverloadName", "Dummy::TestOp");

      CHECK_THROWS_WITH_AS(
          executor.validate(),
          "SharedLayerExecutor::validate failed for op 'Dummy::TestOp'. Possibly incorrect input quantity, signature or TORCH_CHECK inside output_meta.\nOriginal exception: Unexpected CheckNodeWithSharedLayerValidator::ValidateCustom function call.",
          std::logic_error);
    }
  }
}

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

#include <gtest/gtest.h>
#include "backend/helpers/symbolic_expression.h"

TEST(TestSymExpression, SizeExpression_tokenizer) {
    constexpr auto simple_expr = "s35 + s35*((-1) + s44";
    constexpr auto advance_expr = "max(25, s35 + s35*((-1) + s44)), 1024";

    auto one_token = habana::SizeExpression::tokenizer(simple_expr);
    ASSERT_EQ(simple_expr, one_token.at(0));

    constexpr auto expected_first_token{"max(25, s35 + s35*((-1) + s44))"};
    constexpr auto expected_second_token{" 1024"};
    auto two_tokens = habana::SizeExpression::tokenizer(advance_expr);
    ASSERT_EQ(expected_first_token, two_tokens.at(0));
    ASSERT_EQ(expected_second_token, two_tokens.at(1));
}

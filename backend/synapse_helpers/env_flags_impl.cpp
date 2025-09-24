/**
 * Copyright (c) 2021-2025 Intel Corporation
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

/*******************************************************************************
 * INTEL CONFIDENTIAL
 * Copyright 2018-2020 Intel Corporation.
 *
 * This software and the related documents are Intel copyrighted materials, and
 * your use of them is governed by the express license under which they were
 * provided to you ("License"). Unless the License provides otherwise, you may
 * not use, modify, copy, publish, distribute, disclose or transmit this
 * software or the related documents without Intel's prior written permission.
 *
 * This software and the related documents are provided as is, with no express
 * or implied warranties, other than those that are expressly stated in
 * the License.
 *******************************************************************************
 */

#include "env_flags_impl.h"
#include "habana_helpers/logging.h"

// ****************************************************************************
// New style of env var declaration
namespace env_flags::new_style {

namespace fs = std::filesystem;

bool has_insecure_chars(const std::string& input) {
  return std::regex_search(input, insecure_pattern);
}

bool has_unsafe_chars(const std::string& input) {
  return std::regex_search(input, unsafe_pattern);
}

enum class ConstrainsType { ENUM, RANGE, FILEPATH, LIST, UNKNOWN };

std::string trim(const std::string& s) {
  auto is_trim_char = [](unsigned char c) { return std::isspace(c); };
  auto start = std::find_if_not(s.begin(), s.end(), is_trim_char);
  auto end = std::find_if_not(s.rbegin(), s.rend(), is_trim_char).base();
  return (start < end) ? std::string(start, end) : std::string();
}

/**
 * @brief Checks the status of an environment flag and handles deprecated or
 * obsolete flags.
 *
 * This function examines the provided flag status and performs the following
 * actions:
 * - If the flag is marked as deprecated (FLAG_STATUS_DEPRECATED), it logs a
 * warning message but allows execution to continue.
 * - If the flag is marked as obsolete (FLAG_STATUS_OBSOLETE), it logs an error
 * message and throws a std::runtime_error.
 *
 * @param name The name of the environment flag being checked.
 * @param flag_status The status of the flag (e.g., FLAG_STATUS_DEPRECATED,
 * FLAG_STATUS_OBSOLETE
 * ).
 *
 * @throws std::runtime_error if the flag is obsolete.
 */
void check_flag_status(const char* name, const char* flag_status) {
  std::string flag_status_str(flag_status);
  if (flag_status_str == FLAG_STATUS_DEPRECATED) {
    PT_SYNHELPER_WARN(
        "[Config Warning] Environment flag ",
        name,
        " is deprecated and may be removed in the future.");
  } else if (flag_status_str == FLAG_STATUS_OBSOLETE) {
    PT_SYNHELPER_FATAL(
        "[Config Error] Environment flag ",
        name,
        " is obsolete and cannot be set.");
  }
}

/**
 * @brief Determines the type of constraint based on the input string.
 *
 * This function checks the provided string for specific substrings that
 * correspond appropriate ConstrainsType enum value based on the first match
 * found. If none of the known constraint types are found in the input string,
 * it returns ConstrainsType::UNKNOWN.
 *
 * @param constrains_type The string representing the constraint type.
 * @return ConstrainsType The corresponding enum value for the constraint type.
 */
ConstrainsType get_constrains_type(const std::string& constrains_type) {
  if (constrains_type.find(ENUM_CONSTRAINT_TYPE) != std::string::npos)
    return ConstrainsType::ENUM;
  if (constrains_type.find(RANGE_CONSTRAINT_TYPE) != std::string::npos)
    return ConstrainsType::RANGE;
  if (constrains_type.find(FILEPATH_CONSTRAINT_TYPE) != std::string::npos)
    return ConstrainsType::FILEPATH;
  if (constrains_type.find(LIST_CONSTRAINT_TYPE) != std::string::npos)
    return ConstrainsType::LIST;
  return ConstrainsType::UNKNOWN;
}

/**
 * @brief Validates if a given value is part of a set of allowed values defined
 *        by a constraints string in the format "value1|value2|value3".
 *
 * @param value The value to be checked against the allowed values.
 * @param constraints A string containing the allowed values separated by '|'.
 *        Optional spaces around the values are supported.
 * @param error_msg A reference to a string where an error message will be
 * stored if the validation fails. The message will describe the issue.
 * @return true if the value is part of the allowed values, false otherwise.
 *
 * @note The constraints string must follow the format "value1|value2|value3"
 *       . If the format is invalid, the function
 *       will return false and set an appropriate error message.
 */
bool check_enum(
    const std::string& value,
    const std::string& constraints,
    std::string& error_msg) {
  // Regular expression to match the format: value1|value2|value3, and supports
  // optional spaces around the values, such as "value1 | value2 | value3"
  static const std::regex pattern(
      R"(^\s*[^|]+\s*(\|\s*[^|]+\s*)*$)", std::regex::icase);
  std::smatch match;
  if (!std::regex_match(constraints, match, pattern)) {
    error_msg = constraints +
        " does not match expected format. Expected: value1 | value2 | value3";
    return false;
  }

  std::vector<std::string> allowed_values;
  std::istringstream ss(constraints);
  std::string token;
  while (std::getline(ss, token, CONSTRAINTS_SPLIT_PIPE)) {
    std::transform(token.begin(), token.end(), token.begin(), ::tolower);
    allowed_values.push_back(trim(token));
  }

  if (std::find(allowed_values.begin(), allowed_values.end(), value) !=
      allowed_values.end()) {
    return true;
  }

  error_msg = "Value '" + value +
      "' is not allowed. Allowed values: " + constraints + ".";
  return false;
}

/**
 * @brief Checks if a given string value, interpreted as an integer, falls
 * within a specified range.
 *
 * The range is provided as a string in the format "(start,end)".
 * If the value is within the range [start, end], the function returns true.
 * Otherwise, it sets an error message describing the violation and returns
 * false.
 *
 * @param value The string representation of the integer value to check.
 * @param constraints The string specifying the allowed range in the format
 * "(start,end)".
 * @param error_msg Reference to a string where the error message will be set if
 * the value is out of range.
 * @return true if the value is within the specified range, false otherwise.
 */
bool check_range(
    const std::string& value,
    const std::string& constraints,
    std::string& error_msg) {
  // Regular expression to match the range format "(start, end), and supports
  // optional space format, such as "( 1 , 10 )"
  static const std::regex pattern(R"(\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\))");
  std::smatch match;

  if (!std::regex_match(constraints, match, pattern)) {
    error_msg = constraints +
        " does not match expected format. Expected: (start, end), e.g. (1, 10)";
    return false;
  }

  int start = std::stoi(match[1].str());
  int end = std::stoi(match[2].str());

  int val;
  try {
    val = std::stoi(value);
  } catch (...) {
    error_msg = "Value '" + value + "' is not a valid integer.";
    return false;
  }

  if (val >= start && val <= end)
    return true;

  error_msg =
      "Value '" + value + "' is out of range. Allowed range: " + constraints;
  return false;
}

std::vector<std::string> split_csv(const std::string& str) {
  std::vector<std::string> result;
  std::stringstream ss(trim(str));
  std::string item;
  while (std::getline(ss, item, CONSTRAINTS_SPLIT_COMMA)) {
    if (!trim(item).empty())
      result.push_back(trim(item));
  }
  return result;
}

/**
 * @brief Checks if the input list of values satisfies the specified
 * constraints.
 *
 * The constraints string must be in the format:
 * "<ordered_flag>|<allowed_list>", where:
 *   - <ordered_flag> is "true" or "false", indicating if the allowed list must
 * be a suffix of the input (ordered) or if the input must be a subset of the
 * allowed list (unordered).
 *   - <allowed_list> is a comma-separated list of allowed values, enclosed in
 * angle brackets, e.g., "<a,b,c>".
 *
 * @param value      The input list as a comma-separated string.
 * @param constraints The constraint string specifying order and allowed values.
 * @param error_msg   Reference to a string to receive an error message if the
 * check fails.
 * @return true if the input list satisfies the constraints; false otherwise
 * (with error_msg set).
 *
 * Constraint rules:
 *   - If ordered is true: the allowed list must be a suffix of the input list.
 *   - If ordered is false: every item in the input list must be present in the
 * allowed list.
 *   - Returns false and sets error_msg if the constraint format is invalid or
 * the check fails.
 *
 *  e.g., "ordered=true | <A,B,C>"
 */
bool check_list(
    const std::string& value,
    const std::string& constraints,
    std::string& error_msg) {
  if (value.empty()) {
    error_msg = "Input list is empty.";
    return false; // Empty input list
  }
  static const std::regex pattern(
      R"(^\s*ordered\s*=\s*(true|false)\s*,\s*<\s*([^<>]+[^,\s])\s*>\s*$)",
      std::regex::icase);
  std::smatch match;
  if (!std::regex_match(constraints, match, pattern)) {
    error_msg = constraints +
        " does not match expected format. Expected: ordered=true|false,<A,B,C>.";
    return false;
  }
  auto comma_pos = constraints.find(CONSTRAINTS_SPLIT_COMMA);
  if (comma_pos == std::string::npos) {
    error_msg = "Constraint format invalid: missing ','.";
    return false;
  }
  std::string ordered_part = trim(constraints.substr(0, comma_pos));
  std::string list_part = trim(constraints.substr(comma_pos + 1));

  std::string flag = ordered_part.substr(
      ordered_part.find('=') + 1); // Extract the value after 'ordered='
  std::transform(flag.begin(), flag.end(), flag.begin(), ::tolower);
  bool ordered = (flag.find(CONSTRAINTS_TRUE) != std::string::npos);

  std::string allowed_str = list_part.substr(1, list_part.size() - 2);
  std::vector<std::string> allowed = split_csv(allowed_str);
  std::vector<std::string> input = split_csv(value);

  if (ordered) {
    // Ordered = true, allowed must be a **suffix** of input
    if (input.size() < allowed.size()) {
      error_msg = "Input list is shorter than allowed suffix.";
      return false;
    }
    if (!std::equal(
            input.end() - allowed.size(), input.end(), allowed.begin())) {
      error_msg = "Input list does not match required ordered suffix: <" +
          allowed_str + ">.";
      return false;
    }
    return true;
  } else {
    // Ordered = false, input must be subset of allowed
    std::unordered_set<std::string> allowed_set(allowed.begin(), allowed.end());
    for (const auto& item : input) {
      if (allowed_set.count(item) == 0) {
        error_msg = "Input item '" + item + "' is not in allowed set: <" +
            allowed_str + ">.";
        return false;
      }
    }
    return true;
  }
}

/**
 * @brief Validates a file or directory path based on specified constraints.
 *
 * This function checks the provided path against a set of constraints to ensure
 * it meets security and structural requirements. It can validate whether the
 * path is a file or directory, check for unsafe or insecure characters, and
 * optionally create directories if specified.
 *
 * @param path The file or directory path to validate.
 * @param constraints A string containing constraint keywords and values.
 * Supported keywords:
 *   - "is_file": Indicates the path should be a regular file.
 *   - "is_dir": Indicates the path should be a directory.
 *   - "is_create=true": Allows creation of the directory if it does not exist.
 * @param error_msg A reference to a string where error messages will be stored
 * if validation fails.
 *
 * @return `true` if the path meets the constraints, `false` otherwise.
 *
 * @note If the path is empty, the function will log a warning and return
 * `true`.
 * @note Unsafe and insecure characters in the path are checked using helper
 * functions `has_unsafe_chars` and `has_insecure_chars`.
 * @note If the "is_dir" constraint is specified and the directory does not
 * exist, it will be created if "is_create=true" is provided in the constraints.
 * @note Symbolic links are not allowed for directories.
 * @note The function uses `std::filesystem` for file and directory operations.
 */
bool check_file_path(
    const std::string& path,
    const std::string& constraints,
    std::string& error_msg) {
  if (path.empty()) {
    PT_SYNHELPER_WARN("[Config Warning] Empty path provided.");
    return true;
  }

  // Generic checks for security
  if (has_unsafe_chars(path)) {
    error_msg = "Directory path contains unsafe characters: " + path +
        ". Please avoid using characters like " + unsafe_pattern_str +
        " in the path.";
    return false;
  }

  if (has_insecure_chars(path)) {
    error_msg = "Directory path contains insecure characters: " + path +
        ". Please avoid using characters like " + insecure_pattern_str +
        " in the path.";
    return false;
  }

  // Helper: check constraint keyword
  auto has_flag = [&](const std::string& key) {
    return constraints.find(key) != std::string::npos;
  };

  auto get_value = [&](const std::string& key) -> std::optional<std::string> {
    size_t pos = constraints.find(key + CONSTRAINTS_SPLIT_EQUAL);
    if (pos == std::string::npos)
      return std::nullopt;
    size_t start = pos + key.length() + 1;
    size_t end = constraints.find_first_of(CONSTRAINTS_SPLIT_AND, start);
    return constraints.substr(start, end - start);
  };
  // Parse constraint details
  bool is_file = has_flag("is_file");
  bool is_dir = has_flag("is_dir");
  bool is_create =
      get_value("is_create").value_or(CONSTRAINTS_FALSE) == CONSTRAINTS_TRUE;

  if (is_file) {
    if (!fs::is_regular_file(path)) {
      error_msg = "Path is not a regular file: " + path;
      return false;
    }
    return true;
  }

  if (is_dir) {
    if (fs::exists(path)) {
      if (fs::is_symlink(path)) {
        error_msg = "Path is a symbolic link: " + path +
            ". Symbolic links are not allowed.";
        return false; // Path is a symbolic link
      }
      if (!fs::is_directory(path)) {
        error_msg = "Path is not a directory: " + path;
        return false; // Path exists but is not a directory
      }
      if ((fs::status(path).permissions() & fs::perms::owner_write) ==
          fs::perms::none) {
        error_msg = "Directory path is not writable: " + path;
        return false; // Directory is not writable
      }
      return true; // Path exists, valid either way
    } else if (is_create) {
      try {
        fs::create_directories(path);
        return fs::exists(path); // Confirm creation
      } catch (const fs::filesystem_error& e) {
        error_msg = "Filesystem error while creating directory '" + path +
            "': " + e.what();
        return false;
      }
    } else {
      PT_SYNHELPER_WARN(
          "[Config Warning] Path does not exist:" + path +
          "and creation is not needed.");
      return true; // Path doesn't exist and create is not needed
    }
  }

  // Default fallback
  error_msg = "Path does not meet any recognized file/dir constraints.";
  return false;
}

/**
 * @brief Validates the configuration string for recipe cache settings.
 *
 * This function checks if the provided configuration string adheres to the
 * expected format and constraints. The configuration string should consist
 * of four comma-separated fields:
 *     <path>,<true|false>,<int_size_in_MB>,<true|false>
 *
 * @param config The configuration string to validate.
 * @param error_msg A reference to a string where the error message will be
 *                 stored if validation fails. This will be cleared at the
 *                 start of the function.
 * @return true if the configuration is valid, false otherwise.
 *
 * @details The function performs the following checks:
 * PT_HPU_RECIPE_CACHE_CONFIG is a comma-separated list where parameters are
 * encoded as follows:
 * - 1st param: Recipe cache directory path. If empty, disk cache is disabled.
 *   If a proper path is set, disk cache is enabled for all compiled recipes.
 * - 2nd param: Delete recipe cache on init. If set to true, the PT bridge
 *   clears the recipe cache on initialization.
 * - 3rd param: Recipe cache max size in MB. If set to a value > 0, the PT
 *   bridge keeps the size of the cache directory under the defined threshold.
 *   For example, 1GB per worker to save recipes to the disk.
 * - 4th param: Allow recipe cache path to be on NFS. If enabled, the ability
 *   to store recipes on NFS (path provided with the 1st param) is allowed.
 *   Once enabled, the 2nd and 3rd parameters are ignored, as no retention
 *   policy can be safely applied to an NFS location.
 *
 * Example:
 * PT_HPU_RECIPE_CACHE_CONFIG=/tmp/recipe-cache,true,1024,false
 */
bool check_recipe_cache_config(
    const std::string& config,
    std::string& error_msg) {
  static const std::regex pattern(
      R"(^\s*(?:([^\s,<>|?*"]+)\s*)?(?:,\s*(true|false)\s*(?:,\s*(\d+)\s*(?:,\s*(true|false)\s*)?)?)?\s*$)",
      std::regex_constants::icase);
  std::smatch match;
  if (!std::regex_match(config, match, pattern)) {
    error_msg = config + " does not match expected format. " +
        "Expected: <path>,<true|false>,<int_size_in_MB>,<true|false>.";
    return false;
  }

  const std::string dir_path = trim(match[1].str());
  if (!dir_path.empty()) {
    if (has_unsafe_chars(dir_path)) {
      error_msg = "Directory path contains unsafe characters: " + dir_path +
          ". Please avoid using characters like " + unsafe_pattern_str +
          " in the path.";
      return false;
    }

    if (has_insecure_chars(dir_path)) {
      error_msg = "Directory path contains insecure characters: " + dir_path +
          ". Please avoid using characters like " + insecure_pattern_str +
          " in the path.";
      return false;
    }

    if (!fs::exists(dir_path)) {
      PT_SYNHELPER_WARN(
          "[Config Warning] Directory path does not exist: ",
          dir_path,
          ". Attempting to create it if allowed.");
    } else {
      if (!fs::is_directory(dir_path)) {
        error_msg = "Path is not a directory: " + dir_path;
        return false;
      }

      if (fs::is_symlink(dir_path)) {
        error_msg = "Path is a symbolic link: " + dir_path +
            ". Symbolic links are not allowed.";
        return false; // Path is a symbolic link
      }

      // Check if the directory is writable
      if ((fs::status(dir_path).permissions() & fs::perms::owner_write) ==
          fs::perms::none) {
        error_msg = "Directory path is not writable: " + dir_path;
        return false; // Directory is not writable
      }
    }
  }

  return true;
}

/**
 * @brief Validates a value against specified constraints and reports errors.
 *
 * This function checks if the provided value (as a C-string) meets the
 * requirements defined by the given constraint type and constraint string. It
 * performs security checks for insecure characters and delegates validation to
 * specialized functions based on the an appropriate error message is set.
 *
 * @param constrains_type The type of constraint to apply (e.g., "ENUM",
 * "RANGE", etc.).
 * @param value The value to validate, as a C-string.
 * @param constraints The constraint definition string (e.g., allowed values,
 * range).
 * @param error_msg Reference to a string where error messages will be stored if
 * validation fails.
 * @return true if the value is valid according to the constraints; false
 * otherwise.
 */
bool check(
    const char* constrains_type,
    const char* value,
    const char* constraints,
    std::string& error_msg) {
  std::string constrainsType_str =
      constrains_type ? std::string(constrains_type) : CONSTRAINTS_EMPTY;
  std::string constraints_str =
      constraints ? std::string(constraints) : CONSTRAINTS_EMPTY;
  std::string value_str = std::string(value);
  const std::string value_trim = trim(value_str);
  if (has_insecure_chars(value_trim)) {
    error_msg = "Value contains insecure characters: " + value_trim +
        ". Please avoid using characters like " + insecure_pattern_str + ".";
    return false;
  }

  if (!constrainsType_str.empty()) {
    ConstrainsType type = get_constrains_type(constrainsType_str);
    const std::string constraints_trim = trim(constraints_str);
    switch (type) {
      case ConstrainsType::ENUM:
        return check_enum(value_trim, constraints_trim, error_msg);
      case ConstrainsType::RANGE:
        return check_range(value_trim, constraints_trim, error_msg);
      case ConstrainsType::FILEPATH:
        return check_file_path(value_trim, constraints_trim, error_msg);
      case ConstrainsType::LIST:
        return check_list(value_trim, constraints_trim, error_msg);
      default: {
        error_msg = "Unknown constrains type: " + constrainsType_str;
        return false;
      }
    }
  }
  return true; // No constraints, value is valid
}

bool check(
    const char* constrains_type,
    const bool& value,
    const char* constraints,
    std::string& error_msg) {
  (void)constrains_type;
  (void)value;
  (void)constraints;
  (void)error_msg;
  return true; // No constraints, value is valid
}

/**
 * @brief Checks whether a given value satisfies specified constraints.
 *
 * This function converts the input value to a string and validates it against
 * the provided constraints, If the constraint type is unknown, an error message
 * is set and the function returns false.
 *
 * This overload is for base types such as int, long, unsigned, unsigned long,
 * signed long, long long, unsigned long long.
 *
 * @tparam T The type of the value to be checked (base integer types).
 * @param constrains_type The type of constraint to apply (e.g., "ENUM",
 * "RANGE", etc.).
 * @param value The value to be validated.
 * @param constraints The constraint definition as a string.
 * @param error_msg Reference to a string where an error message will be stored
 * if validation fails.
 * @return true if the value satisfies the constraints or if no constraint type
 * is specified; false otherwise.
 */
template <typename T>
bool check(
    const char* constrains_type,
    const T& value,
    const char* constraints,
    std::string& error_msg) {
  std::ostringstream oss;
  oss << value;
  std::string value_str = oss.str();
  std::string constrainsType_str =
      constrains_type ? std::string(constrains_type) : CONSTRAINTS_EMPTY;
  std::string constraints_str =
      constraints ? std::string(constraints) : CONSTRAINTS_EMPTY;

  if (!constrainsType_str.empty()) {
    ConstrainsType type = get_constrains_type(constrainsType_str);
    const std::string constraints_trim = trim(constraints_str);
    switch (type) {
      case ConstrainsType::ENUM:
        return check_enum(value_str, constraints_trim, error_msg);
      case ConstrainsType::RANGE:
        return check_range(value_str, constraints_trim, error_msg);
      case ConstrainsType::FILEPATH:
        return check_file_path(value_str, constraints_trim, error_msg);
      case ConstrainsType::LIST:
        return check_list(value_str, constraints_trim, error_msg);
      default: {
        error_msg = "Unknown constrains type: " + constrainsType_str;
        return false;
      }
    }
  }
  return true;
}

void validate_constraint_custom(
    const char* name,
    const std::string& value,
    std::function<bool(const std::string&, std::string&)> validator) {
  std::string error_msg;
  std::string name_str = name ? std::string(name) : "unknown";
  if (!validator(value, error_msg)) {
    PT_SYNHELPER_FATAL("Flag ", name_str, ": ", error_msg);
  }
}

template <typename T>
void validate_constraint_with_type(
    const char* name,
    const char* constrains_type,
    const T& result,
    const char* constrains) {
  std::string error_msg;
  std::string name_str = name ? std::string(name) : "";
  if (!name_str.empty()) {
    if (!check(constrains_type, result, constrains, error_msg)) {
      PT_SYNHELPER_FATAL("Flag ", name_str, ": ", error_msg);
    }
  } else {
    if (!check(CONSTRAINTS_EMPTY, result, CONSTRAINTS_EMPTY, error_msg)) {
      PT_SYNHELPER_FATAL("Flag unknown: ", error_msg);
    }
  }
}

// ****************************************************************************

} // namespace env_flags::new_style

template void env_flags::new_style::validate_constraint_with_type<int>(
    const char*,
    const char*,
    const int&,
    const char*);
template void env_flags::new_style::validate_constraint_with_type<unsigned int>(
    const char*,
    const char*,
    const unsigned int&,
    const char*);
template void env_flags::new_style::validate_constraint_with_type<long>(
    const char*,
    const char*,
    const long&,
    const char*);
template void env_flags::new_style::validate_constraint_with_type<
    unsigned long>(const char*, const char*, const unsigned long&, const char*);
template void env_flags::new_style::validate_constraint_with_type<long long>(
    const char*,
    const char*,
    const long long&,
    const char*);
template void env_flags::new_style::validate_constraint_with_type<
    unsigned long long>(
    const char*,
    const char*,
    const unsigned long long&,
    const char*);
template void env_flags::new_style::validate_constraint_with_type<const char*>(
    const char*,
    const char*,
    const char* const&,
    const char*);
template void env_flags::new_style::validate_constraint_with_type<std::string>(
    const char*,
    const char*,
    const std::string&,
    const char*);
template void env_flags::new_style::validate_constraint_with_type<bool>(
    const char*,
    const char*,
    const bool&,
    const char*);

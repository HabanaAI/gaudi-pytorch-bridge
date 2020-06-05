/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once

namespace synapse_helpers {

/**
 * Checks if environment variable is set to specified value.
 * @param   variable_name   Name of the environment variable.
 * @param   value           Expected value.
 * @param   unset_default   Value to return if variable is unset.
 * @return                  True if value of variable is same as value.
 */
bool is_env_var_equal(const char* variable_name, const char* value, bool unset_default = false);

/**
 * Checks environment variable state.
 * True, if set to '1' or case-insensitive 'true'.
 * False, if set to '0' or case-insensitive 'false'.
 * 'unset_default' if unset or contains unexpected data.
 * @param   variable_name   Name of the environment variable.
 * @param   unset_default   Value to return if variable is unset or contains unexpected data.
 * @return                  Environment variable state.
 */
bool get_bool_env_var(const char* variable_name, bool unset_default = false);

}  // namespace synapse_helpers
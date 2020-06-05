cmake_minimum_required(VERSION 3.0)

foreach (ABSL_TARGET
  algorithm
  atomic_hook
  bad_any_cast_impl
  bad_optional_access
  bad_variant_access
  base
  base_internal
  bits
  city
  civil_time
  compressed_tuple
  config
  container_common
  container_memory
  core_headers
  debugging_internal
  demangle_internal
  dynamic_annotations
  endian
  examine_stack
  failure_signal_handler
  fixed_array
  flags
  flags_config
  flags_handle
  flags_internal
  flags_marshalling
  flags_parse
  flags_registry
  flags_usage
  flags_usage_internal
  graphcycles_internal
  hash
  hash_policy_traits
  hashtable_debug_hooks
  hashtablez_sampler
  have_sse
  inlined_vector
  inlined_vector_internal
  int128
  layout
  leak_check
  leak_check_disable
  log_severity
  malloc_internal
  memory
  meta
  optional
  random_distributions
  random_internal_distribution_caller
  random_internal_distribution_impl
  random_internal_distribution_test_util
  random_internal_distributions
  random_internal_fast_uniform_bits
  random_internal_fastmath
  random_internal_iostream_state_saver
  random_internal_nonsecure_base
  random_internal_platform
  random_internal_pool_urbg
  random_internal_randen
  random_internal_randen_hwaes
  random_internal_randen_hwaes_impl
  random_internal_randen_slow
  random_internal_salted_seed_seq
  random_internal_seed_material
  random_internal_traits
  random_internal_uniform_helper
  random_seed_gen_exception
  random_seed_sequences
  raw_hash_set
  scoped_set_env
  span
  spinlock_wait
  stacktrace
  str_format
  str_format_internal
  strings
  strings_internal
  symbolize
  synchronization
  throw_delegate
  time
  time_zone
  type_traits
  utility
  variant
  )

  list(APPEND ABSL_TARGETS absl_${ABSL_TARGET})
  set_target_properties(absl_${ABSL_TARGET} PROPERTIES EXPORT_NAME ${ABSL_TARGET})
endforeach ()

export(TARGETS ${ABSL_TARGETS}
  NAMESPACE absl::
  FILE abseilConfig.cmake)

export(PACKAGE abseil)

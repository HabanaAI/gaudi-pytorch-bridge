#include <algorithm>
#include <memory>
#include <ostream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "synapse_helpers/habana_tensor.h"
#include "synapse_helpers/recipe.h"

namespace synapse_helpers {
class device;
class recipe;

class recipe_handle_cache {
 public:
  explicit recipe_handle_cache(device& device);
  recipe_handle_cache(const recipe_handle_cache&) = delete;
  recipe_handle_cache(recipe_handle_cache&&) = delete;
  recipe_handle_cache& operator=(const recipe_handle_cache&) = delete;
  recipe_handle_cache& operator=(recipe_handle_cache&&) = delete;

  ~recipe_handle_cache();

  std::shared_ptr<recipe> get_recipe(
      const size_t key,
      synapse_helpers::graph& graph);
  std::shared_ptr<recipe> get_recipe(const size_t key);
  void remove_recipe(const size_t key);
  bool isCached(size_t key);

 private:
  std::mutex mutex_;
  device& device_;
  absl::flat_hash_map<size_t, std::shared_ptr<recipe>> cache_map_;
};

bool IsCachingEnabled();
bool IsStreamSyncOptEnabled();
} // namespace synapse_helpers

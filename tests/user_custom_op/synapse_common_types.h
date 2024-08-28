/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#ifndef SYNAPSE_COMMON_TYPES_H
#define SYNAPSE_COMMON_TYPES_H

#include <stdint.h>

typedef struct internalTensor* synTensor;
typedef struct syn_cb_internal* synCommandBuffer;
typedef struct InternalWaitHandle* synWaitHandle;
struct synTensorDescriptor;
typedef struct synTensorDescriptor synTensorDescriptor;
struct synTensorNDimDescriptor;
typedef struct synTensorNDimDescriptor synTensorNDimDescriptor;
struct synDeviceInfo;
typedef struct synDeviceInfo synDeviceInfo;
struct synDeviceInfoV2;
typedef struct synDeviceInfoV2 synDeviceInfoV2;

#ifndef __cplusplus
#define size_t uint64_t
#endif

typedef uint64_t TSize;
typedef uint64_t TStride;
typedef int64_t TOffset;

typedef struct {
  uint64_t src;
  uint64_t dst;
  uint64_t size;
} internalMemcopyParamEntry;

#define MAX_NUM_OF_DEVICES_PER_HOST (8)

#define MAX_DIMENSIONS_NUM 5
#define MAX_CONV_DIMS 3

#define ENQUEUE_TENSOR_NAME_MAX_SIZE (1024)
#define MAX_LAYOUT_SIZE 20

#define STATUS_DESCRIPTION_MAX_SIZE (56)

static const uint32_t INVALID_BATCH_POS = 0xFFFFFFFF;
static const uint32_t INVALID_TENSOR_INDEX = 0xFFFFFFFF;
static const uint32_t INVALID_SECTION_ID = 0xFFFFFFFF;
static const uint64_t INVALID_CONST_SECTION_DATA = 0xFFFFFFFFFFFFFFFF;

#define SYN_FLAGS_TENSOR_NAME 0x1

#define TENSOR_INVALID_ID (0xFFFFFFFFFFFFFFFF)

#define GET_TENSOR_INFO(idx, type) ((idx) | (uint64_t)(type) << 56)

#define TENSOR_INFO_TO_TYPE(info) ((synTensorType)((info) >> 56))
#define TENSOR_INFO_TO_INDEX(info) ((info)&0xFFFFFFFFFFFFFF)
#define IS_TENSOR_INVALID(val) ((val) == TENSOR_INVALID_ID)

static const int DMA_MEMCPY_PARALLEL_LEVEL = 4;

static const uint64_t EXECUTION_BLOBS_CHUNK_SIZE_IN_BYTES =
    0; // if set to zero, chunk mechanism is inactive
extern uint64_t
    PATCHING_BLOBS_CHUNK_SIZE_IN_BYTES; // if set to zero, chunk mechanism is
                                        // inactive. Used also for work-dist
                                        // blobs

static const uint32_t SIG_HANDLE_INVALID = UINT32_MAX;

#define SYN_MAX_TENSOR_DIM 5
#define SYN_GAUDI_MAX_TENSOR_DIM 8
#define HABANA_DIM_MAX 25
// supported quantization data types in Goya and Greco - int4, uint4, int8,
// uint8, int16, uint16
static const unsigned int SYN_NUM_DATA_TYPES = 6;

typedef enum synStatus {
  synSuccess = 0,
  synInvalidArgument = 1,
  synCbFull = 2,
  synOutOfHostMemory = 3,
  synOutOfDeviceMemory = 4,
  synObjectAlreadyInitialized = 5,
  synObjectNotInitialized = 6,
  synCommandSubmissionFailure = 7,
  synNoDeviceFound = 8,
  synDeviceTypeMismatch = 9,
  synFailedToInitializeCb = 10,
  synFailedToFreeCb = 11,
  synFailedToMapCb = 12,
  synFailedToUnmapCb = 13,
  synFailedToAllocateDeviceMemory = 14,
  synFailedToFreeDeviceMemory = 15,
  synFailedNotEnoughDevicesFound = 16,
  synDeviceReset = 17,
  synUnsupported = 18,
  synWrongParamsFile = 19,
  synDeviceAlreadyAcquired = 20,
  synNameIsAlreadyUsed = 21,
  synBusy = 22,
  synAllResourcesTaken = 23,
  synUnavailable = 24,
  synInvalidTensorDimensions = 25,
  synFail = 26,
  synOutOfResources = 27,
  synUninitialized = 28,
  synAlreadyInitialized = 29,
  synFailedSectionValidation = 30,
  synSynapseTerminated = 31,
  synAssertAsync = 32,
  synInvalidEventHandle = 33,
  synMappingNotFound = 34,
  synFailedDynamicPatching = 35,
  synFailedStaticPatching = 36,
  synFailedToSubmitWorkload = 37,
  synInvalidSectionsDefinition = 38,
  synInvalidTensorProperties = 39,
  synFailHccl = 40,
  synFailedToCollectTime = 41,
  synTimeout = 42,
  synResourceBadUsage = 43,
  // Always last
  synStatusLast = synResourceBadUsage
} synStatus;

typedef enum synDmaDir {
  HOST_TO_DRAM,
  DRAM_TO_HOST,
  DRAM_TO_DRAM,
  DIRECTION_ENUM_MAX
} synDmaDir;

typedef enum synMemFlags { synMemHost = 0x1, synMemDevice = 0x2 } synMemFlags;

typedef enum synDeviceType {
  synDeviceGoya2 =
      1, // todo: remove once we make sure all other repos don't use it
  synDeviceGreco = synDeviceGoya2,
  synDeviceGaudi,
  synDeviceGaudi2 = 4,
  synDeviceGaudi3,
  synDeviceEmulator,
  synDeviceTypeInvalid,
  synDeviceTypeSize
} synDeviceType;

typedef enum synDataType {
  syn_type_na = 0, // invalid
  syn_type_fixed = 1 << 0, // 8-bit integer
  syn_type_int8 = syn_type_fixed, // alias to syn_type_fixed
  syn_type_bf16 =
      1 << 1, // 16-bit float- 8 bits exponent, 7 bits mantisa, 1 bit sign
  syn_type_single = 1 << 2, // 32-bit floating point, IEEE compliant
  syn_type_float = syn_type_single, // alias to syn_type_single, IEEE compliant
  syn_type_int16 = 1 << 3, // 16-bit integer
  syn_type_int32 = 1 << 4, // 32-bit integer
  syn_type_uint8 = 1 << 5, // 8-bit unsigned integer
  syn_type_int4 = 1 << 6, // 4-bit signed integer
  syn_type_uint4 = 1 << 7, // 4-bit unsigned integer
  syn_type_fp16 = 1 << 8, // 16-bit floating point
  syn_type_uint16 = 1 << 9, // 16-bit unsigned integer
  syn_type_uint32 = 1 << 10, // 32-bit unsigned integer
  syn_type_tf32 = 1 << 11, // 19-bit floating point (a.k.a. FP19)
  syn_type_hb_float = 1 << 12, // 32-bit floating point, not compliant with IEEE
                               // (2x faster than IEEE)
  syn_type_fp8_143 = 1 << 13, // 8-bit floating point (1 signed bit, 4 exponent
                              // bits, 3 mantissa bits)
  syn_type_fp8_152 = 1 << 14, // 8-bit floating point (1 signed bit, 5 exponent
                              // bits, 2 mantissa bits)
  syn_type_int64 = 1 << 15, // 64-bit signed integer
  syn_type_uint64 = 1 << 16, // 64-bit unsigned integer
  syn_type_ufp16 = 1 << 17, // 16-bit unsigned floating point

  // Must be last and this enum must be in ascending order!
  syn_type_max
} synDataType;

typedef enum {
  DATA_TENSOR = 0,
  SHAPE_TENSOR,
  OUTPUT_DESCRIBING_SHAPE_TENSOR = SHAPE_TENSOR,
  INPUT_DESCRIBING_SHAPE_TENSOR_DEPRECATED,
  DATA_TENSOR_DYNAMIC,
  DEVICE_SHAPE_TENSOR,
  HOST_SHAPE_TENSOR,
  HOST_TO_DEVICE_TENSOR,
  TENSOR_TYPE_MAX,
  TENSOR_TYPE_INVALID = TENSOR_TYPE_MAX
} synTensorType;

enum synConvParamIndex {
  // Kernel
  CONV_KERNEL_WIDTH = 0,
  CONV_KERNEL_HEIGHT = 1,
  CONV_KERNEL_DEPTH = 2,
  CONV_KERNEL_SIZE = 3,

  // Stride
  CONV_STRIDE_WIDTH = 0,
  CONV_STRIDE_HEIGHT = 1,
  CONV_STRIDE_DEPTH = 2,
  CONV_STRIDE_SIZE = 3,

  // Padding
  CONV_PAD_LEFT = 0,
  CONV_PAD_RIGHT = 1,
  CONV_PAD_TOP = 2,
  CONV_PAD_BOTTOM = 3,
  CONV_PAD_FRONT = 4,
  CONV_PAD_BACK = 5,
  CONV_PAD_SIZE = 6,

  // Dilation
  CONV_DIL_WIDTH = 0,
  CONV_DIL_HEIGHT = 1,
  CONV_DIL_DEPTH = 2,
  CONV_DIL_SIZE = 3,
};

enum TransposePermutationDim {
  // Assume basic permutation is BDHWC
  TBD_4DimSize = 4,
  TBD_5DimSize = 5,
  TBD_6DimSize = 6,
  TBD_7DimSize = 7,

  TPD_Batch = 4,
  TPD_4Dim_Batch = 3,
  TPD_Depth = 3,
  TPD_Height = 2,
  TPD_Width = 1,
  TPD_Channel = 0,

  // Assume basic permutation is QRSCK
  TPD_Weights_Q = 4,
  TPD_Weights_R = 3,
  TPD_Weights_S = 2,
  TPD_Weights_C = 1,
  TPD_Weights_K = 0
};

enum synDevicePllIndex {
  PLL_CPU = 0,
  PLL_IC,
  PLL_MC,
  PLL_MME,
  PLL_PCI,
  PLL_EMMC,
  PLL_TPC,
  PLL_SIZE
};

struct synInternalQueue {
  uint32_t queueIndex;
  uint32_t size;
  uint64_t address;
};

typedef struct synAxisParams {
  unsigned axis;
} synAxisParams;

typedef synAxisParams synFlattenParams;
typedef synAxisParams synConcatenateParams;
typedef synAxisParams synSplitParams;
typedef synAxisParams synExpandDimsParams;
typedef synAxisParams synSqueezeParams;

struct synSliceAxisParams {
  unsigned axis;
  unsigned begin;
  unsigned end;
};

struct synSliceParams {
  unsigned axes[MAX_DIMENSIONS_NUM];
  unsigned starts[MAX_DIMENSIONS_NUM];
  unsigned ends[MAX_DIMENSIONS_NUM];
  unsigned steps[MAX_DIMENSIONS_NUM];
};

struct synSliceParamsNDims {
  unsigned axes[HABANA_DIM_MAX];
  unsigned starts[HABANA_DIM_MAX];
  unsigned ends[HABANA_DIM_MAX];
  unsigned steps[HABANA_DIM_MAX];
};

struct synNMSParams {
  float scoreTh;
  float iouTh;
  unsigned maxOutSize;
};

struct synTfBatchNormalizationParams {
  float variance_epsilon;
};

struct synAssertAsyncParams {
  uint64_t msg_id;
};

typedef enum {
  synTensorPropUnknown = 0,
  synTensorPropSection = 1 << 0,
  synTensorPropHostPtr = 1 << 1,
  synTensorPropQuantMetadata = 1 << 2,
  synTensorPropDynamicRange = 1 << 3,
  synTensorPropFlags = 1 << 4,
  synTensorPropDeviceLayout = 1 << 5,
  synTensorPropDeviceDataType = synTensorPropDeviceLayout,
  synTensorPropName = 1 << 6,
  synTensorPropSectionOffset = 1 << 7,
  synTensorPropGeometryMin = 1 << 8,
  synTensorPropGeometryMax = 1 << 9,
  synTensorPropGeometryDim = 1 << 10,
  synTensorPropType = 1 << 11,
  synTensorPropPermutation = 1 << 12,
  synTensorPropFpQuantMetadata = 1 << 13,
  synTensorInternalNoProducer = 1 << 14,
  synTensorPropPCDynamicRange = 1 << 15
} synTensorProperty;

typedef enum synRoundingMode {

  synRoundToNearest,
  synRoundToZero,
  synRoundUp,
  synRoundDown,
  synStochasticRounding,
  synRoundAwayFromZero,
  synStochasticRoundingAndNearest

} synRoundingMode;

typedef struct TensorMetadataInfo {
  const char* tensorName;

  uint32_t elementType;
  double zp;
  double scale;
  uint32_t dimensions;
  TSize dimensionsSize[HABANA_DIM_MAX];
  // Not related to this DB. Will be removed on a following commit,
  // so current commit will not break Python-Synapse pair
  uint64_t roiSizeInBytes;
  uint16_t batchSize;
  uint8_t isInput; // TODO currently uint8_t because of python test issues
  char layout[MAX_LAYOUT_SIZE]; // Only relevant for inference
  uint32_t sectionId;
  uint64_t offsetInSection; // offsets in bytes
} TensorMetadataInfo;

typedef struct TensorMetadataInfoExt {
  const char* tensorName;

  uint32_t elementType;
  double zp;
  double scale;
  uint32_t dimensions;
  uint64_t dimensionsSize[HABANA_DIM_MAX];
  // Not related to this DB. Will be removed on a following commit,
  // so current commit will not break Python-Synapse pair
  uint64_t roiSizeInBytes;
  uint16_t batchSize;
  uint8_t isInput; // TODO currently uint8_t because of python test issues
  char layout[MAX_LAYOUT_SIZE]; // Only relevant for inference
  uint32_t sectionId;
  uint64_t offsetInSection; // offsets in bytes
} TensorMetadataInfoExt;

typedef enum synTraceType {
  synTraceAll = 0x1,
  synTraceHost,
  synTraceDevice,
  synTraceTypeSize
} synTraceType;

typedef enum synTraceFormat {
  synTraceFormatTEF = 0x1,
  synTraceFormatSize
} synTraceFormat;

typedef struct {
  const char* key;
  enum ValueType { TYPE_UINT64, TYPE_CHAR_PTR, TYPE_DOUBLE } type;
  union {
    const char* str;
    uint64_t u64;
    double d;
  } value;
} synTraceEventArg;

typedef struct synTraceEventExtraArgs {
  uint8_t count;
  const synTraceEventArg* args;
} synTraceEventExtraArgs;

typedef struct synTraceEvent {
  const char* name;
  const char* category; /* @SerializedName("cat") */
  char type; /* @SerializedName("ph")  */
  long double timestamp; /* @SerializedName("ts")  */
  long double duration; /* @SerializedName("dur") */
  uint32_t engineType; /* @SerializedName("pid") */
  uint32_t engineIndex; /* @SerializedName("tid") */
  uint64_t contextId; /* @SerializedName("id")  */

  union /* @SerializedName("args") */
  {
    struct /* event args */
    {
      const char* dataType; /* @SerializedName("dtype") */
      const char* operation; /* @SerializedName("op")    */
      uint16_t recipeId; /* @SerializedName("recipe id")   */
      const char* recipeName; /* @SerializedName("graph name")   */
      uint64_t streamHandle; /* @SerializedName("streamHandle")   */
      uint64_t eventHandle; /* @SerializedName("eventHandle")   */

      synTraceEventExtraArgs extraArgs;
    };

    struct /* counter args */
    {
      double minimum; /* @SerializedName("Min")   */
      double maximum; /* @SerializedName("Max")   */
      double average; /* @SerializedName("Avg")   */
      uint16_t matches; /* @SerializedName("Match") */
    };

    struct /* metadata args */
    {
      const char* name; /* @SerializedName("name") */
      const char* group; /* @SerializedName("level") */
    };

    struct /* reserved */
    {
      char reserved[128];
    };

    /* spmu value */
    uint64_t value;
  } arguments;
} synTraceEvent;

struct synDeviceInfo {
  uint64_t sramBaseAddress;
  uint64_t dramBaseAddress;
  uint32_t sramSize;
  uint64_t dramSize;
  uint64_t tpcEnabledMask;
  uint8_t dramEnabled;
  uint32_t deviceId;
  uint32_t fd;
  synDeviceType deviceType;
};

struct synDeviceInfoV2 {
  uint64_t sramBaseAddress;
  uint64_t dramBaseAddress;
  uint32_t sramSize;
  uint64_t dramSize;
  uint64_t tpcEnabledMask;
  uint8_t dramEnabled;
  uint32_t deviceId;
  uint32_t fd;
  synDeviceType deviceType;
  uint8_t deviceIndex;
  uint64_t reserved;
};

typedef enum {
  synStreamPriorityHigh = 0,
  synStreamPriorityLow,
  synStreamPriorityNotSet
} synStreamPriority;

#ifdef __cplusplus
#include "synapse_common_types.hpp"
#endif
#endif /*SYNAPSE_COMMON_TYPES_H*/

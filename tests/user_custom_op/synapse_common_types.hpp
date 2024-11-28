/**
* Copyright (c) 2021-2024 Intel Corporation
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

#pragma once

#include "synapse_common_types.h"

#include <cstring>
#include <string>

class CommandSubmission;
// To use POD in struct
#define MAX_EINSUM_EQUATION_LENGTH 50
#define EXP_BIAS_143_DEFAULT 7
static const char INVALID_EINSUM_EQUATION[] = "invalid";
static const size_t TENSOR_DEFAULT_MIN_SIZE = 0;

struct synActivationParams {
  synActivationParams()
      : reluEnable(false),
        resAfterPWL(false),
        numChannels(1),
        pwlNegCoeff(0),
        pwlPosCoeff(0) {}
  bool reluEnable; // enable PWL (param name kept for compatibility with Goya-1)
  bool resAfterPWL; // add residual(cin) before PWL function (e.g. relu/prelu)
  uint16_t structPadding = 0;
  uint32_t numChannels; // number of channels, or 1 for global coefficients
  // Below coeff are unsued fields
  uint64_t pwlNegCoeff; // coefficients for the negative PWL - according to
                        // numChannels
  uint64_t pwlPosCoeff; // coefficients for the positive PWL - according to
                        // numChannels

  synActivationParams(const synActivationParams& params) {
    *this = params;
  }

  synActivationParams& operator=(const synActivationParams& params) {
    if (this != &params) {
      reluEnable = params.reluEnable;
      resAfterPWL = params.resAfterPWL;
      numChannels = params.numChannels;
      pwlNegCoeff = 0;
      pwlPosCoeff = 0;
    }
    return *this;
  }
};

enum synConvolutionPaddingType {
  /* PADDING_VALID -- not needed, same as PADDING_EXPLICIT with all padding
     values zeros */
  PADDING_EXPLICIT,
  PADDING_SAME
};

struct synConvolutionParams {
  synConvolutionParams()
      : kW(1),
        kH(1),
        dW(1),
        dH(1),
        padL(0),
        padR(0),
        padT(0),
        padB(0),
        dilW(1),
        dilH(1),
        activation(),
        nGroups(1) {} // default values

  synConvolutionParams(
      unsigned _kW,
      unsigned _kH,
      unsigned _dW,
      unsigned _dH,
      unsigned _padL,
      unsigned _padR,
      unsigned _padT,
      unsigned _padB,
      unsigned _dilW,
      unsigned _dilH)
      : kW(_kW),
        kH(_kH),
        dW(_dW),
        dH(_dH),
        padL(_padL),
        padR(_padR),
        padT(_padT),
        padB(_padB),
        dilW(_dilW),
        dilH(_dilH),
        activation(),
        nGroups(1) {} // default values

  synConvolutionParams(const synConvolutionParams& params) {
    *this = params;
  }

  synConvolutionParams& operator=(const synConvolutionParams& params) {
    if (this != &params) {
      kW = params.kW;
      kH = params.kH;
      dW = params.dW;
      dH = params.dH;
      padL = params.padL;
      padR = params.padR;
      padT = params.padT;
      padB = params.padB;
      dilW = params.dilW;
      dilH = params.dilH;
      activation = params.activation;
      nGroups = params.nGroups;
    }
    return *this;
  }

  int getPadL() const {
    return padL;
  }
  int getPadR() const {
    return padR;
  }
  void setPadL(int _padL) {
    padL = _padL;
  }
  void setPadR(int _padR) {
    padR = _padR;
  }

  int getPadT() const {
    return padT;
  }
  int getPadB() const {
    return padB;
  }
  void setPadT(int _padT) {
    padT = _padT;
  }
  void setPadB(int _padB) {
    padB = _padB;
  }

  // Kernel
  unsigned kW;
  unsigned kH;
  // Stride
  unsigned dW; // stride width
  unsigned dH; // stride height
  // Padding
  int padL;
  int padR;
  int padT;
  int padB;
  // Dilation
  unsigned dilW;
  unsigned dilH;
  // Activation params
  struct synActivationParams activation;

  // Number of convolution groups, 1 means regular convolution
  unsigned nGroups;
  uint32_t structPadding1 = 0;
};

struct synConvolutionParamsV2 : synConvolutionParams {
  synConvolutionPaddingType paddingType;
  unsigned reserved[9]{};

  synConvolutionParamsV2()
      : synConvolutionParams(), paddingType(PADDING_EXPLICIT) {}

  synConvolutionParamsV2(
      unsigned _kW,
      unsigned _kH,
      unsigned _dW,
      unsigned _dH,
      unsigned _padL,
      unsigned _padR,
      unsigned _padT,
      unsigned _padB,
      unsigned _dilW,
      unsigned _dilH,
      synConvolutionPaddingType _paddingType)
      : synConvolutionParams(
            _kW,
            _kH,
            _dW,
            _dH,
            _padL,
            _padR,
            _padT,
            _padB,
            _dilW,
            _dilH),
        paddingType(_paddingType) {}
  synConvolutionParamsV2(const synConvolutionParams& v1)
      : synConvolutionParams(v1), paddingType(PADDING_EXPLICIT) {}

  synConvolutionParamsV2(const synConvolutionParamsV2& params)
      : synConvolutionParams(params) {
    *this = params;
  }

  synConvolutionParamsV2& operator=(const synConvolutionParamsV2& params) {
    if (this != &params) {
      synConvolutionParams::operator=(params);
      paddingType = params.paddingType;
    }
    return *this;
  }
};

struct synDeformableConvParams : public synConvolutionParams {
  synDeformableConvParams()
      : synConvolutionParams(), nDeformableGroups(1) {} // default values

  synDeformableConvParams(
      unsigned _kW,
      unsigned _kH,
      unsigned _dW,
      unsigned _dH,
      unsigned _padL,
      unsigned _padR,
      unsigned _padT,
      unsigned _padB,
      unsigned _dilW,
      unsigned _dilH,
      unsigned _nDeformableGroups)
      : synConvolutionParams(
            _kW,
            _kH,
            _dW,
            _dH,
            _padL,
            _padR,
            _padT,
            _padB,
            _dilW,
            _dilH),
        nDeformableGroups(_nDeformableGroups) {}

  synDeformableConvParams(const synDeformableConvParams& params)
      : synConvolutionParams(params) {
    *this = params;
  }

  synDeformableConvParams& operator=(const synDeformableConvParams& params) {
    if (this != &params) {
      synConvolutionParams::operator=(params);
      nDeformableGroups = params.nDeformableGroups;
    }
    return *this;
  }

  // Number of deformable convolution groups
  unsigned nDeformableGroups;
  uint32_t _structPadding2 = 0;
};

struct synDeformableConvParamsV2 : public synDeformableConvParams {
  synConvolutionPaddingType paddingType;
  unsigned reserved[9]{};

  synDeformableConvParamsV2()
      : synDeformableConvParams(), paddingType(PADDING_EXPLICIT) {}

  synDeformableConvParamsV2(
      unsigned _kW,
      unsigned _kH,
      unsigned _dW,
      unsigned _dH,
      unsigned _padL,
      unsigned _padR,
      unsigned _padT,
      unsigned _padB,
      unsigned _dilW,
      unsigned _dilH,
      unsigned _nDeformableGroups,
      synConvolutionPaddingType _paddingType)
      : synDeformableConvParams(
            _kW,
            _kH,
            _dW,
            _dH,
            _padL,
            _padR,
            _padT,
            _padB,
            _dilW,
            _dilH,
            _nDeformableGroups),
        paddingType(_paddingType) {}
  synDeformableConvParamsV2(const synDeformableConvParams& v1)
      : synDeformableConvParams(v1), paddingType(PADDING_EXPLICIT) {}

  synDeformableConvParamsV2(const synDeformableConvParamsV2& params)
      : synDeformableConvParams(params) {
    *this = params;
  }

  synDeformableConvParamsV2& operator=(
      const synDeformableConvParamsV2& params) {
    if (this != &params) {
      synDeformableConvParams::operator=(params);
      paddingType = params.paddingType;
    }
    return *this;
  }
};

struct synConvolution3DParams {
  // Kernel, stride. padding and dilation
  unsigned kernel[CONV_KERNEL_SIZE]{}; // CONV_KERNEL_WIDTH, CONV_KERNEL_HEIGHT
                                       // , CONV_KERNEL_DEPTH
  unsigned stride[CONV_STRIDE_SIZE]{}; // CONV_STRIDE_WIDTH, CONV_STRIDE_HEIGHT,
                                       // CONV_STRIDE_DEPTH
  int padding[CONV_PAD_SIZE]{}; // CONV_PAD_LEFT, CONV_PAD_RIGHT, CONV_PAD_TOP,
                                // CONV_PAD_BOTTOM,
  // CONV_PAD_FRONT, CONV_PAD_BACK
  unsigned dilation[CONV_DIL_SIZE]{}; // CONV_DIL_WIDTH, CONV_DIL_HEIGHT,
                                      // CONV_DIL_DEPTH
  uint32_t structPadding1 = 0;

  // Activation params
  struct synActivationParams activation;

  // Number of convolution groups, 1 means regular convolution
  unsigned nGroups;
  uint32_t structPadding2 = 0;

  synConvolution3DParams() {
    kernel[CONV_KERNEL_WIDTH] = 1;
    kernel[CONV_KERNEL_HEIGHT] = 1;
    kernel[CONV_KERNEL_DEPTH] = 1;
    stride[CONV_STRIDE_WIDTH] = 1;
    stride[CONV_STRIDE_HEIGHT] = 1;
    stride[CONV_STRIDE_DEPTH] = 1;
    padding[CONV_PAD_LEFT] = 0;
    padding[CONV_PAD_RIGHT] = 0;
    padding[CONV_PAD_TOP] = 0;
    padding[CONV_PAD_BOTTOM] = 0;
    padding[CONV_PAD_FRONT] = 0;
    padding[CONV_PAD_BACK] = 0;
    dilation[CONV_DIL_WIDTH] = 1;
    dilation[CONV_DIL_HEIGHT] = 1;
    dilation[CONV_DIL_DEPTH] = 1;
    nGroups = 1;
  } // default values

  synConvolution3DParams(
      unsigned _kW,
      unsigned _kH,
      unsigned _kD,
      unsigned _dW,
      unsigned _dH,
      unsigned _dD,
      int _padLeft,
      int _padRight,
      int _padTop,
      int _padBottom,
      int _padFront,
      int _padBack,
      unsigned _dilW,
      unsigned _dilH,
      unsigned _dilD) {
    kernel[CONV_KERNEL_WIDTH] = _kW;
    kernel[CONV_KERNEL_HEIGHT] = _kH;
    kernel[CONV_KERNEL_DEPTH] = _kD;
    stride[CONV_STRIDE_WIDTH] = _dW;
    stride[CONV_STRIDE_HEIGHT] = _dH;
    stride[CONV_STRIDE_DEPTH] = _dD;
    padding[CONV_PAD_LEFT] = _padLeft;
    padding[CONV_PAD_RIGHT] = _padRight;
    padding[CONV_PAD_TOP] = _padTop;
    padding[CONV_PAD_BOTTOM] = _padBottom;
    padding[CONV_PAD_FRONT] = _padFront;
    padding[CONV_PAD_BACK] = _padBack;
    dilation[CONV_DIL_WIDTH] = _dilW;
    dilation[CONV_DIL_HEIGHT] = _dilH;
    dilation[CONV_DIL_DEPTH] = _dilD;
    nGroups = 1;
  } // default values

  synConvolution3DParams(const synConvolutionParams& userConvParam) {
    kernel[CONV_KERNEL_WIDTH] = userConvParam.kW;
    kernel[CONV_KERNEL_HEIGHT] = userConvParam.kH;
    kernel[CONV_KERNEL_DEPTH] = 1;

    stride[CONV_STRIDE_WIDTH] = userConvParam.dW;
    stride[CONV_STRIDE_HEIGHT] = userConvParam.dH;
    stride[CONV_STRIDE_DEPTH] = 1;

    padding[CONV_PAD_LEFT] = userConvParam.padL;
    padding[CONV_PAD_RIGHT] = userConvParam.padR;
    padding[CONV_PAD_TOP] = userConvParam.padT;
    padding[CONV_PAD_BOTTOM] = userConvParam.padB;
    padding[CONV_PAD_FRONT] = 0;
    padding[CONV_PAD_BACK] = 0;

    dilation[CONV_DIL_WIDTH] = userConvParam.dilW;
    dilation[CONV_DIL_HEIGHT] = userConvParam.dilH;
    dilation[CONV_DIL_DEPTH] = 1;

    activation = userConvParam.activation;
    nGroups = userConvParam.nGroups;
  }

  synConvolution3DParams(const synConvolution3DParams& params) {
    *this = params;
  }

  synConvolution3DParams& operator=(const synConvolution3DParams& params) {
    if (this != &params) {
      kernel[CONV_KERNEL_WIDTH] = params.kernel[CONV_KERNEL_WIDTH];
      kernel[CONV_KERNEL_HEIGHT] = params.kernel[CONV_KERNEL_HEIGHT];
      kernel[CONV_KERNEL_DEPTH] = params.kernel[CONV_KERNEL_DEPTH];
      stride[CONV_STRIDE_WIDTH] = params.stride[CONV_STRIDE_WIDTH];
      stride[CONV_STRIDE_HEIGHT] = params.stride[CONV_STRIDE_HEIGHT];
      stride[CONV_STRIDE_DEPTH] = params.stride[CONV_STRIDE_DEPTH];
      padding[CONV_PAD_LEFT] = params.padding[CONV_PAD_LEFT];
      padding[CONV_PAD_RIGHT] = params.padding[CONV_PAD_RIGHT];
      padding[CONV_PAD_TOP] = params.padding[CONV_PAD_TOP];
      padding[CONV_PAD_BOTTOM] = params.padding[CONV_PAD_BOTTOM];
      padding[CONV_PAD_FRONT] = params.padding[CONV_PAD_FRONT];
      padding[CONV_PAD_BACK] = params.padding[CONV_PAD_BACK];
      dilation[CONV_DIL_WIDTH] = params.dilation[CONV_DIL_WIDTH];
      dilation[CONV_DIL_HEIGHT] = params.dilation[CONV_DIL_HEIGHT];
      dilation[CONV_DIL_DEPTH] = params.dilation[CONV_DIL_DEPTH];
      activation = params.activation;
      nGroups = params.nGroups;
    }
    return *this;
  }
};

struct synConvolution3DParamsV2 : synConvolution3DParams {
  synConvolutionPaddingType paddingType;
  unsigned reserved[9]{};

  synConvolution3DParamsV2()
      : synConvolution3DParams(), paddingType(PADDING_EXPLICIT) {}
  synConvolution3DParamsV2(
      unsigned _kW,
      unsigned _kH,
      unsigned _kD,
      unsigned _dW,
      unsigned _dH,
      unsigned _dD,
      int _padLeft,
      int _padRight,
      int _padTop,
      int _padBottom,
      int _padFront,
      int _padBack,
      unsigned _dilW,
      unsigned _dilH,
      unsigned _dilD,
      synConvolutionPaddingType _paddingType)
      : synConvolution3DParams(
            _kW,
            _kH,
            _kD,
            _dW,
            _dH,
            _dD,
            _padLeft,
            _padRight,
            _padTop,
            _padBottom,
            _padFront,
            _padBack,
            _dilW,
            _dilH,
            _dilD),
        paddingType(_paddingType) {}

  synConvolution3DParamsV2(
      const synConvolutionParams& userConvParam,
      synConvolutionPaddingType _paddingType)
      : synConvolution3DParams(userConvParam), paddingType(_paddingType) {}

  synConvolution3DParamsV2(const synConvolutionParamsV2& userConvParam)
      : synConvolution3DParams(userConvParam),
        paddingType(userConvParam.paddingType) {}

  synConvolution3DParamsV2(const synConvolution3DParams& v1)
      : synConvolution3DParams(v1), paddingType(PADDING_EXPLICIT) {}

  synConvolution3DParamsV2(const synConvolution3DParamsV2& params)
      : synConvolution3DParams(params) {
    *this = params;
  }

  synConvolution3DParamsV2& operator=(const synConvolution3DParamsV2& params) {
    if (this != &params) {
      synConvolution3DParams::operator=(params);
      paddingType = params.paddingType;
    }
    return *this;
  }
};

enum class synRotateRelMode : uint8_t { ABSOLUTE = 0, RELATIVE = 1 };

enum class synRotateMeshFormat : uint8_t {
  RESERVED0 = 0,
  FLEX = 1,
  RESERVED2 = 2,
  RESERVED3 = 3,
  FP32 = 4
};

enum class synRotateMeshOrder : uint8_t {
  PRE_DISTORTION = 0,
  POST_DISTORTION = 1
};

enum class synRotateMeshDataType : uint8_t {
  INT8 = 0,
  INT16 = 1,
  FP16 = 2,
  BF16 = 3,
  FP32 = 4
};

enum class synRotateMeshMode : uint8_t {
  ROTATION = 0,
  AFFINE = 1,
  PROJECTION = 2,
  DISTORTION = 3
};

enum class synRotateMode : uint8_t {
  ROTATION = 0,
  AFFINE = 1,
  PROJECTION = 2,
  MESH = 3,
  RESAMPLE_FWD = 4,
  RESAMPLE_BWD1 = 5,
  RESAMPLE_BWD2 = 6,
  RESCALE = 7,
  BILINEAR_GRAD = 8
};

enum class synRotateInterpolationMode : uint8_t {
  ROT_BILINEAR = 0,
  ROT_NEAREST_NEIGHBOR = 1,
  ROT_LANCZOS2 = 2,
  ROT_LANCZOS3 = 3,
  ROT_BICUBIC = 4
};

enum class synRotateCoordinateMode : uint8_t {
  FIXED_POINT = 0,
  FLOATING_POINT = 1
};

struct synRotateParams {
  synRotateParams() = default;

  // rotation
  synRotateParams(
      float angle,
      uint32_t input_center_X,
      uint32_t input_center_Y,
      uint32_t output_center_X,
      uint32_t output_center_Y,
      uint8_t background)
      : m_angle(angle),
        m_inputCenterX(input_center_X),
        m_inputCenterY(input_center_Y),
        m_outputCenterX(output_center_X),
        m_outputCenterY(output_center_Y),
        m_background(background) {}

  // rotation with output dims
  synRotateParams(
      float angle,
      uint32_t input_center_X,
      uint32_t input_center_Y,
      uint32_t output_center_X,
      uint32_t output_center_Y,
      uint8_t background,
      uint32_t out_w,
      uint32_t out_h)
      : m_angle(angle),
        m_inputCenterX(input_center_X),
        m_inputCenterY(input_center_Y),
        m_outputCenterX(output_center_X),
        m_outputCenterY(output_center_Y),
        m_background(background),
        m_out_width(out_w),
        m_out_height(out_h) {}

  // debug info
  synRotateParams(
      float angle,
      uint32_t input_center_X,
      uint32_t input_center_Y,
      uint32_t output_center_X,
      uint32_t output_center_Y,
      uint8_t background,
      bool isDumpDescriptors,
      std::string descFilePrefix)
      : m_angle(angle),
        m_inputCenterX(input_center_X),
        m_inputCenterY(input_center_Y),
        m_outputCenterX(output_center_X),
        m_outputCenterY(output_center_Y),
        m_background(background),
        m_isDumpDescriptors(isDumpDescriptors),
        m_descFilePrefix(descFilePrefix) {}

  float m_angle = 0;
  uint32_t m_inputCenterX = 0;
  uint32_t m_inputCenterY = 0;
  uint32_t m_outputCenterX = 0;
  uint32_t m_outputCenterY = 0;
  uint8_t m_background = 0;

  synRotateMode m_rotation_mode = synRotateMode::ROTATION;
  synRotateInterpolationMode m_interpolation_mode =
      synRotateInterpolationMode::ROT_BILINEAR;

  uint32_t m_out_width = 0;
  uint32_t m_out_height = 0;
  bool m_preserve_aspect_ratio = false;
  bool m_antialias = false;

  // mesh highlevel params
  synRotateMeshFormat m_mesh_format = synRotateMeshFormat::FLEX;
  synRotateRelMode m_mesh_rel_mode = synRotateRelMode::ABSOLUTE;
  synRotateMeshMode m_mesh_mode = synRotateMeshMode::ROTATION;
  synRotateMeshOrder m_mesh_order = synRotateMeshOrder::PRE_DISTORTION;
  float m_mesh_distortion_x = 0;
  float m_mesh_distortion_y = 0;
  float m_mesh_distortion_r = 0;
  float m_mesh_Sh = 0;
  float m_mesh_Sv = 0;
  synRotateMeshDataType m_mesh_datatype = synRotateMeshDataType::INT8;

  // For debug
  bool m_isDumpDescriptors = false;
  uint16_t m_structPad = 0;
  std::string m_descFilePrefix = "";
};

struct synWaitParams {
  synWaitParams(unsigned waitCycles) : waitCycles(waitCycles) {}

  unsigned waitCycles;
};

struct synGEMMParams {
  synGEMMParams(bool transpose_a = false, bool transpose_b = false)
      : transpose_a(transpose_a), transpose_b(transpose_b) {}

  bool transpose_a;
  bool transpose_b;
};

struct synFCParams {
  // Activation params
  struct synActivationParams activation;
};

struct synTransposeParams {
  TransposePermutationDim permutation[MAX_DIMENSIONS_NUM];
  unsigned int tensorDim;
};

struct synInferMaxParams {
  bool shouldInferMax[HABANA_DIM_MAX];
};

struct synTransposeParamsNDims {
  unsigned int tensorDim;
  uint32_t permutation[HABANA_DIM_MAX];
};

struct synDepthSpaceParams {
  unsigned blockSize;
  unsigned mode = 0;
};

class synQuantizationParams {
 public:
  synQuantizationParams()
      : m_zp(0),
        m_scale(1.),
        m_expBias(EXP_BIAS_143_DEFAULT),
        m_qDataType(syn_type_na) {}

  synQuantizationParams(synDataType dataType)
      : m_zp(0),
        m_scale(1.),
        m_expBias(EXP_BIAS_143_DEFAULT),
        m_qDataType(dataType) {}

  synQuantizationParams(
      double zpVal,
      double scaleVal,
      synDataType dataType = syn_type_na)
      : m_zp(zpVal),
        m_scale(scaleVal),
        m_expBias(EXP_BIAS_143_DEFAULT),
        m_qDataType(dataType) {}

  bool operator==(const synQuantizationParams& other) const {
    return (
        this->m_zp == other.m_zp && this->m_qDataType == other.m_qDataType &&
        this->m_scale == other.m_scale && this->m_expBias == other.m_expBias);
  }

  double m_zp;
  double m_scale;
  unsigned m_expBias;
  synDataType m_qDataType;
  uint32_t m_structPad = 0;
};

class synPerChannelQuantizationParams {
 public:
  synPerChannelQuantizationParams()
      : m_pcZps(nullptr),
        m_pcScales(nullptr),
        m_pcExpBias(nullptr),
        m_numChannels(0),
        m_qDataType(syn_type_na) {}

  bool operator==(const synPerChannelQuantizationParams& other) const {
    if (this->m_qDataType != other.m_qDataType ||
        this->m_numChannels != other.m_numChannels) {
      return false;
    }

    for (unsigned i = 0; i < m_numChannels; i++) {
      if (this->m_pcZps[i] != other.m_pcZps[i] ||
          this->m_pcScales[i] != other.m_pcScales[i] ||
          this->m_pcExpBias[i] != other.m_pcExpBias[i]) {
        return false;
      }
    }

    return true;
  }

  double* m_pcZps;
  double* m_pcScales;
  unsigned* m_pcExpBias;
  unsigned m_numChannels;
  synDataType m_qDataType;
};

struct DynamicRange {
  double min = 2;
  double max = 1;
  bool isSet = false;
};

struct synTensorDescriptor {
 public:
  synTensorDescriptor()
      : m_batchPos(INVALID_BATCH_POS),
        m_dataType(syn_type_na),
        m_dims(0),
        m_ptr(nullptr),
        m_isWeights(false),
        m_isQuantized(false),
        m_name(nullptr),
        m_isOutput(false),
        m_enablePerChannelQuant(false),
        m_isInput(false),
        m_isStatic(false),
        m_tensorType(DATA_TENSOR),
        m_isSparsityWeights(false) {
    memset(m_sizes, 0, SYN_MAX_TENSOR_DIM * sizeof(unsigned));
    memset(m_strides, 0, SYN_MAX_TENSOR_DIM * sizeof(unsigned));
    memset(
        m_minSizes,
        TENSOR_DEFAULT_MIN_SIZE,
        SYN_MAX_TENSOR_DIM * sizeof(unsigned));
  }

  synTensorDescriptor(
      synDataType dataType,
      unsigned dims,
      unsigned sizes[SYN_MAX_TENSOR_DIM],
      void* ptr,
      bool isWeights = false,
      const char* name = nullptr,
      unsigned batchPos = INVALID_BATCH_POS,
      bool isQuantized = false,
      bool enablePerChannelQuant = false,
      bool setDynamicRange = false,
      double DynamicRangeMin = 2,
      double DynamicRangeMax = 1,
      bool isSparsityWeights = false)
      : m_batchPos(batchPos),
        m_dataType(dataType),
        m_dims(dims),
        m_ptr(ptr),
        m_isWeights(isWeights),
        m_isQuantized(isQuantized),
        m_name(name),
        m_isOutput(false),
        m_enablePerChannelQuant(enablePerChannelQuant),
        m_isInput(false),
        m_isStatic(false),
        m_tensorType(DATA_TENSOR),
        m_isSparsityWeights(isSparsityWeights) {
    memcpy(m_sizes, sizes, dims * sizeof(unsigned));
    memset(m_strides, 0, SYN_MAX_TENSOR_DIM * sizeof(unsigned));
    memset(
        m_minSizes,
        TENSOR_DEFAULT_MIN_SIZE,
        SYN_MAX_TENSOR_DIM * sizeof(unsigned));
    m_dynamicRange.min = DynamicRangeMin;
    m_dynamicRange.max = DynamicRangeMax;
    m_dynamicRange.isSet = setDynamicRange;
  }

  unsigned m_batchPos;
  synDataType m_dataType;
  unsigned m_dims;
  unsigned
      m_sizes[SYN_MAX_TENSOR_DIM]; // In dynamic tensors this is the maxSize
  synQuantizationParams m_quantizationParams[SYN_NUM_DATA_TYPES];
  void* m_ptr;
  bool m_isWeights;
  bool m_isQuantized; // TODO 17465 - change name to reflect also fp16/bf16
  const char* m_name;
  unsigned m_strides[SYN_MAX_TENSOR_DIM];
  bool m_isOutput; // used by the GC to optimize memory usage of a segmented op
                   // when
  // flag is true, the tensor is marked as output tensor
  bool m_enablePerChannelQuant; // Enable Per Channel Quantization for tensor
  synPerChannelQuantizationParams
      m_perChannelQuantizationParams[SYN_NUM_DATA_TYPES];
  bool m_isInput;
  bool m_isStatic;
  unsigned m_minSizes[SYN_MAX_TENSOR_DIM];
  synTensorType m_tensorType;
  DynamicRange m_dynamicRange;
  bool m_isSparsityWeights;
};

struct synBeamParams {
  unsigned bsw;
  unsigned axis;
  bool bottomK; // If "bottomK" is false (the default value) then the k largest
                // elements are returned
  uint8_t structPad1 = 0;
  uint16_t structPad2 = 0;

  synBeamParams() : bsw(0), axis(0), bottomK(false) {}

  synBeamParams(unsigned _bsw, unsigned _axis, unsigned _bottomK)
      : bsw(_bsw), axis(_axis), bottomK(_bottomK) {}
};

struct synEinsumParams {
  char equation[MAX_EINSUM_EQUATION_LENGTH + 1];
  explicit synEinsumParams(const char* _equation) {
    if (strlen(_equation) > MAX_EINSUM_EQUATION_LENGTH) {
      snprintf(
          equation,
          MAX_EINSUM_EQUATION_LENGTH + 1,
          "%s",
          INVALID_EINSUM_EQUATION);
    } else {
      snprintf(equation, MAX_EINSUM_EQUATION_LENGTH + 1, "%s", _equation);
    }
  }
  synEinsumParams() : equation() {}
};
struct synStridedOpParams {
  uint64_t baseOffset;
  uint64_t strides[HABANA_DIM_MAX];
};

struct synSliceAxisParamsV2 {
  unsigned axis;
  unsigned _structPadding = 0;
  uint64_t begin;
  uint64_t end;
};

struct synSliceParamsV2 {
  unsigned axes[HABANA_DIM_MAX];
  unsigned _structPadding = 0;
  TSize starts[HABANA_DIM_MAX];
  TSize ends[HABANA_DIM_MAX];
  TSize steps[HABANA_DIM_MAX];
};

struct synPhysicalConcatSplitSubnodeParams {
  unsigned concatSplitDim;
  unsigned nodeNumberInConcatSplit;
  bool isSplit;
  uint8_t structPad1 = 0;
  uint16_t structPad2 = 0;
};

struct synStaticReshapeSifParams {
  TSize inputMaxSizes[HABANA_DIM_MAX];
  TSize outputMaxSizes[HABANA_DIM_MAX];
  char inputStaticDims[HABANA_DIM_MAX]; // acting as boolean
  char outputStaticDims[HABANA_DIM_MAX]; // acting as boolean
  uint16_t structPad2 = 0;
  unsigned dimsNum;
};
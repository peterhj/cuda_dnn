use cuda::ffi::runtime::{cudaStream_t};

use libc::{c_void, c_char, c_int, c_float, size_t};

enum cudnnContext {}
pub type cudnnHandle_t = *mut cudnnContext;

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub enum cudnnStatus_t {
  Success         = 0,
  NotInitialized  = 1,
  AllocFailed     = 2,
  BadParam        = 3,
  InternalError   = 4,
  InvalidError    = 5,
  ArchMismatch    = 6,
  MappingError    = 7,
  ExecutionFailed = 8,
  NotSupported    = 9,
  LicenseError    = 10,
}

impl Default for cudnnStatus_t {
  fn default() -> cudnnStatus_t {
    cudnnStatus_t::Success
  }
}

impl cudnnStatus_t {
  pub fn is_err(&self) -> bool {
    if let cudnnStatus_t::Success = *self {
      false
    } else {
      true
    }
  }
}

enum cudnnTensorStruct {}
pub type cudnnTensorDescriptor_t = *mut cudnnTensorStruct;

enum cudnnConvolutionStruct {}
pub type cudnnConvolutionDescriptor_t = *mut cudnnConvolutionStruct;

enum cudnnPoolingStruct {}
pub type cudnnPoolingDescriptor_t = *mut cudnnPoolingStruct;

enum cudnnFilterStruct {}
pub type cudnnFilterDescriptor_t = *mut cudnnFilterStruct;

enum cudnnLRNStruct {}
pub type cudnnLRNDescriptor_t = *mut cudnnLRNStruct;

enum cudnnActivationStruct {}
pub type cudnnActivationDescriptor_t = *mut cudnnActivationStruct;

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnDataType_t {
  Float   = 0,
  Double  = 1,
  Half    = 2,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnTensorFormat_t {
  RowMajorNCHW    = 0,
  InterleavedNHWC = 1,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnAddMode_t {
  //SameHW      = 0,
  Image       = 0,
  //SameCHW     = 1,
  FeatureMap  = 1,
  SameC       = 2,
  FullTensor  = 3,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnConvolutionMode_t {
  Convolution       = 0,
  CrossCorrelation  = 1,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnConvolutionFwdPreference_t {
  NoWorkspace           = 0,
  PreferFastest         = 1,
  SpecifyWorkspaceLimit = 2,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub enum cudnnConvolutionFwdAlgo_t {
  ImplicitGemm        = 0,
  ImplicitPrecompGemm = 1,
  Gemm                = 2,
  Direct              = 3,
  Fft                 = 4,
  FftTiling           = 5,
  Winograd            = 6,
}

impl Default for cudnnConvolutionFwdAlgo_t {
  fn default() -> cudnnConvolutionFwdAlgo_t {
    cudnnConvolutionFwdAlgo_t::ImplicitGemm
  }
}

#[derive(Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct cudnnConvolutionFwdAlgoPerf_t {
  pub algo:   cudnnConvolutionFwdAlgo_t,
  pub status: cudnnStatus_t,
  pub time:   c_float,
  pub memory: size_t,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnConvolutionBwdFilterPreference_t {
  NoWorkspace           = 0,
  PreferFastest         = 1,
  SpecifyWorkspaceLimit = 2,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub enum cudnnConvolutionBwdFilterAlgo_t {
  NonDeterministic          = 0,
  Deterministic             = 1,
  Fft                       = 2,
  NonDeterministicWorkspace = 3,
}

impl Default for cudnnConvolutionBwdFilterAlgo_t {
  fn default() -> cudnnConvolutionBwdFilterAlgo_t {
    cudnnConvolutionBwdFilterAlgo_t::NonDeterministic
  }
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct cudnnConvolutionBwdFilterAlgoPerf_t {
  pub algo:   cudnnConvolutionBwdFilterAlgo_t,
  pub status: cudnnStatus_t,
  pub time:   c_float,
  pub memory: size_t,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnConvolutionBwdDataPreference_t {
  NoWorkspace           = 0,
  PreferFastest         = 1,
  SpecifyWorkspaceLimit = 2,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub enum cudnnConvolutionBwdDataAlgo_t {
  NonDeterministic  = 0,
  Deterministic     = 1,
  Fft               = 2,
  FftTiling         = 3,
  Winograd          = 4,
}

impl Default for cudnnConvolutionBwdDataAlgo_t {
  fn default() -> cudnnConvolutionBwdDataAlgo_t {
    cudnnConvolutionBwdDataAlgo_t::NonDeterministic
  }
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct cudnnConvolutionBwdDataAlgoPerf_t {
  pub algo:   cudnnConvolutionBwdDataAlgo_t,
  pub status: cudnnStatus_t,
  pub time:   c_float,
  pub memory: size_t,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnSoftmaxAlgorithm_t {
  Fast      = 0,
  Accurate  = 1,
  Log       = 2,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnSoftmaxMode_t {
  Instance  = 0,
  Channel   = 1,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnPoolingMode_t {
  Max                           = 0,
  AverageCountIncludingPadding  = 1,
  AverageCountExcludingPadding  = 2,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnActivationMode_t {
  Sigmoid     = 0,
  Relu        = 1,
  Tanh        = 2,
  ClippedRelu = 3,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnNanPropagation_t {
  NotPropagateNan = 0,
  PropagateNan    = 1,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub enum cudnnBatchNormMode_t {
  PerActivation = 0,
  Spatial       = 1,
}

#[link(name = "cudnn", kind = "dylib")]
extern "C" {
  pub fn cudnnGetVersion() -> size_t;

  pub fn cudnnGetErrorString(status: cudnnStatus_t) -> *const c_char;

  pub fn cudnnCreate(handle: *mut cudnnHandle_t) -> cudnnStatus_t;
  pub fn cudnnDestroy(handle: cudnnHandle_t) -> cudnnStatus_t;
  pub fn cudnnSetStream(handle: cudnnHandle_t, stream: cudaStream_t) -> cudnnStatus_t;
  pub fn cudnnGetStream(handle: cudnnHandle_t, stream: *mut cudaStream_t) -> cudnnStatus_t;

  pub fn cudnnCreateTensorDescriptor(tensor_desc: *mut cudnnTensorDescriptor_t) -> cudnnStatus_t;
  pub fn cudnnDestroyTensorDescriptor(tensor_desc: cudnnTensorDescriptor_t) -> cudnnStatus_t;
  pub fn cudnnSetTensor4dDescriptor(
      tensor_desc: cudnnTensorDescriptor_t,
      format: cudnnTensorFormat_t,
      data_ty: cudnnDataType_t,
      n: c_int,
      c: c_int,
      h: c_int,
      w: c_int,
  ) -> cudnnStatus_t;
  pub fn cudnnSetTensor4dDescriptorEx(
      tensor_desc: cudnnTensorDescriptor_t,
      data_ty: cudnnDataType_t,
      n: c_int,
      c: c_int,
      h: c_int,
      w: c_int,
      n_stride: c_int,
      c_stride: c_int,
      h_stride: c_int,
      w_stride: c_int,
  ) -> cudnnStatus_t;
  pub fn cudnnGetTensor4dDescriptor(
      tensor_desc: cudnnTensorDescriptor_t,
      data_ty: *mut cudnnDataType_t,
      n: *mut c_int,
      c: *mut c_int,
      h: *mut c_int,
      w: *mut c_int,
      n_stride: *mut c_int,
      c_stride: *mut c_int,
      h_stride: *mut c_int,
      w_stride: *mut c_int,
  ) -> cudnnStatus_t;
  pub fn cudnnCreateFilterDescriptor(filter_desc: *mut cudnnFilterDescriptor_t) -> cudnnStatus_t;
  pub fn cudnnDestroyFilterDescriptor(filter_desc: cudnnFilterDescriptor_t) -> cudnnStatus_t;
  pub fn cudnnSetFilter4dDescriptor(
      filter_desc: cudnnFilterDescriptor_t,
      data_ty: cudnnDataType_t,
      format: cudnnTensorFormat_t,
      k: c_int,
      c: c_int,
      h: c_int,
      w: c_int,
  ) -> cudnnStatus_t;
  pub fn cudnnCreateConvolutionDescriptor(conv_desc: *mut cudnnConvolutionDescriptor_t) -> cudnnStatus_t;
  pub fn cudnnDestroyConvolutionDescriptor(conv_desc: cudnnConvolutionDescriptor_t) -> cudnnStatus_t;
  pub fn cudnnSetConvolution2dDescriptor(
      conv_desc: cudnnConvolutionDescriptor_t,
      pad_h: c_int,
      pad_w: c_int,
      u: c_int,
      v: c_int,
      upscalex: c_int,
      upscaley: c_int,
      conv_mode: cudnnConvolutionMode_t,
  ) -> cudnnStatus_t;
  pub fn cudnnCreateActivationDescriptor(activation_desc: *mut cudnnActivationDescriptor_t) -> cudnnStatus_t;
  pub fn cudnnDestroyActivationDescriptor(activation_desc: cudnnActivationDescriptor_t) -> cudnnStatus_t;
  pub fn cudnnSetActivationDescriptor(
      activation_desc: cudnnActivationDescriptor_t,
      mode: cudnnActivationMode_t,
      relu_nan_opt: cudnnNanPropagation_t,
      relu_ceiling: f64,
  ) -> cudnnStatus_t;
  pub fn cudnnCreatePoolingDescriptor(pooling_desc: *mut cudnnPoolingDescriptor_t) -> cudnnStatus_t;
  pub fn cudnnDestroyPoolingDescriptor(pooling_desc: cudnnPoolingDescriptor_t) -> cudnnStatus_t;
  pub fn cudnnSetPooling2dDescriptor(
      pooling_desc: cudnnPoolingDescriptor_t,
      mode: cudnnPoolingMode_t,
      maxpooling_nan_opt: cudnnNanPropagation_t,
      window_height: c_int,
      window_width: c_int,
      vertical_padding: c_int,
      horizontal_padding: c_int,
      vertical_stride: c_int,
      horizontal_stride: c_int,
  ) -> cudnnStatus_t;

  // TODO

  pub fn cudnnFindConvolutionForwardAlgorithm(
      handle: cudnnHandle_t,
      src_desc: cudnnTensorDescriptor_t,
      filter_desc: cudnnFilterDescriptor_t,
      conv_desc: cudnnConvolutionDescriptor_t,
      dest_desc: cudnnTensorDescriptor_t,
      requested_algo_count: c_int,
      returned_algo_count: *mut c_int,
      perf_results: *mut cudnnConvolutionFwdAlgoPerf_t,
  ) -> cudnnStatus_t;
  pub fn cudnnFindConvolutionBackwardFilterAlgorithm(
      handle: cudnnHandle_t,
      src_desc: cudnnTensorDescriptor_t,
      diff_desc: cudnnTensorDescriptor_t,
      conv_desc: cudnnConvolutionDescriptor_t,
      grad_desc: cudnnFilterDescriptor_t,
      requested_algo_count: c_int,
      returned_algo_count: *mut c_int,
      perf_results: *mut cudnnConvolutionBwdFilterAlgoPerf_t,
  ) -> cudnnStatus_t;
  pub fn cudnnFindConvolutionBackwardDataAlgorithm(
      handle: cudnnHandle_t,
      filter_desc: cudnnFilterDescriptor_t,
      diff_desc: cudnnTensorDescriptor_t,
      conv_desc: cudnnConvolutionDescriptor_t,
      grad_desc: cudnnTensorDescriptor_t,
      requested_algo_count: c_int,
      returned_algo_count: *mut c_int,
      perf_results: *mut cudnnConvolutionBwdDataAlgoPerf_t,
  ) -> cudnnStatus_t;
  pub fn cudnnGetConvolutionForwardWorkspaceSize(
      handle: cudnnHandle_t,
      src_desc: cudnnTensorDescriptor_t,
      filter_desc: cudnnFilterDescriptor_t,
      conv_desc: cudnnConvolutionDescriptor_t,
      dest_desc: cudnnTensorDescriptor_t,
      algo: cudnnConvolutionFwdAlgo_t,
      size_in_bytes: *mut size_t,
  ) -> cudnnStatus_t;
  pub fn cudnnGetConvolutionBackwardFilterWorkspaceSize(
      handle: cudnnHandle_t,
      src_desc: cudnnTensorDescriptor_t,
      diff_desc: cudnnTensorDescriptor_t,
      conv_desc: cudnnConvolutionDescriptor_t,
      grad_desc: cudnnFilterDescriptor_t,
      algo: cudnnConvolutionBwdFilterAlgo_t,
      size_in_bytes: *mut size_t,
  ) -> cudnnStatus_t;
  pub fn cudnnGetConvolutionBackwardDataWorkspaceSize(
      handle: cudnnHandle_t,
      filter_desc: cudnnFilterDescriptor_t,
      diff_desc: cudnnTensorDescriptor_t,
      conv_desc: cudnnConvolutionDescriptor_t,
      grad_desc: cudnnTensorDescriptor_t,
      algo: cudnnConvolutionBwdDataAlgo_t,
      size_in_bytes: *mut size_t,
  ) -> cudnnStatus_t;

  pub fn cudnnConvolutionForward(
      handle: cudnnHandle_t,
      alpha: *const c_void,
      src_desc: cudnnTensorDescriptor_t,
      src_data: *const c_void,
      filter_desc: cudnnFilterDescriptor_t,
      filter_data: *const c_void,
      conv_desc: cudnnConvolutionDescriptor_t,
      algo: cudnnConvolutionFwdAlgo_t,
      work_space: *mut c_void,
      work_space_size_in_bytes: size_t,
      beta: *const c_void,
      dst_desc: cudnnTensorDescriptor_t,
      dst_data: *mut c_void,
  ) -> cudnnStatus_t;
  pub fn cudnnConvolutionBackwardBias(
      handle: cudnnHandle_t,
      alpha: *const c_void,
      src_desc: cudnnTensorDescriptor_t,
      src_data: *const c_void,
      beta: *const c_void,
      dst_desc: cudnnTensorDescriptor_t,
      dst_data: *mut c_void,
  ) -> cudnnStatus_t;
  pub fn cudnnConvolutionBackwardFilter(
      handle: cudnnHandle_t,
      alpha: *const c_void,
      src_desc: cudnnTensorDescriptor_t,
      src_data: *const c_void,
      diff_desc: cudnnTensorDescriptor_t,
      diff_data: *const c_void,
      conv_desc: cudnnConvolutionDescriptor_t,
      algo: cudnnConvolutionBwdFilterAlgo_t,
      work_space: *mut c_void,
      work_space_size_in_bytes: size_t,
      beta: *const c_void,
      grad_desc: cudnnFilterDescriptor_t,
      grad_data: *mut c_void,
  ) -> cudnnStatus_t;
  pub fn cudnnConvolutionBackwardData(
      handle: cudnnHandle_t,
      alpha: *const c_void,
      filter_desc: cudnnFilterDescriptor_t,
      filter_data: *const c_void,
      diff_desc: cudnnTensorDescriptor_t,
      diff_data: *const c_void,
      conv_desc: cudnnConvolutionDescriptor_t,
      algo: cudnnConvolutionBwdDataAlgo_t,
      work_space: *mut c_void,
      work_space_size_in_bytes: size_t,
      beta: *const c_void,
      grad_desc: cudnnTensorDescriptor_t,
      grad_data: *mut c_void,
  ) -> cudnnStatus_t;

  pub fn cudnnAddTensor(
      handle: cudnnHandle_t,
      alpha: *const c_void,
      bias_desc: cudnnTensorDescriptor_t,
      bias_data: *const c_void,
      beta: *const c_void,
      src_dst_desc: cudnnTensorDescriptor_t,
      src_dst_data: *mut c_void,
  ) -> cudnnStatus_t;

  pub fn cudnnActivationForward_v3(
      handle: cudnnHandle_t,
      mode: cudnnActivationMode_t,
      alpha: *const c_void,
      src_desc: cudnnTensorDescriptor_t,
      src_data: *const c_void,
      beta: *const c_void,
      dst_desc: cudnnTensorDescriptor_t,
      dst_data: *mut c_void,
  ) -> cudnnStatus_t;
  pub fn cudnnActivationForward_v4(
      handle: cudnnHandle_t,
      activation_desc: cudnnActivationDescriptor_t,
      alpha: *const c_void,
      src_desc: cudnnTensorDescriptor_t,
      src_data: *const c_void,
      beta: *const c_void,
      dst_desc: cudnnTensorDescriptor_t,
      dst_data: *mut c_void,
  ) -> cudnnStatus_t;
  pub fn cudnnActivationBackward_v3(
      handle: cudnnHandle_t,
      mode: cudnnActivationMode_t,
      alpha: *const c_void,
      src_desc: cudnnTensorDescriptor_t,
      src_data: *const c_void,
      src_diff_desc: cudnnTensorDescriptor_t,
      src_diff_data: *const c_void,
      dst_desc: cudnnTensorDescriptor_t,
      dst_data: *const c_void,
      beta: *const c_void,
      dst_diff_desc: cudnnTensorDescriptor_t,
      dst_diff_data: *mut c_void,
  ) -> cudnnStatus_t;
  pub fn cudnnActivationBackward_v4(
      handle: cudnnHandle_t,
      activation_desc: cudnnActivationDescriptor_t,
      alpha: *const c_void,
      src_desc: cudnnTensorDescriptor_t,
      src_data: *const c_void,
      src_diff_desc: cudnnTensorDescriptor_t,
      src_diff_data: *const c_void,
      dst_desc: cudnnTensorDescriptor_t,
      dst_data: *const c_void,
      beta: *const c_void,
      dst_diff_desc: cudnnTensorDescriptor_t,
      dst_diff_data: *mut c_void,
  ) -> cudnnStatus_t;

  pub fn cudnnSoftmaxForward(
      handle: cudnnHandle_t,
      algorithm: cudnnSoftmaxAlgorithm_t,
      mode: cudnnSoftmaxMode_t,
      alpha: *const c_void,
      src_desc: cudnnTensorDescriptor_t,
      src_data: *const c_void,
      beta: *const c_void,
      dst_desc: cudnnTensorDescriptor_t,
      dst_data: *mut c_void,
  ) -> cudnnStatus_t;
  pub fn cudnnSoftmaxBackward(
      handle: cudnnHandle_t,
      algorithm: cudnnSoftmaxAlgorithm_t,
      mode: cudnnSoftmaxMode_t,
      alpha: *const c_void,
      src_desc: cudnnTensorDescriptor_t,
      src_data: *const c_void,
      src_diff_desc: cudnnTensorDescriptor_t,
      src_diff_data: *const c_void,
      beta: *const c_void,
      dst_diff_desc: cudnnTensorDescriptor_t,
      dst_diff_data: *mut c_void,
  ) -> cudnnStatus_t;

  pub fn cudnnPoolingForward(
      handle: cudnnHandle_t,
      pooling_desc: cudnnPoolingDescriptor_t,
      alpha: *const c_void,
      x_desc: cudnnTensorDescriptor_t,
      x: *const c_void,
      beta: *const c_void,
      y_desc: cudnnTensorDescriptor_t,
      y: *mut c_void,
  ) -> cudnnStatus_t;
  pub fn cudnnPoolingBackward(
      handle: cudnnHandle_t,
      pooling_desc: cudnnPoolingDescriptor_t,
      alpha: *const c_void,
      y_desc: cudnnTensorDescriptor_t,
      y: *const c_void,
      dy_desc: cudnnTensorDescriptor_t,
      dy: *const c_void,
      x_desc: cudnnTensorDescriptor_t,
      x: *const c_void,
      beta: *const c_void,
      dx_desc: cudnnTensorDescriptor_t,
      dx: *mut c_void,
  ) -> cudnnStatus_t;

  pub fn cudnnTransformTensor(
      handle: cudnnHandle_t,
      alpha: *const c_void,
      x_desc: cudnnTensorDescriptor_t,
      x: *const c_void,
      beta: *const c_void,
      y_desc: cudnnTensorDescriptor_t,
      y: *mut c_void,
  ) -> cudnnStatus_t;

  pub fn cudnnBatchNormalizationForwardInference(
      handle: cudnnHandle_t,
      mode: cudnnBatchNormMode_t,
      alpha: *const c_void,
      beta: *const c_void,
      x_desc: cudnnTensorDescriptor_t,
      x: *const c_void,
      y_desc: cudnnTensorDescriptor_t,
      y: *mut c_void,
      bn_scale_bias_mean_var_desc: cudnnTensorDescriptor_t,
      bn_scale: *const c_void,
      bn_bias: *const c_void,
      estimated_mean: *const c_void,
      estimated_inv_variance: *const c_void,
      epsilon: f64,
  ) -> cudnnStatus_t;
  pub fn cudnnBatchNormalizationForwardTraining(
      handle: cudnnHandle_t,
      mode: cudnnBatchNormMode_t,
      alpha: *const c_void,
      beta: *const c_void,
      x_desc: cudnnTensorDescriptor_t,
      x: *const c_void,
      y_desc: cudnnTensorDescriptor_t,
      y: *mut c_void,
      bn_scale_bias_mean_var_desc: cudnnTensorDescriptor_t,
      bn_scale: *const c_void,
      bn_bias: *const c_void,
      exponential_average_factor: f64,
      result_running_mean: *mut c_void,
      result_running_inv_variance: *mut c_void,
      epsilon: f64,
      result_save_mean: *mut c_void,
      result_save_inv_variance: *mut c_void,
  ) -> cudnnStatus_t;
  pub fn cudnnBatchNormalizationBackward(
      handle: cudnnHandle_t,
      mode: cudnnBatchNormMode_t,
      alpha_data_diff: *const c_void,
      beta_data_diff: *const c_void,
      alpha_param_diff: *const c_void,
      beta_param_diff: *const c_void,
      x_desc: cudnnTensorDescriptor_t,
      x: *const c_void,
      dy_desc: cudnnTensorDescriptor_t,
      dy: *const c_void,
      dx_desc: cudnnTensorDescriptor_t,
      dx: *mut c_void,
      bn_scale_bias_mean_var_desc: cudnnTensorDescriptor_t,
      bn_scale: *const c_void,
      result_bn_scale_diff: *mut c_void,
      result_bn_bias_diff: *mut c_void,
      epsilon: f64,
      saved_mean: *const c_void,
      saved_inv_variance: *const c_void,
  ) -> cudnnStatus_t;

  // TODO
}

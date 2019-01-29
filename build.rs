#[cfg(feature = "fresh")]
extern crate bindgen;

use std::env;
#[cfg(feature = "fresh")]
use std::fs;
use std::path::{PathBuf};

fn to_cuda_lib_dir(cuda_dir: &PathBuf) -> PathBuf {
  if cfg!(target_os = "linux") {
    if cfg!(target_arch = "x86_64") {
      cuda_dir.join("lib64")
    } else if cfg!(target_arch = "powerpc64le") {
      panic!("todo: ppc64le support on linux is not yet implemented");
    } else {
      panic!("unsupported target arch on linux");
    }
  } else if cfg!(target_os = "windows") {
    if cfg!(target_arch = "x86_64") {
      cuda_dir.join("lib").join("x64")
    } else {
      panic!("unsupported target arch on windows");
    }
  } else if cfg!(target_os = "macos") {
    unimplemented!();
  } else {
    panic!("unsupported target os");
  }
}

#[cfg(not(feature = "fresh"))]
fn main() {
  println!("cargo:rerun-if-changed=build.rs");
  println!("cargo:rerun-if-env-changed=CUDA_HOME");
  println!("cargo:rerun-if-env-changed=CUDA_PATH");
  println!("cargo:rerun-if-env-changed=CUDA_LIBRARY_PATH");
  println!("cargo:rerun-if-env-changed=CUDNN_HOME");
  println!("cargo:rustc-link-lib=cudnn");
  let maybe_cuda_dir =
      env::var("CUDA_HOME")
        .or_else(|_| env::var("CUDA_PATH"))
        .ok().map(|s| PathBuf::from(s));
  let maybe_cuda_fallback_lib_dir =
      maybe_cuda_dir.as_ref().map(|d| to_cuda_lib_dir(d));
  let maybe_cuda_lib_dir =
      env::var("CUDA_LIBRARY_PATH")
        .ok().map(|s| PathBuf::from(s));
  let maybe_cudnn_dir =
      env::var("CUDNN_HOME")
        .ok().map(|s| PathBuf::from(s));
  let maybe_cudnn_lib_dir =
      maybe_cudnn_dir.as_ref().map(|d| to_cuda_lib_dir(d));
  if let Some(cudnn_lib_dir) = maybe_cudnn_lib_dir {
    println!("cargo:rustc-link-search=native={}", cudnn_lib_dir.display());
  }
  if let Some(cuda_lib_dir) = maybe_cuda_lib_dir {
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
  }
  if let Some(cuda_lib_dir) = maybe_cuda_fallback_lib_dir {
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
  }
}

#[cfg(feature = "fresh")]
fn main() {
  println!("cargo:rerun-if-changed=build.rs");
  println!("cargo:rerun-if-env-changed=CUDA_HOME");
  println!("cargo:rerun-if-env-changed=CUDA_PATH");
  println!("cargo:rerun-if-env-changed=CUDA_LIBRARY_PATH");
  println!("cargo:rerun-if-env-changed=CUDNN_HOME");
  println!("cargo:rustc-link-lib=cudnn");

  let maybe_cuda_dir =
      env::var("CUDA_HOME")
        .or_else(|_| env::var("CUDA_PATH"))
        .ok().map(|s| PathBuf::from(s));
  let maybe_cuda_fallback_lib_dir =
      maybe_cuda_dir.as_ref().map(|d| to_cuda_lib_dir(d));
  let maybe_cuda_lib_dir =
      env::var("CUDA_LIBRARY_PATH")
        .ok().map(|s| PathBuf::from(s));
  let maybe_cudnn_dir =
      env::var("CUDNN_HOME")
        .ok().map(|s| PathBuf::from(s));
  let maybe_cudnn_lib_dir =
      maybe_cudnn_dir.as_ref().map(|d| to_cuda_lib_dir(d));

  if let Some(cudnn_lib_dir) = maybe_cudnn_lib_dir {
    println!("cargo:rustc-link-search=native={}", cudnn_lib_dir.display());
  }
  if let Some(cuda_lib_dir) = maybe_cuda_lib_dir {
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
  }
  if let Some(cuda_lib_dir) = maybe_cuda_fallback_lib_dir {
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
  }

  let maybe_cuda_include_dir =
      maybe_cuda_dir.as_ref().map(|d| d.join("include"));
  let maybe_cudnn_include_dir =
      maybe_cudnn_dir.as_ref().map(|d| d.join("include"));

  #[cfg(feature = "cudnn_7_4")]
  let a_cudnn_version_feature_must_be_enabled = "v7_4";
  let v = a_cudnn_version_feature_must_be_enabled;

  let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
  let gensrc_dir = manifest_dir.join("gensrc").join("ffi").join(v);
  println!("cargo:rerun-if-changed={}", gensrc_dir.display());
  fs::create_dir_all(&gensrc_dir).ok();

  println!("cargo:rerun-if-changed={}", gensrc_dir.join("_cudnn").display());
  fs::remove_file(gensrc_dir.join("_cudnn.rs")).ok();
  let builder = bindgen::Builder::default();
  let builder = if let Some(ref cudnn_include_dir) = maybe_cudnn_include_dir {
    builder.clang_arg(format!("-I{}", cudnn_include_dir.display()))
  } else {
    builder
  };
  let builder = if let Some(ref cuda_include_dir) = maybe_cuda_include_dir {
    builder.clang_arg(format!("-I{}", cuda_include_dir.display()))
  } else {
    builder
  };
  builder
    .header("wrapped_cudnn.h")
    .whitelist_recursively(false)
    .whitelist_var("CUDNN_MAJOR")
    .whitelist_var("CUDNN_MINOR")
    .whitelist_var("CUDNN_PATCHLEVEL")
    .whitelist_var("CUDNN_VERSION")
    .whitelist_var("CUDNN_DIM_MAX")
    .whitelist_type("cudnnContext")
    .whitelist_type("cudnnHandle_t")
    .whitelist_type("cudnnRuntimeTag_t")
    .whitelist_type("cudnnTensorStruct")
    .whitelist_type("cudnnTensorDescriptor_t")
    .whitelist_type("cudnnConvolutionStruct")
    .whitelist_type("cudnnConvolutionDescriptor_t")
    .whitelist_type("cudnnPoolingStruct")
    .whitelist_type("cudnnPoolingDescriptor_t")
    .whitelist_type("cudnnFilterStruct")
    .whitelist_type("cudnnFilterDescriptor_t")
    .whitelist_type("cudnnLRNStruct")
    .whitelist_type("cudnnLRNDescriptor_t")
    .whitelist_type("cudnnActivationStruct")
    .whitelist_type("cudnnActivationDescriptor_t")
    .whitelist_type("cudnnSpatialTransformerStruct")
    .whitelist_type("cudnnSpatialTransformerDescriptor_t")
    .whitelist_type("cudnnOpTensorStruct")
    .whitelist_type("cudnnOpTensorDescriptor_t")
    .whitelist_type("cudnnReduceTensorStruct")
    .whitelist_type("cudnnReduceTensorDescriptor_t")
    .whitelist_type("cudnnCTCLossStruct")
    .whitelist_type("cudnnCTCLossDescriptor_t")
    .whitelist_type("cudnnConvolutionFwdAlgoPerf_t")
    .whitelist_type("cudnnConvolutionBwdFilterAlgoPerf_t")
    .whitelist_type("cudnnConvolutionBwdDataAlgoPerf_t")
    .whitelist_type("cudnnDropoutStruct")
    .whitelist_type("cudnnDropoutDescriptor_t")
    .whitelist_type("cudnnAlgorithmStruct")
    .whitelist_type("cudnnAlgorithmDescriptor_t")
    .whitelist_type("cudnnAlgorithmPerformanceStruct")
    .whitelist_type("cudnnAlgorithmPerformance_t")
    .whitelist_type("cudnnRNNStruct")
    .whitelist_type("cudnnRNNDescriptor_t")
    .whitelist_type("cudnnPersistentRNNPlan")
    .whitelist_type("cudnnPersistentRNNPlan_t")
    .whitelist_type("cudnnAlgorithm_t")
    .whitelist_type("cudnnDebug_t")
    .whitelist_type("cudnnRNNDataStruct")
    .whitelist_type("cudnnRNNDataDescriptor_t")
    .whitelist_type("cudnnStatus_t")
    .whitelist_type("cudnnErrQueryMode_t")
    .whitelist_type("cudnnDataType_t")
    .whitelist_type("cudnnMathType_t")
    .whitelist_type("cudnnNanPropagation_t")
    .whitelist_type("cudnnDeterminism_t")
    .whitelist_type("cudnnTensorFormat_t")
    .whitelist_type("cudnnOpTensorOp_t")
    .whitelist_type("cudnnReduceTensorOp_t")
    .whitelist_type("cudnnReduceTensorIndices_t")
    .whitelist_type("cudnnIndicesType_t")
    .whitelist_type("cudnnConvolutionMode_t")
    .whitelist_type("cudnnConvolutionFwdPreference_t")
    .whitelist_type("cudnnConvolutionFwdAlgo_t")
    .whitelist_type("cudnnConvolutionBwdFilterPreference_t")
    .whitelist_type("cudnnConvolutionBwdFilterAlgo_t")
    .whitelist_type("cudnnConvolutionBwdDataPreference_t")
    .whitelist_type("cudnnConvolutionBwdDataAlgo_t")
    .whitelist_type("cudnnSoftmaxAlgorithm_t")
    .whitelist_type("cudnnSoftmaxMode_t")
    .whitelist_type("cudnnPoolingMode_t")
    .whitelist_type("cudnnActivationMode_t")
    .whitelist_type("cudnnLRNMode_t")
    .whitelist_type("cudnnDivNormMode_t")
    .whitelist_type("cudnnBatchNormMode_t")
    .whitelist_type("cudnnBatchNormOps_t")
    .whitelist_type("cudnnSamplerType_t")
    .whitelist_type("cudnnRNNMode_t")
    .whitelist_type("cudnnDirectionMode_t")
    .whitelist_type("cudnnRNNInputMode_t")
    .whitelist_type("cudnnRNNAlgo_t")
    .whitelist_type("cudnnCTCLossAlgo_t")
    .whitelist_type("cudnnRNNClipMode_t")
    .whitelist_type("cudnnSeverity_t")
    .whitelist_type("cudnnRNNDataLayout_t")
    .whitelist_type("cudnnRNNPaddingMode_t")
    .whitelist_function("cudnnGetVersion")
    .whitelist_function("cudnnGetCudartVersion")
    .whitelist_function("cudnnGetErrorString")
    .whitelist_function("cudnnCreate")
    .whitelist_function("cudnnDestroy")
    .whitelist_function("cudnnGetProperty")
    .whitelist_function("cudnnSetStream")
    .whitelist_function("cudnnGetStream")
    .whitelist_function("cudnnCreateTensorDescriptor")
    .whitelist_function("cudnnDestroyTensorDescriptor")
    .whitelist_function("cudnnSetTensor4dDescriptor")
    .whitelist_function("cudnnSetTensorNdDescriptor")
    .whitelist_function("cudnnCreateFilterDescriptor")
    .whitelist_function("cudnnDestroyFilterDescriptor")
    .whitelist_function("cudnnSetFilter4dDescriptor")
    .whitelist_function("cudnnSetFilterNdDescriptor")
    .whitelist_function("cudnnCreateConvolutionDescriptor")
    .whitelist_function("cudnnDestroyConvolutionDescriptor")
    .whitelist_function("cudnnSetConvolution2dDescriptor")
    .whitelist_function("cudnnSetConvolutionNdDescriptor")
    .whitelist_function("cudnnSetConvolutionGroupCount")
    .whitelist_function("cudnnSetConvolutionMathType")
    .whitelist_function("cudnnConvolutionForward")
    .whitelist_function("cudnnConvolutionBackwardBias")
    .whitelist_function("cudnnConvolutionBackwardFilter")
    .whitelist_function("cudnnConvolutionBackwardData")
    .whitelist_function("cudnnCreatePoolingDescriptor")
    .whitelist_function("cudnnDestroyPoolingDescriptor")
    .whitelist_function("cudnnSetPooling2dDescriptor")
    .whitelist_function("cudnnSetPoolingNdDescriptor")
    .whitelist_function("cudnnPoolingForward")
    .whitelist_function("cudnnPoolingBackward")
    .whitelist_function("cudnnCreateActivationDescriptor")
    .whitelist_function("cudnnDestroyActivationDescriptor")
    .whitelist_function("cudnnSetActivationDescriptor")
    .whitelist_function("cudnnSoftmaxForward")
    .whitelist_function("cudnnSoftmaxBackward")
    .prepend_enum_name(false)
    .generate_comments(false)
    .rustfmt_bindings(true)
    .generate()
    .expect("bindgen failed to generate cudnn bindings")
    .write_to_file(gensrc_dir.join("_cudnn.rs"))
    .expect("bindgen failed to write cudnn bindings");
}

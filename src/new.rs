use bind_ffi::*;

use cuda::runtime_new::*;

pub struct CudnnHandle {
  ptr:  cudnnHandle_t,
}

pub struct CudnnTensorDesc {
}

pub struct CudnnFilterDesc {
}

pub trait CudnnConvExt {
  unsafe fn conv_fwd(&self);
  unsafe fn conv_bwd_filter(&self);
  unsafe fn conv_bwd_data(&self);
}

pub trait CudnnPoolExt {
  unsafe fn pool_fwd(&self);
  unsafe fn pool_bwd(&self);
}

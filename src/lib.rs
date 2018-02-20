#![allow(non_upper_case_globals)]

extern crate cuda;
extern crate float;

use ffi::*;

use cuda::runtime::*;
use float::stub::*;

use std::marker::{PhantomData};
use std::mem::{uninitialized};
use std::os::raw::{c_int};
use std::ptr::{null_mut};

pub mod ffi;

#[derive(Clone, Copy, Debug)]
pub struct CudnnError(pub cudnnStatus_t);

pub type CudnnResult<T> = Result<T, CudnnError>;

pub trait CudnnDataTypeExt: Copy {
  fn cudnn_data_ty() -> cudnnDataType_t;
}

impl CudnnDataTypeExt for f16_stub {
  fn cudnn_data_ty() -> cudnnDataType_t {
    cudnnDataType_t_CUDNN_DATA_HALF
  }
}

impl CudnnDataTypeExt for f32 {
  fn cudnn_data_ty() -> cudnnDataType_t {
    cudnnDataType_t_CUDNN_DATA_FLOAT
  }
}

impl CudnnDataTypeExt for f64 {
  fn cudnn_data_ty() -> cudnnDataType_t {
    cudnnDataType_t_CUDNN_DATA_DOUBLE
  }
}

pub struct CudnnHandle {
  ptr:  cudnnHandle_t,
}

impl Drop for CudnnHandle {
  fn drop(&mut self) {
    let status = unsafe { cudnnDestroy(self.ptr) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => {}
      _ => panic!("failed to destroy cudnn handle: {:?}", status),
    }
  }
}

impl CudnnHandle {
  pub fn create() -> CudnnResult<CudnnHandle> {
    let mut ptr: cudnnHandle_t = null_mut();
    let status = unsafe { cudnnCreate(&mut ptr as *mut cudnnHandle_t) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => Ok(CudnnHandle{ptr: ptr}),
      _ => Err(CudnnError(status)),
    }
  }

  pub unsafe fn as_mut_ptr(&mut self) -> cudnnHandle_t {
    self.ptr
  }

  pub fn set_stream(&mut self, stream: &mut CudaStream) -> CudnnResult<()> {
    let status = unsafe { cudnnSetStream(self.ptr, stream.as_ptr()) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => Ok(()),
      _ => Err(CudnnError(status)),
    }
  }
}

pub struct CudnnTensorDesc<T> {
  ptr:  cudnnTensorDescriptor_t,
  _m:   PhantomData<T>,
}

impl<T> Drop for CudnnTensorDesc<T> {
  fn drop(&mut self) {
    let status = unsafe { cudnnDestroyTensorDescriptor(self.ptr) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => {}
      _ => panic!("failed to destroy cudnn tensor desc: {:?}", status),
    }
  }
}

impl<T> CudnnTensorDesc<T> where T: CudnnDataTypeExt {
  pub fn create() -> CudnnResult<CudnnTensorDesc<T>> {
    let mut ptr: cudnnTensorDescriptor_t = null_mut();
    let status = unsafe { cudnnCreateTensorDescriptor(&mut ptr as *mut _) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => Ok(CudnnTensorDesc{ptr: ptr, _m: PhantomData}),
      e => Err(CudnnError(e)),
    }
  }

  pub unsafe fn as_mut_ptr(&self) -> cudnnTensorDescriptor_t {
    self.ptr
  }

  pub fn set_4d_nchw(&self, num: i32, channels: i32, height: i32, width: i32) -> CudnnResult<()> {
    // TODO
    unimplemented!();
  }

  pub fn create_4d_nchw(num: usize, channels: usize, height: usize, width: usize) -> CudnnResult<CudnnTensorDesc<T>> {
    let mut ptr: cudnnTensorDescriptor_t = null_mut();
    let status = unsafe { cudnnCreateTensorDescriptor(&mut ptr as *mut _) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => {}
      _ => return Err(CudnnError(status)),
    }
    let status = unsafe { cudnnSetTensor4dDescriptor(
        ptr,
        cudnnTensorFormat_t_CUDNN_TENSOR_NCHW,
        T::cudnn_data_ty(),
        num as _,
        channels as _,
        height as _,
        width as _,
    ) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => Ok(CudnnTensorDesc{
        ptr: ptr,
        _m: PhantomData,
      }),
      _ => Err(CudnnError(status)),
    }
  }

  pub fn create_4d_nchw_strided(num: usize, channels: usize, height: usize, width: usize, s_num: usize, s_channels: usize, s_height: usize, s_width: usize) -> CudnnResult<CudnnTensorDesc<T>> {
    // TODO
    unimplemented!();
  }
}

pub struct CudnnFilterDesc<T> {
  ptr:  cudnnFilterDescriptor_t,
  _m:   PhantomData<T>,
}

impl<T> Drop for CudnnFilterDesc<T> {
  fn drop(&mut self) {
    let status = unsafe { cudnnDestroyFilterDescriptor(self.ptr) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => {}
      _ => panic!("failed to destroy cudnn filter desc: {:?}", status),
    }
  }
}

impl<T> CudnnFilterDesc<T> where T: CudnnDataTypeExt {
  pub fn create() -> CudnnResult<CudnnFilterDesc<T>> {
    let mut ptr: cudnnFilterDescriptor_t = null_mut();
    let status = unsafe { cudnnCreateFilterDescriptor(&mut ptr as *mut _) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => Ok(CudnnFilterDesc{ptr: ptr, _m: PhantomData}),
      e => Err(CudnnError(e)),
    }
  }

  pub unsafe fn as_mut_ptr(&self) -> cudnnFilterDescriptor_t {
    self.ptr
  }

  pub fn set_4d_nchw(&self, num: i32, channels: i32, height: i32, width: i32) -> CudnnResult<()> {
    // TODO
    unimplemented!();
  }

  pub fn create_4d_nchw(num: usize, channels: usize, height: usize, width: usize) -> CudnnResult<CudnnTensorDesc<T>> {
    // TODO
    unimplemented!();
  }
}

pub struct CudnnConvDesc {
  ptr:  cudnnConvolutionDescriptor_t,
}

impl CudnnConvDesc {
  pub fn create() -> CudnnResult<CudnnConvDesc> {
    let mut ptr = null_mut();
    let status = unsafe { cudnnCreateConvolutionDescriptor(&mut ptr as *mut _) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => Ok(CudnnConvDesc{ptr: ptr}),
      e => Err(CudnnError(e)),
    }
  }

  pub unsafe fn as_mut_ptr(&self) -> cudnnConvolutionDescriptor_t {
    self.ptr
  }

  pub fn set_2d(&self, pad_h: i32, pad_w: i32, stride_h: i32, stride_w: i32, dilation_h: i32, dilation_w: i32, mode: cudnnConvolutionMode_t, ty: cudnnDataType_t) -> CudnnResult<()> {
    let status = unsafe { cudnnSetConvolution2dDescriptor(
        self.ptr,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        mode,
        ty,
    ) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => Ok(()),
      e => Err(CudnnError(e)),
    }
  }

  pub fn set_group_count(&self, group_count: i32) -> CudnnResult<()> {
    let status = unsafe { cudnnSetConvolutionGroupCount(self.ptr, group_count) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => Ok(()),
      e => Err(CudnnError(e)),
    }
  }

  pub fn set_math_type(&self, math_type: cudnnMathType_t) -> CudnnResult<()> {
    let status = unsafe { cudnnSetConvolutionMathType(self.ptr, math_type) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => Ok(()),
      e => Err(CudnnError(e)),
    }
  }
}

pub trait CudnnConvExt<T> {
  unsafe fn conv_fwd(&self, alpha: T, beta: T);
  unsafe fn conv_bwd_filter(&self, alpha: T, beta: T);
  unsafe fn conv_bwd_bias(&self, alpha: T, beta: T);
  unsafe fn conv_bwd_data(&self, alpha: T, beta: T);
}

pub struct CudnnConv2dOp<T> {
  fwd_algo:             cudnnConvolutionFwdAlgo_t,
  fwd_workspace:        usize,
  bwd_filter_algo:      cudnnConvolutionBwdFilterAlgo_t,
  bwd_filter_workspace: usize,
  bwd_data_algo:        cudnnConvolutionBwdDataAlgo_t,
  bwd_data_workspace:   usize,
  _m:   PhantomData<T>,
}

impl<T> CudnnConv2dOp<T> {
  pub fn new(handle: &CudnnHandle, conv_desc: CudnnConvDesc, x_desc: CudnnTensorDesc<T>, w_desc: CudnnFilterDesc<T>, b_desc: Option<CudnnTensorDesc<T>>, y_desc: CudnnTensorDesc<T>, deterministic: bool) -> Self {
    let mut fwd_rank = None;
    let mut tmp_perfs: Vec<cudnnConvolutionFwdAlgoPerf_t> = Vec::with_capacity(10);
    for _ in 0 .. 10 {
      tmp_perfs.push(unsafe { uninitialized() });
    }
    {
      let mut perf_count: c_int = 0;
      let status = unsafe { cudnnFindConvolutionForwardAlgorithm(
          handle.ptr,
          x_desc.ptr,
          w_desc.ptr,
          conv_desc.ptr,
          y_desc.ptr,
          10,
          &mut perf_count as *mut _,
          (&mut tmp_perfs[..]).as_mut_ptr(),
      ) };
      for rank in 0 .. perf_count as usize {
        if deterministic && tmp_perfs[rank].determinism == cudnnDeterminism_t_CUDNN_DETERMINISTIC {
          fwd_rank = Some(rank);
          break;
        } else if tmp_perfs[rank].algo != cudnnConvolutionFwdAlgo_t_CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING {
          fwd_rank = Some(rank);
          break;
        }
      }
    }
    assert!(fwd_rank.is_some());
    let fwd_algo = tmp_perfs[fwd_rank.unwrap()].algo;
    let fwd_workspace = tmp_perfs[fwd_rank.unwrap()].memory;

    let mut bwd_filter_rank = None;
    let mut tmp_perfs: Vec<cudnnConvolutionBwdFilterAlgoPerf_t> = Vec::with_capacity(10);
    for _ in 0 .. 10 {
      tmp_perfs.push(unsafe { uninitialized() });
    }
    {
      let mut perf_count: c_int = 0;
      let status = unsafe { cudnnFindConvolutionBackwardFilterAlgorithm(
          handle.ptr,
          x_desc.ptr,
          y_desc.ptr,
          conv_desc.ptr,
          w_desc.ptr,
          10,
          &mut perf_count as *mut _,
          (&mut tmp_perfs[..]).as_mut_ptr(),
      ) };
      for rank in 0 .. perf_count as usize {
        if deterministic && tmp_perfs[rank].determinism == cudnnDeterminism_t_CUDNN_DETERMINISTIC {
          bwd_filter_rank = Some(rank);
          break;
        } else {
          bwd_filter_rank = Some(rank);
          break;
        }
      }
    }
    assert!(bwd_filter_rank.is_some());
    let bwd_filter_algo = tmp_perfs[bwd_filter_rank.unwrap()].algo;
    let bwd_filter_workspace = tmp_perfs[bwd_filter_rank.unwrap()].memory;

    let mut bwd_data_rank = None;
    let mut tmp_perfs: Vec<cudnnConvolutionBwdDataAlgoPerf_t> = Vec::with_capacity(10);
    for _ in 0 .. 10 {
      tmp_perfs.push(unsafe { uninitialized() });
    }
    {
      let mut perf_count: c_int = 0;
      let status = unsafe { cudnnFindConvolutionBackwardDataAlgorithm(
          handle.ptr,
          w_desc.ptr,
          y_desc.ptr,
          conv_desc.ptr,
          x_desc.ptr,
          10,
          &mut perf_count as *mut _,
          (&mut tmp_perfs[..]).as_mut_ptr(),
      ) };
      for rank in 0 .. perf_count as usize {
        if deterministic && tmp_perfs[rank].determinism == cudnnDeterminism_t_CUDNN_DETERMINISTIC {
          bwd_data_rank = Some(rank);
          break;
        } else {
          bwd_data_rank = Some(rank);
          break;
        }
      }
    }
    assert!(bwd_data_rank.is_some());
    let bwd_data_algo = tmp_perfs[bwd_data_rank.unwrap()].algo;
    let bwd_data_workspace = tmp_perfs[bwd_data_rank.unwrap()].memory;

    CudnnConv2dOp{
      fwd_algo:             fwd_algo,
      fwd_workspace:        fwd_workspace,
      bwd_filter_algo:      bwd_filter_algo,
      bwd_filter_workspace: bwd_filter_workspace,
      bwd_data_algo:        bwd_data_algo,
      bwd_data_workspace:   bwd_data_workspace,
      _m:   PhantomData,
    }
  }

  pub fn new_ex() -> Self {
    // TODO
    unimplemented!();
  }
}

pub trait CudnnPoolExt<T> {
  unsafe fn pool_fwd(&self, alpha: T, beta: T);
  unsafe fn pool_bwd(&self, alpha: T, beta: T);
}

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

impl CudnnDataTypeExt for f16_stub {
  fn cudnn_data_ty() -> cudnnDataType_t {
    cudnnDataType_t_CUDNN_DATA_HALF
  }
}

impl CudnnDataTypeExt for i8 {
  fn cudnn_data_ty() -> cudnnDataType_t {
    cudnnDataType_t_CUDNN_DATA_INT8
  }
}

impl CudnnDataTypeExt for i32 {
  fn cudnn_data_ty() -> cudnnDataType_t {
    cudnnDataType_t_CUDNN_DATA_INT32
  }
}

/*impl CudnnDataTypeExt for i8x4 {
  fn cudnn_data_ty() -> cudnnDataType_t {
    cudnnDataType_t_CUDNN_DATA_INT8x4
  }
}*/

impl CudnnDataTypeExt for u8 {
  fn cudnn_data_ty() -> cudnnDataType_t {
    cudnnDataType_t_CUDNN_DATA_UINT8
  }
}

/*impl CudnnDataTypeExt for u8x4 {
  fn cudnn_data_ty() -> cudnnDataType_t {
    cudnnDataType_t_CUDNN_DATA_UINT8x4
  }
}*/

pub struct CudnnHandle {
  ptr:  cudnnHandle_t,
}

unsafe impl Send for CudnnHandle {}
unsafe impl Sync for CudnnHandle {}

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
    let status = unsafe { cudnnSetStream(self.ptr, stream.as_mut_ptr()) };
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

unsafe impl<T> Send for CudnnTensorDesc<T> where T: CudnnDataTypeExt {}
unsafe impl<T> Sync for CudnnTensorDesc<T> where T: CudnnDataTypeExt {}

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

  pub unsafe fn as_ptr(&self) -> *const cudnnTensorStruct {
    self.ptr
  }

  pub unsafe fn as_mut_ptr(&mut self) -> cudnnTensorDescriptor_t {
    self.ptr
  }

  pub fn set_4d_nchw(&mut self, num: i32, channels: i32, height: i32, width: i32) -> CudnnResult<()> {
    let status = unsafe { cudnnSetTensor4dDescriptor(
        self.ptr,
        cudnnTensorFormat_t_CUDNN_TENSOR_NCHW,
        T::cudnn_data_ty(),
        num as _,
        channels as _,
        height as _,
        width as _,
    ) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => Ok(()),
      _ => Err(CudnnError(status)),
    }
  }
}

pub struct CudnnFilterDesc<T> {
  ptr:  cudnnFilterDescriptor_t,
  _m:   PhantomData<T>,
}

unsafe impl<T> Send for CudnnFilterDesc<T> where T: CudnnDataTypeExt {}
unsafe impl<T> Sync for CudnnFilterDesc<T> where T: CudnnDataTypeExt {}

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

  pub unsafe fn as_ptr(&self) -> *const cudnnFilterStruct {
    self.ptr
  }

  pub unsafe fn as_mut_ptr(&mut self) -> cudnnFilterDescriptor_t {
    self.ptr
  }

  pub fn set_4d_nchw(&mut self, dst_channels: i32, src_channels: i32, height: i32, width: i32) -> CudnnResult<()> {
    let status = unsafe { cudnnSetFilter4dDescriptor(
        self.ptr,
        T::cudnn_data_ty(),
        cudnnTensorFormat_t_CUDNN_TENSOR_NCHW,
        dst_channels,
        src_channels,
        height,
        width,
    ) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => Ok(()),
      _ => Err(CudnnError(status)),
    }
  }
}

pub struct CudnnConvDesc {
  ptr:  cudnnConvolutionDescriptor_t,
}

unsafe impl Send for CudnnConvDesc {}
unsafe impl Sync for CudnnConvDesc {}

impl Drop for CudnnConvDesc {
  fn drop(&mut self) {
    let status = unsafe { cudnnDestroyConvolutionDescriptor(self.ptr) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => {}
      _ => panic!("failed to destroy cudnn conv desc: {:?}", status),
    }
  }
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

  pub unsafe fn as_ptr(&self) -> *const cudnnConvolutionStruct {
    self.ptr
  }

  pub unsafe fn as_mut_ptr(&mut self) -> cudnnConvolutionDescriptor_t {
    self.ptr
  }

  pub fn set_2d(&mut self, pad_h: i32, pad_w: i32, stride_h: i32, stride_w: i32, dilation_h: i32, dilation_w: i32, mode: cudnnConvolutionMode_t, data_ty: cudnnDataType_t) -> CudnnResult<()> {
    let status = unsafe { cudnnSetConvolution2dDescriptor(
        self.ptr,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        mode,
        data_ty,
    ) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => Ok(()),
      e => Err(CudnnError(e)),
    }
  }

  pub fn set_group_count(&mut self, group_count: i32) -> CudnnResult<()> {
    let status = unsafe { cudnnSetConvolutionGroupCount(self.ptr, group_count) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => Ok(()),
      e => Err(CudnnError(e)),
    }
  }

  pub fn set_math_type(&mut self, math_type: cudnnMathType_t) -> CudnnResult<()> {
    let status = unsafe { cudnnSetConvolutionMathType(self.ptr, math_type) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => Ok(()),
      e => Err(CudnnError(e)),
    }
  }
}

pub trait CudnnConvExt<WTy, XTy, YTy> {
  type HostScalar;

  unsafe fn conv_fwd(&mut self,
      alpha: Self::HostScalar,
      x_desc: &mut CudnnTensorDesc<XTy>,
      x: *const XTy,
      w_desc: &mut CudnnFilterDesc<WTy>,
      w: *const WTy,
      conv_desc: &mut CudnnConvDesc,
      algo_desc: cudnnConvolutionFwdAlgo_t,
      workspace: *mut u8,
      workspace_size: usize,
      beta: Self::HostScalar,
      y_desc: &mut CudnnTensorDesc<YTy>,
      y: *mut YTy) -> CudnnResult<()>;
  unsafe fn conv_fwd_bias_act(&mut self,
      alpha: Self::HostScalar,
      x_desc: &mut CudnnTensorDesc<XTy>,
      x: *const XTy,
      w_desc: &mut CudnnFilterDesc<WTy>,
      w: *const WTy,
      conv_desc: &mut CudnnConvDesc,
      algo_desc: cudnnConvolutionFwdAlgo_t,
      workspace: *mut u8,
      workspace_size: usize,
      beta: Self::HostScalar,
      z_desc: &mut CudnnTensorDesc<YTy>,
      z: *const YTy,
      b_desc: &mut CudnnTensorDesc<WTy>,
      b: *const WTy,
      act_desc: &mut CudnnActDesc,
      y_desc: &mut CudnnTensorDesc<YTy>,
      y: *mut YTy) -> CudnnResult<()>;
  unsafe fn conv_bwd_bias(&mut self,
      alpha: Self::HostScalar,
      dy_desc: &mut CudnnTensorDesc<YTy>,
      dy: *const YTy,
      beta: Self::HostScalar,
      db_desc: &mut CudnnTensorDesc<WTy>,
      db: *mut WTy) -> CudnnResult<()>;
  unsafe fn conv_bwd_filter(&mut self,
      alpha: Self::HostScalar,
      x_desc: &mut CudnnTensorDesc<XTy>,
      x: *const XTy,
      dy_desc: &mut CudnnTensorDesc<YTy>,
      dy: *const YTy,
      conv_desc: &mut CudnnConvDesc,
      algo_desc: cudnnConvolutionBwdFilterAlgo_t,
      workspace: *mut u8,
      workspace_size: usize,
      beta: Self::HostScalar,
      dw_desc: &mut CudnnFilterDesc<WTy>,
      dw: *mut WTy) -> CudnnResult<()>;
  unsafe fn conv_bwd_data(&mut self,
      alpha: Self::HostScalar,
      w_desc: &mut CudnnFilterDesc<WTy>,
      w: *const WTy,
      dy_desc: &mut CudnnTensorDesc<YTy>,
      dy: *const YTy,
      conv_desc: &mut CudnnConvDesc,
      algo_desc: cudnnConvolutionBwdDataAlgo_t,
      workspace: *mut u8,
      workspace_size: usize,
      beta: Self::HostScalar,
      dx_desc: &mut CudnnTensorDesc<XTy>,
      dx: *mut XTy) -> CudnnResult<()>;
}

impl CudnnConvExt<f32, f32, f32> for CudnnHandle {
  type HostScalar = f32;

  unsafe fn conv_fwd(&mut self,
      alpha: Self::HostScalar,
      x_desc: &mut CudnnTensorDesc<f32>,
      x: *const f32,
      w_desc: &mut CudnnFilterDesc<f32>,
      w: *const f32,
      conv_desc: &mut CudnnConvDesc,
      algo_desc: cudnnConvolutionFwdAlgo_t,
      workspace: *mut u8,
      workspace_size: usize,
      beta: Self::HostScalar,
      y_desc: &mut CudnnTensorDesc<f32>,
      y: *mut f32) -> CudnnResult<()>
  {
    let status = unsafe { cudnnConvolutionForward(
        self.as_mut_ptr(),
        &alpha as *const _ as *const _,
        x_desc.as_mut_ptr(),
        x as *const _,
        w_desc.as_mut_ptr(),
        w as *const _,
        conv_desc.as_mut_ptr(),
        algo_desc,
        workspace as *mut _,
        workspace_size,
        &beta as *const _ as *const _,
        y_desc.as_mut_ptr(),
        y as *mut _,
    ) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => Ok(()),
      e => Err(CudnnError(e)),
    }
  }

  unsafe fn conv_fwd_bias_act(&mut self,
      alpha: Self::HostScalar,
      x_desc: &mut CudnnTensorDesc<f32>,
      x: *const f32,
      w_desc: &mut CudnnFilterDesc<f32>,
      w: *const f32,
      conv_desc: &mut CudnnConvDesc,
      algo_desc: cudnnConvolutionFwdAlgo_t,
      workspace: *mut u8,
      workspace_size: usize,
      beta: Self::HostScalar,
      z_desc: &mut CudnnTensorDesc<f32>,
      z: *const f32,
      b_desc: &mut CudnnTensorDesc<f32>,
      b: *const f32,
      act_desc: &mut CudnnActDesc,
      y_desc: &mut CudnnTensorDesc<f32>,
      y: *mut f32) -> CudnnResult<()>
  {
    // TODO
    unimplemented!();
  }

  unsafe fn conv_bwd_bias(&mut self,
      alpha: Self::HostScalar,
      dy_desc: &mut CudnnTensorDesc<f32>,
      dy: *const f32,
      beta: Self::HostScalar,
      db_desc: &mut CudnnTensorDesc<f32>,
      db: *mut f32) -> CudnnResult<()>
  {
    let status = unsafe { cudnnConvolutionBackwardBias(
        self.as_mut_ptr(),
        &alpha as *const _ as *const _,
        dy_desc.as_mut_ptr(),
        dy as *const _,
        &beta as *const _ as *const _,
        db_desc.as_mut_ptr(),
        db as *mut _,
    ) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => Ok(()),
      e => Err(CudnnError(e)),
    }
  }

  unsafe fn conv_bwd_filter(&mut self,
      alpha: Self::HostScalar,
      x_desc: &mut CudnnTensorDesc<f32>,
      x: *const f32,
      dy_desc: &mut CudnnTensorDesc<f32>,
      dy: *const f32,
      conv_desc: &mut CudnnConvDesc,
      algo_desc: cudnnConvolutionBwdFilterAlgo_t,
      workspace: *mut u8,
      workspace_size: usize,
      beta: Self::HostScalar,
      dw_desc: &mut CudnnFilterDesc<f32>,
      dw: *mut f32) -> CudnnResult<()>
  {
    let status = unsafe { cudnnConvolutionBackwardFilter(
        self.as_mut_ptr(),
        &alpha as *const _ as *const _,
        x_desc.as_mut_ptr(),
        x as *const _,
        dy_desc.as_mut_ptr(),
        dy as *const _,
        conv_desc.as_mut_ptr(),
        algo_desc,
        workspace as *mut _,
        workspace_size,
        &beta as *const _ as *const _,
        dw_desc.as_mut_ptr(),
        dw as *mut _,
    ) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => Ok(()),
      e => Err(CudnnError(e)),
    }
  }

  unsafe fn conv_bwd_data(&mut self,
      alpha: Self::HostScalar,
      w_desc: &mut CudnnFilterDesc<f32>,
      w: *const f32,
      dy_desc: &mut CudnnTensorDesc<f32>,
      dy: *const f32,
      conv_desc: &mut CudnnConvDesc,
      algo_desc: cudnnConvolutionBwdDataAlgo_t,
      workspace: *mut u8,
      workspace_size: usize,
      beta: Self::HostScalar,
      dx_desc: &mut CudnnTensorDesc<f32>,
      dx: *mut f32) -> CudnnResult<()>
  {
    let status = unsafe { cudnnConvolutionBackwardData(
        self.as_mut_ptr(),
        &alpha as *const _ as *const _,
        w_desc.as_mut_ptr(),
        w as *const _,
        dy_desc.as_mut_ptr(),
        dy as *const _,
        conv_desc.as_mut_ptr(),
        algo_desc,
        workspace as *mut _,
        workspace_size,
        &beta as *const _ as *const _,
        dx_desc.as_mut_ptr(),
        dx as *mut _,
    ) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => Ok(()),
      e => Err(CudnnError(e)),
    }
  }
}

/*impl CudnnConvExt<f16_stub, f16_stub, f16_stub> for CudnnHandle {
  type HostScalar = f32;

  // TODO
}*/

pub struct CudnnPoolDesc {
  ptr:  cudnnPoolingDescriptor_t,
}

unsafe impl Send for CudnnPoolDesc {}
unsafe impl Sync for CudnnPoolDesc {}

impl Drop for CudnnPoolDesc {
  fn drop(&mut self) {
    let status = unsafe { cudnnDestroyPoolingDescriptor(self.ptr) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => {}
      _ => panic!("failed to destroy cudnn pool desc: {:?}", status),
    }
  }
}

impl CudnnPoolDesc {
  pub fn create() -> CudnnResult<CudnnPoolDesc> {
    let mut ptr = null_mut();
    let status = unsafe { cudnnCreatePoolingDescriptor(&mut ptr as *mut _) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => Ok(CudnnPoolDesc{ptr: ptr}),
      e => Err(CudnnError(e)),
    }
  }

  pub unsafe fn as_ptr(&self) -> *const cudnnPoolingStruct {
    self.ptr
  }

  pub unsafe fn as_mut_ptr(&mut self) -> cudnnPoolingDescriptor_t {
    self.ptr
  }

  pub fn set_2d(&mut self, window_h: i32, window_w: i32, pad_h: i32, pad_w: i32, stride_h: i32, stride_w: i32, mode: cudnnPoolingMode_t, nan_prop: cudnnNanPropagation_t) -> CudnnResult<()> {
    let status = unsafe { cudnnSetPooling2dDescriptor(
        self.ptr,
        mode,
        nan_prop,
        window_h, window_w,
        pad_h, pad_w,
        stride_h, stride_w,
    ) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => Ok(()),
      e => Err(CudnnError(e)),
    }
  }
}

pub trait CudnnPoolExt<T> {
  type HostScalar;

  unsafe fn pool_fwd(&mut self,
      pool_desc: &mut CudnnPoolDesc,
      alpha: Self::HostScalar,
      x_desc: &mut CudnnTensorDesc<T>,
      x: *const T,
      beta: Self::HostScalar,
      y_desc: &mut CudnnTensorDesc<T>,
      y: *mut T) -> CudnnResult<()>;
  unsafe fn pool_bwd(&mut self,
      pool_desc: &mut CudnnPoolDesc,
      alpha: Self::HostScalar,
      y_desc: &mut CudnnTensorDesc<T>,
      y: *const T,
      dy_desc: &mut CudnnTensorDesc<T>,
      dy: *const T,
      x_desc: &mut CudnnTensorDesc<T>,
      x: *const T,
      beta: Self::HostScalar,
      dx_desc: &mut CudnnTensorDesc<T>,
      dx: *mut T) -> CudnnResult<()>;
}

impl CudnnPoolExt<f32> for CudnnHandle {
  type HostScalar = f32;

  unsafe fn pool_fwd(&mut self,
      pool_desc: &mut CudnnPoolDesc,
      alpha: Self::HostScalar,
      x_desc: &mut CudnnTensorDesc<f32>,
      x: *const f32,
      beta: Self::HostScalar,
      y_desc: &mut CudnnTensorDesc<f32>,
      y: *mut f32)
  -> CudnnResult<()>
  {
    let status = unsafe { cudnnPoolingForward(
        self.as_mut_ptr(),
        pool_desc.as_mut_ptr(),
        &alpha as *const _ as *const _,
        x_desc.as_mut_ptr(),
        x as *const _,
        &beta as *const _ as *const _,
        y_desc.as_mut_ptr(),
        y as *mut _,
    ) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => Ok(()),
      e => Err(CudnnError(e)),
    }
  }

  unsafe fn pool_bwd(&mut self,
      pool_desc: &mut CudnnPoolDesc,
      alpha: Self::HostScalar,
      y_desc: &mut CudnnTensorDesc<f32>,
      y: *const f32,
      dy_desc: &mut CudnnTensorDesc<f32>,
      dy: *const f32,
      x_desc: &mut CudnnTensorDesc<f32>,
      x: *const f32,
      beta: Self::HostScalar,
      dx_desc: &mut CudnnTensorDesc<f32>,
      dx: *mut f32)
  -> CudnnResult<()>
  {
    let status = unsafe { cudnnPoolingBackward(
        self.as_mut_ptr(),
        pool_desc.as_mut_ptr(),
        &alpha as *const _ as *const _,
        y_desc.as_mut_ptr(),
        y as *const _,
        dy_desc.as_mut_ptr(),
        dy as *const _,
        x_desc.as_mut_ptr(),
        x as *const _,
        &beta as *const _ as *const _,
        dx_desc.as_mut_ptr(),
        dx as *mut _,
    ) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => Ok(()),
      e => Err(CudnnError(e)),
    }
  }
}

pub struct CudnnActDesc {
  ptr:  cudnnActivationDescriptor_t,
}

unsafe impl Send for CudnnActDesc {}
unsafe impl Sync for CudnnActDesc {}

impl Drop for CudnnActDesc {
  fn drop(&mut self) {
    let status = unsafe { cudnnDestroyActivationDescriptor(self.ptr) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => {}
      _ => panic!("failed to destroy cudnn act desc: {:?}", status),
    }
  }
}

impl CudnnActDesc {
  pub fn create() -> CudnnResult<CudnnActDesc> {
    let mut ptr = null_mut();
    let status = unsafe { cudnnCreateActivationDescriptor(&mut ptr as *mut _) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => Ok(CudnnActDesc{ptr: ptr}),
      e => Err(CudnnError(e)),
    }
  }

  pub unsafe fn as_ptr(&self) -> *const cudnnActivationStruct {
    self.ptr
  }

  pub unsafe fn as_mut_ptr(&mut self) -> cudnnActivationDescriptor_t {
    self.ptr
  }

  pub fn set(&mut self, coef: f64, mode: cudnnActivationMode_t, nan_prop: cudnnNanPropagation_t) -> CudnnResult<()> {
    let status = unsafe { cudnnSetActivationDescriptor(
        self.ptr,
        mode,
        nan_prop,
        coef,
    ) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => Ok(()),
      e => Err(CudnnError(e)),
    }
  }
}

pub trait CudnnSoftmaxExt<T> {
  type HostScalar;

  unsafe fn softmax_fwd(&mut self,
      algo: cudnnSoftmaxAlgorithm_t,
      mode: cudnnSoftmaxMode_t,
      alpha: Self::HostScalar,
      x_desc: &mut CudnnTensorDesc<T>,
      x: *const T,
      beta: Self::HostScalar,
      y_desc: &mut CudnnTensorDesc<T>,
      y: *mut T) -> CudnnResult<()>;
  unsafe fn softmax_bwd(&mut self,
      algo: cudnnSoftmaxAlgorithm_t,
      mode: cudnnSoftmaxMode_t,
      alpha: Self::HostScalar,
      y_desc: &mut CudnnTensorDesc<T>,
      y: *const T,
      dy_desc: &mut CudnnTensorDesc<T>,
      dy: *const T,
      beta: Self::HostScalar,
      dx_desc: &mut CudnnTensorDesc<T>,
      dx: *mut T) -> CudnnResult<()>;
}

impl CudnnSoftmaxExt<f32> for CudnnHandle {
  type HostScalar = f32;

  unsafe fn softmax_fwd(&mut self,
      algo: cudnnSoftmaxAlgorithm_t,
      mode: cudnnSoftmaxMode_t,
      alpha: Self::HostScalar,
      x_desc: &mut CudnnTensorDesc<f32>,
      x: *const f32,
      beta: Self::HostScalar,
      y_desc: &mut CudnnTensorDesc<f32>,
      y: *mut f32) -> CudnnResult<()>
  {
    let status = unsafe { cudnnSoftmaxForward(
        self.as_mut_ptr(),
        algo,
        mode,
        &alpha as *const _ as *const _,
        x_desc.as_mut_ptr(),
        x as *const _,
        &beta as *const _ as *const _,
        y_desc.as_mut_ptr(),
        y as *mut _,
    ) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => Ok(()),
      e => Err(CudnnError(e)),
    }
  }

  unsafe fn softmax_bwd(&mut self,
      algo: cudnnSoftmaxAlgorithm_t,
      mode: cudnnSoftmaxMode_t,
      alpha: Self::HostScalar,
      y_desc: &mut CudnnTensorDesc<f32>,
      y: *const f32,
      dy_desc: &mut CudnnTensorDesc<f32>,
      dy: *const f32,
      beta: Self::HostScalar,
      dx_desc: &mut CudnnTensorDesc<f32>,
      dx: *mut f32) -> CudnnResult<()>
  {
    let status = unsafe { cudnnSoftmaxBackward(
        self.as_mut_ptr(),
        algo,
        mode,
        &alpha as *const _ as *const _,
        y_desc.as_mut_ptr(),
        y as *const _,
        dy_desc.as_mut_ptr(),
        dy as *const _,
        &beta as *const _ as *const _,
        dx_desc.as_mut_ptr(),
        dx as *mut _,
    ) };
    match status {
      cudnnStatus_t_CUDNN_STATUS_SUCCESS => Ok(()),
      e => Err(CudnnError(e)),
    }
  }
}

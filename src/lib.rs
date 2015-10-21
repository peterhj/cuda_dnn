use ffi::*;

use cuda::runtime::{CudaStream};

use libc::{c_void, c_int, size_t};
use std::marker::{PhantomData};
use std::ptr::{null_mut};

extern crate cuda;
extern crate libc;

pub mod ffi;

pub trait CudnnDataTypeExt {
  fn data_ty() -> cudnnDataType_t;
}

impl CudnnDataTypeExt for f32 {
  fn data_ty() -> cudnnDataType_t {
    cudnnDataType_t::Float
  }
}

impl CudnnDataTypeExt for f64 {
  fn data_ty() -> cudnnDataType_t {
    cudnnDataType_t::Double
  }
}

pub type CudnnResult<T> = Result<T, cudnnStatus_t>;

fn new_result<T>(value: T, status: cudnnStatus_t) -> CudnnResult<T> {
  match status {
    cudnnStatus_t::Success => Ok(value),
    x => Err(x),
  }
}

pub struct CudnnHandle {
  ptr: cudnnHandle_t,
}

impl CudnnHandle {
  pub fn create() -> CudnnResult<CudnnHandle> {
    let mut inner: cudnnHandle_t = null_mut();
    let status = unsafe { cudnnCreate(&mut inner as *mut cudnnHandle_t) };
    new_result(CudnnHandle{ptr: inner}, status)
  }

  pub fn set_stream(&self, stream: &CudaStream) -> CudnnResult<()> {
    new_result((), unsafe { cudnnSetStream(self.ptr, stream.ptr) })
  }
}

impl Drop for CudnnHandle {
  fn drop(&mut self) {
    match unsafe { cudnnDestroy(self.ptr) } {
      cudnnStatus_t::Success => {}
      x => panic!("PANIC: failed to destroy cudnn handle: {:?}", x),
    }
  }
}

pub struct CudnnTensorDesc<T> where T: CudnnDataTypeExt {
  ptr: cudnnTensorDescriptor_t,
  _marker: PhantomData<T>,
}

impl<T> CudnnTensorDesc<T> where T: CudnnDataTypeExt {
  pub fn create_4d(width: usize, height: usize, channels: usize, num: usize) -> CudnnResult<CudnnTensorDesc<f32>> {
    let mut inner: cudnnTensorDescriptor_t = null_mut();
    let status = unsafe { cudnnCreateTensorDescriptor(&mut inner as *mut _) };
    new_result(CudnnTensorDesc{ptr: inner, _marker: PhantomData}, status)
      .and_then(|desc| {
        let status = unsafe { cudnnSetTensor4dDescriptor(
            desc.ptr,
            // FIXME(20151001): may want to specify data layout.
            cudnnTensorFormat_t::RowMajorNCHW,
            T::data_ty(),
            num as c_int, channels as c_int, height as c_int, width as c_int,
        ) };
        new_result(desc, status)
      })
  }
}

impl<T> Drop for CudnnTensorDesc<T> where T: CudnnDataTypeExt {
  fn drop(&mut self) {
    match unsafe { cudnnDestroyTensorDescriptor(self.ptr) } {
      cudnnStatus_t::Success => {}
      x => panic!("PANIC: failed to destroy cudnn tensor desc: {:?}", x),
    }
  }
}

pub struct CudnnFilterDesc<T> where T: CudnnDataTypeExt {
  ptr: cudnnFilterDescriptor_t,
  _marker: PhantomData<T>,
}

impl<T> CudnnFilterDesc<T> where T: CudnnDataTypeExt {
  pub fn create_4d(width: usize, height: usize, in_channels: usize, out_channels: usize) -> CudnnResult<CudnnFilterDesc<f32>> {
    let mut inner: cudnnFilterDescriptor_t = null_mut();
    let status = unsafe { cudnnCreateFilterDescriptor(&mut inner as *mut _) };
    new_result(CudnnFilterDesc{ptr: inner, _marker: PhantomData}, status)
      .and_then(|desc| {
        let status = unsafe { cudnnSetFilter4dDescriptor(
            desc.ptr,
            T::data_ty(),
            out_channels as c_int, in_channels as c_int, height as c_int, width as c_int,
        ) };
        new_result(desc, status)
      })
  }
}

impl<T> Drop for CudnnFilterDesc<T> where T: CudnnDataTypeExt {
  fn drop(&mut self) {
    match unsafe { cudnnDestroyFilterDescriptor(self.ptr) } {
      cudnnStatus_t::Success => {}
      x => panic!("PANIC: failed to destroy cudnn filter desc: {:?}", x),
    }
  }
}

pub struct CudnnConvDesc {
  ptr: cudnnConvolutionDescriptor_t,
}

impl CudnnConvDesc {
  pub fn create_2d_symmetric(stride: usize, pad: usize) -> CudnnResult<CudnnConvDesc> {
    CudnnConvDesc::create_2d(stride, stride, pad, pad)
  }

  pub fn create_2d(stride_w: usize, stride_h: usize, pad_w: usize, pad_h: usize) -> CudnnResult<CudnnConvDesc> {
    let mut inner: cudnnConvolutionDescriptor_t = null_mut();
    let status = unsafe { cudnnCreateConvolutionDescriptor(&mut inner as *mut _) };
    new_result(CudnnConvDesc{ptr: inner}, status)
      .and_then(|desc| {
        let status = unsafe { cudnnSetConvolution2dDescriptor(
            desc.ptr,
            // XXX(20151001): be careful about the argument order.
            pad_h as c_int, pad_w as c_int,
            stride_w as c_int, stride_h as c_int,
            1, 1,
            // FIXME(20151001): may want to specify this.
            cudnnConvolutionMode_t::CrossCorrelation,
        ) };
        new_result(desc, status)
      })
  }
}

impl Drop for CudnnConvDesc {
  fn drop(&mut self) {
    match unsafe { cudnnDestroyConvolutionDescriptor(self.ptr) } {
      cudnnStatus_t::Success => {}
      x => panic!("PANIC: failed to destroy cudnn conv desc: {:?}", x),
    }
  }
}

pub struct CudnnConvFwdOp {
  pub algo:         cudnnConvolutionFwdAlgo_t,
  pub work_size:    usize,
  pub time_ms:      f32,
  pub src_desc:     CudnnTensorDesc<f32>,
  pub filter_desc:  CudnnFilterDesc<f32>,
  pub conv_desc:    CudnnConvDesc,
  pub dst_desc:     CudnnTensorDesc<f32>,
}

impl CudnnConvFwdOp {
  pub fn create_fastest(src_desc: CudnnTensorDesc<f32>, filter_desc: CudnnFilterDesc<f32>, conv_desc: CudnnConvDesc, dst_desc: CudnnTensorDesc<f32>, handle: &CudnnHandle) -> CudnnResult<CudnnConvFwdOp> {
    let mut count: c_int = 0;
    let mut inner: cudnnConvolutionFwdAlgoPerf_t = Default::default();
    let status = unsafe { cudnnFindConvolutionForwardAlgorithm(
        handle.ptr,
        src_desc.ptr,
        filter_desc.ptr,
        conv_desc.ptr,
        dst_desc.ptr,
        1, &mut count as *mut _,
        &mut inner as *mut _,
    ) };
    //println!("DEBUG: perf: {:?}", inner);
    new_result(CudnnConvFwdOp{
      algo: inner.algo, work_size: inner.memory as usize, time_ms: inner.time,
      src_desc: src_desc,
      filter_desc: filter_desc,
      conv_desc: conv_desc,
      dst_desc: dst_desc,
    }, status)
  }

  pub unsafe fn forward(&self, in_act: *const f32, filter: *const f32, out_act: *mut f32, work_space: *mut u8, handle: &CudnnHandle) -> CudnnResult<()> {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let status = unsafe { cudnnConvolutionForward(
        handle.ptr,
        &alpha as *const f32 as *const c_void,
        self.src_desc.ptr,
        in_act as *const c_void,
        self.filter_desc.ptr,
        filter as *const c_void,
        self.conv_desc.ptr,
        self.algo,
        work_space as *mut c_void,
        self.work_size as size_t,
        &beta as *const f32 as *const c_void,
        self.dst_desc.ptr,
        out_act as *mut c_void,
    ) };
    new_result((), status)
  }
}

pub struct CudnnConvBwdFilterOp {
  pub algo:       cudnnConvolutionBwdFilterAlgo_t,
  pub work_size:  usize,
  pub time_ms:    f32,
  pub src_desc:         CudnnTensorDesc<f32>,
  pub diff_desc:        CudnnTensorDesc<f32>,
  pub conv_desc:        CudnnConvDesc,
  pub grad_filter_desc: CudnnFilterDesc<f32>,
  pub grad_bias_desc:   CudnnTensorDesc<f32>,
}

impl CudnnConvBwdFilterOp {
  pub fn create_fastest(src_desc: CudnnTensorDesc<f32>, diff_desc: CudnnTensorDesc<f32>, conv_desc: CudnnConvDesc, grad_filter_desc: CudnnFilterDesc<f32>, grad_bias_desc: CudnnTensorDesc<f32>, handle: &CudnnHandle) -> CudnnResult<CudnnConvBwdFilterOp> {
    let mut count: c_int = 0;
    let mut inner: cudnnConvolutionBwdFilterAlgoPerf_t = Default::default();
    let status = unsafe { cudnnFindConvolutionBackwardFilterAlgorithm(
        handle.ptr,
        src_desc.ptr,
        diff_desc.ptr,
        conv_desc.ptr,
        grad_filter_desc.ptr,
        1, &mut count as *mut _,
        &mut inner as *mut _,
    ) };
    //println!("DEBUG: perf: {:?}", inner);
    new_result(CudnnConvBwdFilterOp{
      algo: inner.algo, work_size: inner.memory as usize, time_ms: inner.time,
      src_desc: src_desc,
      diff_desc: diff_desc,
      conv_desc: conv_desc,
      grad_filter_desc: grad_filter_desc,
      grad_bias_desc: grad_bias_desc,
    }, status)
  }

  pub unsafe fn backward_filter(&self, in_act: *const f32, out_delta: *const f32, grad_filter_accum: *mut f32, work_space: *mut u8, handle: &CudnnHandle) -> CudnnResult<()> {
    let alpha: f32 = 1.0;
    let beta: f32 = 1.0;
    let status = unsafe { cudnnConvolutionBackwardFilter_v3(
        handle.ptr,
        &alpha as *const f32 as *const c_void,
        self.src_desc.ptr,
        in_act as *const c_void,
        self.diff_desc.ptr,
        out_delta as *const c_void,
        self.conv_desc.ptr,
        self.algo,
        work_space as *mut c_void,
        self.work_size as size_t,
        &beta as *const f32 as *const c_void,
        self.grad_filter_desc.ptr,
        grad_filter_accum as *mut c_void,
    ) };
    new_result((), status)
  }

  pub unsafe fn backward_bias(&self, out_delta: *const f32, grad_bias_accum: *mut f32, handle: &CudnnHandle) -> CudnnResult<()> {
    let alpha: f32 = 1.0;
    let beta: f32 = 1.0;
    let status = unsafe { cudnnConvolutionBackwardBias(
        handle.ptr,
        &alpha as *const f32 as *const c_void,
        self.diff_desc.ptr,
        out_delta as *const c_void,
        &beta as *const f32 as *const c_void,
        self.grad_bias_desc.ptr,
        grad_bias_accum as *mut c_void,
    ) };
    new_result((), status)
  }
}

pub struct CudnnConvBwdDataOp {
  pub algo:         cudnnConvolutionBwdDataAlgo_t,
  pub work_size:    usize,
  pub time_ms:      f32,
  pub filter_desc:  CudnnFilterDesc<f32>,
  pub diff_desc:    CudnnTensorDesc<f32>,
  pub conv_desc:    CudnnConvDesc,
  pub grad_desc:    CudnnTensorDesc<f32>,
}

impl CudnnConvBwdDataOp {
  pub fn create_fastest(filter_desc: CudnnFilterDesc<f32>, diff_desc: CudnnTensorDesc<f32>, conv_desc: CudnnConvDesc, grad_desc: CudnnTensorDesc<f32>, handle: &CudnnHandle) -> CudnnResult<CudnnConvBwdDataOp> {
    let mut count: c_int = 0;
    let mut inner: cudnnConvolutionBwdDataAlgoPerf_t = Default::default();
    let status = unsafe { cudnnFindConvolutionBackwardDataAlgorithm(
        handle.ptr,
        filter_desc.ptr,
        diff_desc.ptr,
        conv_desc.ptr,
        grad_desc.ptr,
        1, &mut count as *mut _,
        &mut inner as *mut _,
    ) };
    //println!("DEBUG: perf: {:?}", inner);
    new_result(CudnnConvBwdDataOp{
      algo: inner.algo, work_size: inner.memory as usize, time_ms: inner.time,
      filter_desc: filter_desc,
      diff_desc: diff_desc,
      conv_desc: conv_desc,
      grad_desc: grad_desc,
    }, status)
  }

  pub unsafe fn backward_data(&self, filter: *const f32, out_delta: *const f32, in_delta: *mut f32, work_space: *mut u8, handle: &CudnnHandle) -> CudnnResult<()> {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let status = unsafe { cudnnConvolutionBackwardData_v3(
        handle.ptr,
        &alpha as *const f32 as *const c_void,
        self.filter_desc.ptr,
        filter as *const c_void,
        self.diff_desc.ptr,
        out_delta as *const c_void,
        self.conv_desc.ptr,
        self.algo,
        work_space as *mut c_void,
        self.work_size as size_t,
        &beta as *const f32 as *const c_void,
        self.grad_desc.ptr,
        in_delta as *mut c_void,
    ) };
    new_result((), status)
  }
}

/*pub fn find_fastest_conv_fwd_alg(src_desc: &CudnnTensorDesc<f32>, filter_desc: &CudnnFilterDesc<f32>, conv_desc: &CudnnConvDesc, dst_desc: &CudnnTensorDesc<f32>,  handle: &CudnnHandle) -> CudnnResult<cudnnConvolutionFwdAlgo_t> {
  let mut count: c_int = 0;
  let mut alg: cudnnConvolutionFwdAlgo_t = Default::default();
  let status = unsafe { cudnnGetConvolutionForwardAlgorithm(
      handle.ptr,
      src_desc.ptr,
      filter_desc.ptr,
      conv_desc.ptr,
      dst_desc.ptr,
      cudnnConvolutionFwdPreference_t::PreferFastest,
      -1usize as size_t,
      &mut alg as *mut _,
  ) };
  println!("DEBUG: fastest alg: {:?}", alg);
  new_result(alg, status)
}*/

pub struct CudnnAddOp {
  bias_desc:    CudnnTensorDesc<f32>,
  src_dst_desc: CudnnTensorDesc<f32>,
}

impl CudnnAddOp {
  pub fn new(bias_desc: CudnnTensorDesc<f32>, src_dst_desc: CudnnTensorDesc<f32>) -> CudnnAddOp {
    CudnnAddOp{
      bias_desc:    bias_desc,
      src_dst_desc: src_dst_desc,
    }
  }

  pub unsafe fn forward(&self, bias: *const f32, src_dst: *mut f32, handle: &CudnnHandle) -> CudnnResult<()> {
    let alpha: f32 = 1.0;
    let beta: f32 = 1.0;
    let status = unsafe { cudnnAddTensor_v3(
        handle.ptr,
        &alpha as *const f32 as *const c_void,
        self.bias_desc.ptr,
        bias as *const c_void,
        &beta as *const f32 as *const c_void,
        self.src_dst_desc.ptr,
        src_dst as *mut c_void,
    ) };
    new_result((), status)
  }
}

#[derive(Clone, Copy)]
pub enum CudnnActKind {
  Relu,
  Sigmoid,
  Tanh,
}

impl CudnnActKind {
  pub fn to_cudnn(&self) -> cudnnActivationMode_t {
    match *self {
      CudnnActKind::Relu    => cudnnActivationMode_t::Relu,
      CudnnActKind::Sigmoid => cudnnActivationMode_t::Sigmoid,
      CudnnActKind::Tanh    => cudnnActivationMode_t::Tanh,
    }
  }
}

pub struct CudnnActOp {
  mode:           cudnnActivationMode_t,
  src_desc:       CudnnTensorDesc<f32>,
  src_diff_desc:  CudnnTensorDesc<f32>,
  dst_desc:       CudnnTensorDesc<f32>,
  dst_diff_desc:  CudnnTensorDesc<f32>,
}

impl CudnnActOp {
  pub fn new(kind: CudnnActKind, in_act_desc: CudnnTensorDesc<f32>, in_delta_desc: CudnnTensorDesc<f32>, out_act_desc: CudnnTensorDesc<f32>, out_delta_desc: CudnnTensorDesc<f32>) -> CudnnActOp {
    CudnnActOp{
      mode:           kind.to_cudnn(),
      src_desc:       in_act_desc,
      src_diff_desc:  out_delta_desc,
      dst_desc:       out_act_desc,
      dst_diff_desc:  in_delta_desc,
    }
  }

  pub unsafe fn forward_in_place(&self, out_act: *mut f32, handle: &CudnnHandle) -> CudnnResult<()> {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let status = unsafe { cudnnActivationForward(
        handle.ptr,
        self.mode,
        &alpha as *const f32 as *const c_void,
        self.src_desc.ptr,
        out_act as *const c_void,
        &beta as *const f32 as *const c_void,
        self.dst_desc.ptr,
        out_act as *mut c_void,
    ) };
    new_result((), status)
  }

  pub unsafe fn backward_in_place(&self, in_act: *const f32, out_act: *const f32, out_delta: *mut f32, handle: &CudnnHandle) -> CudnnResult<()> {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let status = unsafe { cudnnActivationBackward(
        handle.ptr,
        self.mode,
        &alpha as *const f32 as *const c_void,
        self.src_desc.ptr,
        in_act as *const c_void,
        self.src_diff_desc.ptr,
        out_delta as *const c_void,
        self.dst_desc.ptr,
        out_act as *const c_void,
        &beta as *const f32 as *const c_void,
        self.dst_diff_desc.ptr,
        out_delta as *mut c_void,
    ) };
    new_result((), status)
  }
}

pub struct CudnnSoftmaxOp {
  src_desc:       CudnnTensorDesc<f32>,
  //src_diff_desc:  CudnnTensorDesc<f32>,
  dst_desc:       CudnnTensorDesc<f32>,
  //dst_diff_desc:  CudnnTensorDesc<f32>,
}

impl CudnnSoftmaxOp {
  pub fn new(in_act_desc: CudnnTensorDesc<f32>, prob_act_desc: CudnnTensorDesc<f32>) -> CudnnSoftmaxOp {
    CudnnSoftmaxOp{
      src_desc: in_act_desc,
      dst_desc: prob_act_desc,
    }
  }

  pub unsafe fn forward(&self, in_act: *const f32, out_act: *mut f32, handle: &CudnnHandle) -> CudnnResult<()> {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let status = unsafe { cudnnSoftmaxForward(
        handle.ptr,
        cudnnSoftmaxAlgorithm_t::Accurate,
        cudnnSoftmaxMode_t::Instance,
        &alpha as *const f32 as *const c_void,
        self.src_desc.ptr,
        in_act as *const c_void,
        &beta as *const f32 as *const c_void,
        self.dst_desc.ptr,
        out_act as *mut c_void,
    ) };
    new_result((), status)
  }

  // FIXME(20151016): the notation here is really unclear.
  /*pub unsafe fn backward(&self, out_act: *const f32, out_delta: *const f32, in_delta: *mut f32, handle: &CudnnHandle) -> CudnnResult<()> {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let status = unsafe { cudnnSoftmaxForward(
        handle.ptr,
        cudnnSoftmaxAlgorithm_t::Accurate,
        cudnnSoftmaxMode_t::Instance,
        &alpha as *const f32 as *const c_void,
        self.dst_desc.ptr,
        out_act as *const c_void,
        self.src_diff_desc.ptr,
        out_delta as *const c_void,
        &beta as *const f32 as *const c_void,
        self.dst_diff_desc.ptr,
        in_act as *mut c_void,
    ) };
    new_result((), status)
  }*/
}

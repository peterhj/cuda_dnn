use v5::ffi::*;

use cuda::runtime::{CudaStream};
use float::stub::{f16_stub};

use libc::{c_void, c_int, size_t};
use std::cmp::{max};
use std::marker::{PhantomData};
use std::ptr::{null_mut};

pub mod ffi;

pub trait CudnnDataTypeExt {
  fn data_ty() -> cudnnDataType_t;
}

impl CudnnDataTypeExt for f16_stub {
  fn data_ty() -> cudnnDataType_t {
    cudnnDataType_t::Half
  }
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

#[derive(Clone, Copy, Debug)]
pub enum CudnnTensorLayout {
  NHWC,
  NCHW,
}

pub struct CudnnTensorDesc<T> where T: CudnnDataTypeExt {
  pub ptr:          cudnnTensorDescriptor_t,
  layout:       CudnnTensorLayout,
  width:        usize,
  height:       usize,
  channels:     usize,
  batch_size:   usize,
  s_width:      usize,
  s_height:     usize,
  s_channels:   usize,
  s_batch_size: usize,
  _marker:      PhantomData<T>,
}

impl<T> CudnnTensorDesc<T> where T: CudnnDataTypeExt {
  pub fn create_4d(layout: CudnnTensorLayout, width: usize, height: usize, channels: usize, num: usize) -> CudnnResult<CudnnTensorDesc<f32>> {
    let mut inner: cudnnTensorDescriptor_t = null_mut();
    let status = unsafe { cudnnCreateTensorDescriptor(&mut inner as *mut _) };
    if status.is_err() {
      return Err(status);
    }
    let status = unsafe { cudnnSetTensor4dDescriptor(
        inner,
        match layout {
          CudnnTensorLayout::NCHW =>  cudnnTensorFormat_t::NCHW,
          CudnnTensorLayout::NHWC =>  cudnnTensorFormat_t::NHWC,
        },
        T::data_ty(),
        num as c_int,
        channels as c_int,
        height as c_int,
        width as c_int,
    ) };
    if status.is_err() {
      return Err(status);
    }
    /*let mut _data_ty = cudnnDataType_t::Float;
    let mut _n = 0;
    let mut _c = 0;
    let mut _h = 0;
    let mut _w = 0;
    let mut _n_stride = 0;
    let mut _c_stride = 0;
    let mut _h_stride = 0;
    let mut _w_stride = 0;
    let status = unsafe { cudnnGetTensor4dDescriptor(
        inner,
        &mut _data_ty as *mut _,
        &mut _n as *mut _,
        &mut _c as *mut _,
        &mut _h as *mut _,
        &mut _w as *mut _,
        &mut _n_stride as *mut _,
        &mut _c_stride as *mut _,
        &mut _h_stride as *mut _,
        &mut _w_stride as *mut _,
    ) };
    if status.is_err() {
      panic!("failed debug call to cudnnGetTensor4dDescriptor");
    }
    println!("DEBUG: create_4d: dims: {} {} {} {} stride: {} {} {} {}",
        _w, _h, _c, _n,
        _w_stride, _h_stride, _c_stride, _n_stride,
    );*/
    Ok(CudnnTensorDesc{
      ptr: inner,
      layout:   layout,
      width: width,
      height: height,
      channels: channels,
      batch_size: num,
      s_width:      1,
      s_height:     width,
      s_channels:   width * height,
      s_batch_size: width * height * channels,
      _marker: PhantomData,
    })
  }

  /*pub fn create_4d_strided(
      width: usize, height: usize, channels: usize, batch_size: usize,
      s_width: usize, s_height: usize, s_channels: usize, s_batch_size: usize,
  ) -> CudnnResult<CudnnTensorDesc<f32>> {
    let mut inner: cudnnTensorDescriptor_t = null_mut();
    let status = unsafe { cudnnCreateTensorDescriptor(&mut inner as *mut _) };
    if status.is_err() {
      return Err(status);
    }
    let status = unsafe { cudnnSetTensor4dDescriptorEx(
        inner,
        T::data_ty(),
        batch_size as c_int,
        channels as c_int,
        height as c_int,
        width as c_int,
        s_batch_size as c_int,
        s_channels as c_int,
        s_height as c_int,
        s_width as c_int,
    ) };
    if status.is_err() {
      return Err(status);
    }
    Ok(CudnnTensorDesc{
      ptr:          inner,
      width:        width,
      height:       height,
      channels:     channels,
      batch_size:   batch_size,
      s_width:      s_width,
      s_height:     s_height,
      s_channels:   s_channels,
      s_batch_size: s_batch_size,
      _marker: PhantomData,
    })
  }*/

  pub fn set_batch_size(&mut self, new_batch_size: usize) -> CudnnResult<()> {
    if new_batch_size == self.batch_size {
      Ok(())
    } else {
      let status = unsafe { cudnnSetTensor4dDescriptor(
          self.ptr,
          match self.layout {
            CudnnTensorLayout::NCHW =>  cudnnTensorFormat_t::NCHW,
            CudnnTensorLayout::NHWC =>  cudnnTensorFormat_t::NHWC,
          },
          T::data_ty(),
          new_batch_size as c_int,
          self.channels as c_int,
          self.height as c_int,
          self.width as c_int,
      ) };
      new_result((), status)
        .map(|_| { self.batch_size = new_batch_size; () })
    }
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
  pub ptr:          cudnnFilterDescriptor_t,
  width:        usize,
  height:       usize,
  in_channels:  usize,
  out_channels: usize,
  _marker:      PhantomData<T>,
}

impl<T> CudnnFilterDesc<T> where T: CudnnDataTypeExt {
  pub fn create_4d(width: usize, height: usize, in_channels: usize, out_channels: usize) -> CudnnResult<CudnnFilterDesc<f32>> {
    let mut inner: cudnnFilterDescriptor_t = null_mut();
    let status = unsafe { cudnnCreateFilterDescriptor(&mut inner as *mut _) };
    new_result(CudnnFilterDesc{
      ptr: inner,
      width: width,
      height: height,
      in_channels: in_channels,
      out_channels: out_channels,
      _marker: PhantomData,
    }, status)
      .and_then(|desc| {
        let status = unsafe { cudnnSetFilter4dDescriptor(
            desc.ptr,
            T::data_ty(),
            cudnnTensorFormat_t::NCHW,
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
  pub ptr: cudnnConvolutionDescriptor_t,
  stride_w: usize,
  stride_h: usize,
  pad_w:    usize,
  pad_h:    usize,
}

impl CudnnConvDesc {
  pub fn create_2d_symmetric(stride: usize, pad: usize) -> CudnnResult<CudnnConvDesc> {
    CudnnConvDesc::create_2d(stride, stride, pad, pad)
  }

  pub fn create_2d(stride_w: usize, stride_h: usize, pad_w: usize, pad_h: usize) -> CudnnResult<CudnnConvDesc> {
    let mut inner: cudnnConvolutionDescriptor_t = null_mut();
    let status = unsafe { cudnnCreateConvolutionDescriptor(&mut inner as *mut _) };
    new_result(CudnnConvDesc{
      ptr: inner,
      stride_w: stride_w,
      stride_h: stride_h,
      pad_w: pad_w,
      pad_h: pad_h,
    }, status)
      .and_then(|desc| {
        let status = unsafe { cudnnSetConvolution2dDescriptor(
            desc.ptr,
            // XXX(20151001): be careful about the argument order.
            pad_h as c_int, pad_w as c_int,
            stride_h as c_int, stride_w as c_int,
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
  pub fn create_fastest(mut src_desc: CudnnTensorDesc<f32>, filter_desc: CudnnFilterDesc<f32>, conv_desc: CudnnConvDesc, mut dst_desc: CudnnTensorDesc<f32>, handle: &CudnnHandle) -> CudnnResult<CudnnConvFwdOp> {
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

    Self::create_algo(inner.algo, src_desc, filter_desc, conv_desc, dst_desc, handle)

    /*let mut workspace_size = inner.memory as usize;
    let batch_size = src_desc.batch_size;
    assert_eq!(batch_size, dst_desc.batch_size);
    for s in 1 .. batch_size + 1 {
      src_desc.set_batch_size(s).unwrap();
      dst_desc.set_batch_size(s).unwrap();
      let mut tmp_size = 0;
      let status = unsafe { cudnnGetConvolutionForwardWorkspaceSize(
          handle.ptr,
          //tmp_src_desc.ptr,
          src_desc.ptr,
          filter_desc.ptr,
          conv_desc.ptr,
          //tmp_dst_desc.ptr,
          dst_desc.ptr,
          inner.algo,
          &mut tmp_size as *mut _,
      ) };
      assert!(status.is_ok());
      workspace_size = max(workspace_size, tmp_size);
    }
    src_desc.set_batch_size(batch_size).unwrap();
    dst_desc.set_batch_size(batch_size).unwrap();

    new_result(CudnnConvFwdOp{
      algo: inner.algo,
      //work_size: inner.memory as usize,
      work_size: workspace_size,
      time_ms: inner.time,
      src_desc: src_desc,
      filter_desc: filter_desc,
      conv_desc: conv_desc,
      dst_desc: dst_desc,
    }, status)*/
  }

  pub fn create_algo(algo: cudnnConvolutionFwdAlgo_t, mut src_desc: CudnnTensorDesc<f32>, filter_desc: CudnnFilterDesc<f32>, conv_desc: CudnnConvDesc, mut dst_desc: CudnnTensorDesc<f32>, handle: &CudnnHandle) -> CudnnResult<CudnnConvFwdOp> {
    let mut status = cudnnStatus_t::Success;

    let mut workspace_size: usize = 0;
    let batch_size = src_desc.batch_size;
    assert_eq!(batch_size, dst_desc.batch_size);
    for s in (1 .. batch_size + 1).rev() {
      src_desc.set_batch_size(s).unwrap();
      dst_desc.set_batch_size(s).unwrap();
      let mut tmp_size = 0;
      status = unsafe { cudnnGetConvolutionForwardWorkspaceSize(
          handle.ptr,
          //tmp_src_desc.ptr,
          src_desc.ptr,
          filter_desc.ptr,
          conv_desc.ptr,
          //tmp_dst_desc.ptr,
          dst_desc.ptr,
          algo,
          &mut tmp_size as *mut _,
      ) };
      assert!(status.is_ok());
      workspace_size = max(workspace_size, tmp_size);
    }
    src_desc.set_batch_size(batch_size).unwrap();
    dst_desc.set_batch_size(batch_size).unwrap();

    new_result(CudnnConvFwdOp{
      algo: algo,
      //work_size: inner.memory as usize,
      work_size: workspace_size,
      time_ms: 0.0,
      src_desc: src_desc,
      filter_desc: filter_desc,
      conv_desc: conv_desc,
      dst_desc: dst_desc,
    }, status)
  }

  pub unsafe fn forward(&self, scale: f32, in_act: *const f32, filter: *const f32, prev_scale: f32, out_act: *mut f32, work_space: *mut u8, handle: &CudnnHandle) -> CudnnResult<()> {
    let alpha: f32 = scale;
    let beta: f32 = prev_scale;
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

  pub fn set_batch_size(&mut self, new_batch_size: usize) -> CudnnResult<()> {
    let res = self.src_desc.set_batch_size(new_batch_size);
    if res.is_err() {
      return res;
    }
    let res = self.dst_desc.set_batch_size(new_batch_size);
    res
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
  pub fn create_fastest(mut src_desc: CudnnTensorDesc<f32>, mut diff_desc: CudnnTensorDesc<f32>, conv_desc: CudnnConvDesc, grad_filter_desc: CudnnFilterDesc<f32>, grad_bias_desc: CudnnTensorDesc<f32>, handle: &CudnnHandle) -> CudnnResult<CudnnConvBwdFilterOp> {
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

    Self::create_algo(inner.algo, src_desc, diff_desc, conv_desc, grad_filter_desc, grad_bias_desc, handle)

    /*let mut workspace_size = inner.memory as usize;
    let batch_size = src_desc.batch_size;
    assert_eq!(batch_size, diff_desc.batch_size);
    for s in 1 .. batch_size + 1 {
      src_desc.set_batch_size(s).unwrap();
      diff_desc.set_batch_size(s).unwrap();
      let mut tmp_size = 0;
      let status = unsafe { cudnnGetConvolutionBackwardFilterWorkspaceSize(
          handle.ptr,
          src_desc.ptr,
          diff_desc.ptr,
          conv_desc.ptr,
          grad_filter_desc.ptr,
          inner.algo,
          &mut tmp_size as *mut _,
      ) };
      assert!(status.is_ok());
      workspace_size = max(workspace_size, tmp_size);
    }
    src_desc.set_batch_size(batch_size).unwrap();
    diff_desc.set_batch_size(batch_size).unwrap();

    new_result(CudnnConvBwdFilterOp{
      algo: inner.algo,
      //work_size: inner.memory as usize,
      work_size: workspace_size,
      time_ms: inner.time,
      src_desc: src_desc,
      diff_desc: diff_desc,
      conv_desc: conv_desc,
      grad_filter_desc: grad_filter_desc,
      grad_bias_desc: grad_bias_desc,
    }, status)*/
  }

  pub fn create_algo(algo: cudnnConvolutionBwdFilterAlgo_t, mut src_desc: CudnnTensorDesc<f32>, mut diff_desc: CudnnTensorDesc<f32>, conv_desc: CudnnConvDesc, grad_filter_desc: CudnnFilterDesc<f32>, grad_bias_desc: CudnnTensorDesc<f32>, handle: &CudnnHandle) -> CudnnResult<CudnnConvBwdFilterOp> {
    let mut status = cudnnStatus_t::Success;
    let mut workspace_size = 0;
    let batch_size = src_desc.batch_size;
    assert_eq!(batch_size, diff_desc.batch_size);
    for s in 1 .. batch_size + 1 {
      src_desc.set_batch_size(s).unwrap();
      diff_desc.set_batch_size(s).unwrap();
      let mut tmp_size = 0;
      let status = unsafe { cudnnGetConvolutionBackwardFilterWorkspaceSize(
          handle.ptr,
          src_desc.ptr,
          diff_desc.ptr,
          conv_desc.ptr,
          grad_filter_desc.ptr,
          algo,
          &mut tmp_size as *mut _,
      ) };
      assert!(status.is_ok());
      workspace_size = max(workspace_size, tmp_size);
    }
    src_desc.set_batch_size(batch_size).unwrap();
    diff_desc.set_batch_size(batch_size).unwrap();

    new_result(CudnnConvBwdFilterOp{
      algo: algo,
      work_size: workspace_size,
      time_ms: 0.0,
      src_desc: src_desc,
      diff_desc: diff_desc,
      conv_desc: conv_desc,
      grad_filter_desc: grad_filter_desc,
      grad_bias_desc: grad_bias_desc,
    }, status)
  }

  pub unsafe fn backward_filter(&self, scale: f32, in_act: *const f32, out_delta: *const f32, prev_scale: f32, grad_filter_accum: *mut f32, work_space: *mut u8, handle: &CudnnHandle) -> CudnnResult<()> {
    let alpha: f32 = scale;
    let beta: f32 = prev_scale;
    let status = unsafe { cudnnConvolutionBackwardFilter(
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

  pub unsafe fn backward_bias(&self, scale: f32, out_delta: *const f32, prev_scale: f32, grad_bias_accum: *mut f32, handle: &CudnnHandle) -> CudnnResult<()> {
    let alpha: f32 = scale;
    let beta: f32 = prev_scale;
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

  pub fn set_batch_size(&mut self, new_batch_size: usize) -> CudnnResult<()> {
    let res = self.src_desc.set_batch_size(new_batch_size);
    if res.is_err() {
      return res;
    }
    let res = self.diff_desc.set_batch_size(new_batch_size);
    res
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
  pub fn create_fastest(filter_desc: CudnnFilterDesc<f32>, mut diff_desc: CudnnTensorDesc<f32>, conv_desc: CudnnConvDesc, mut grad_desc: CudnnTensorDesc<f32>, handle: &CudnnHandle) -> CudnnResult<CudnnConvBwdDataOp> {
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

    Self::create_algo(inner.algo, filter_desc, diff_desc, conv_desc, grad_desc, handle)

    /*let mut workspace_size = inner.memory as usize;
    let batch_size = diff_desc.batch_size;
    assert_eq!(batch_size, grad_desc.batch_size);
    for s in 1 .. batch_size + 1 {
      diff_desc.set_batch_size(s).unwrap();
      grad_desc.set_batch_size(s).unwrap();
      let mut tmp_size = 0;
      let status = unsafe { cudnnGetConvolutionBackwardDataWorkspaceSize(
          handle.ptr,
          filter_desc.ptr,
          diff_desc.ptr,
          conv_desc.ptr,
          grad_desc.ptr,
          inner.algo,
          &mut tmp_size as *mut _,
      ) };
      assert!(status.is_ok());
      workspace_size = max(workspace_size, tmp_size);
    }
    diff_desc.set_batch_size(batch_size).unwrap();
    grad_desc.set_batch_size(batch_size).unwrap();

    new_result(CudnnConvBwdDataOp{
      algo: inner.algo,
      //work_size: inner.memory as usize,
      work_size: workspace_size,
      time_ms: inner.time,
      filter_desc: filter_desc,
      diff_desc: diff_desc,
      conv_desc: conv_desc,
      grad_desc: grad_desc,
    }, status)*/
  }

  pub fn create_algo(algo: cudnnConvolutionBwdDataAlgo_t, filter_desc: CudnnFilterDesc<f32>, mut diff_desc: CudnnTensorDesc<f32>, conv_desc: CudnnConvDesc, mut grad_desc: CudnnTensorDesc<f32>, handle: &CudnnHandle) -> CudnnResult<CudnnConvBwdDataOp> {
    let mut status = cudnnStatus_t::Success;
    let mut workspace_size = 0;
    let batch_size = diff_desc.batch_size;
    assert_eq!(batch_size, grad_desc.batch_size);
    for s in 1 .. batch_size + 1 {
      diff_desc.set_batch_size(s).unwrap();
      grad_desc.set_batch_size(s).unwrap();
      let mut tmp_size = 0;
      let status = unsafe { cudnnGetConvolutionBackwardDataWorkspaceSize(
          handle.ptr,
          filter_desc.ptr,
          diff_desc.ptr,
          conv_desc.ptr,
          grad_desc.ptr,
          algo,
          &mut tmp_size as *mut _,
      ) };
      assert!(status.is_ok());
      workspace_size = max(workspace_size, tmp_size);
    }
    diff_desc.set_batch_size(batch_size).unwrap();
    grad_desc.set_batch_size(batch_size).unwrap();

    new_result(CudnnConvBwdDataOp{
      algo: algo,
      work_size: workspace_size,
      time_ms: 0.0,
      filter_desc: filter_desc,
      diff_desc: diff_desc,
      conv_desc: conv_desc,
      grad_desc: grad_desc,
    }, status)
  }

  pub unsafe fn backward_data(&self, scale: f32, filter: *const f32, out_delta: *const f32, prev_scale: f32, in_delta: *mut f32, work_space: *mut u8, handle: &CudnnHandle) -> CudnnResult<()> {
    let alpha: f32 = scale;
    let beta: f32 = prev_scale;
    let status = unsafe { cudnnConvolutionBackwardData(
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

  pub fn set_batch_size(&mut self, new_batch_size: usize) -> CudnnResult<()> {
    let res = self.diff_desc.set_batch_size(new_batch_size);
    if res.is_err() {
      return res;
    }
    let res = self.grad_desc.set_batch_size(new_batch_size);
    res
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

  pub fn set_batch_size(&mut self, new_batch_size: usize) -> CudnnResult<()> {
    let res = self.src_dst_desc.set_batch_size(new_batch_size);
    res
  }

  pub unsafe fn forward(&self, alpha: f32, bias: *const f32, beta: f32, src_dst: *mut f32, handle: &CudnnHandle) -> CudnnResult<()> {
    /*let alpha: f32 = 1.0;
    let beta: f32 = 1.0;*/
    let status = unsafe { cudnnAddTensor(
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
  Sigmoid,
  Relu,
  Tanh,
  ClippedRelu,
}

impl CudnnActKind {
  pub fn to_cudnn(&self) -> cudnnActivationMode_t {
    match *self {
      CudnnActKind::Relu        => cudnnActivationMode_t::Relu,
      CudnnActKind::Sigmoid     => cudnnActivationMode_t::Sigmoid,
      CudnnActKind::Tanh        => cudnnActivationMode_t::Tanh,
      CudnnActKind::ClippedRelu => cudnnActivationMode_t::ClippedRelu,
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
    let status = unsafe { cudnnActivationForward_v3(
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
    let status = unsafe { cudnnActivationBackward_v3(
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

  pub fn set_batch_size(&mut self, new_batch_size: usize) -> CudnnResult<()> {
    let res = self.src_desc.set_batch_size(new_batch_size);
    if res.is_err() {
      return res;
    }
    let res = self.dst_desc.set_batch_size(new_batch_size);
    res
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

  /*pub unsafe fn backward(&self, in_act: *const f32, out_delta: *const f32, in_delta: *mut f32, handle: &CudnnHandle) -> CudnnResult<()> {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let status = unsafe { cudnnSoftmaxBackward(
        handle.ptr,
        cudnnSoftmaxAlgorithm_t::Accurate,
        cudnnSoftmaxMode_t::Instance,
        &alpha as *const f32 as *const c_void,
        self.src_desc.ptr,
        in_act as *const c_void,
        self.src_diff_desc.ptr,
        out_delta as *const c_void,
        &beta as *const f32 as *const c_void,
        self.dst_diff_desc.ptr,
        in_delta as *mut c_void,
    ) };
    new_result((), status)
  }*/
}

pub struct CudnnPoolingOp {
  pooling_desc: cudnnPoolingDescriptor_t,
  src_desc:         CudnnTensorDesc<f32>,
  src_diff_desc:    CudnnTensorDesc<f32>,
  dst_desc:         CudnnTensorDesc<f32>,
  dst_diff_desc:    CudnnTensorDesc<f32>,
}

impl CudnnPoolingOp {
  pub fn create_2d_symmetric(src_desc: CudnnTensorDesc<f32>, src_diff_desc: CudnnTensorDesc<f32>, dst_desc: CudnnTensorDesc<f32>, dst_diff_desc: CudnnTensorDesc<f32>, pool_size: usize, pool_stride: usize, pool_pad: usize, pooling_mode: cudnnPoolingMode_t) -> CudnnResult<CudnnPoolingOp> {
    Self::create_2d(src_desc, src_diff_desc, dst_desc, dst_diff_desc, pool_size, pool_size, pool_stride, pool_stride, pool_pad, pool_pad, pooling_mode)
  }

  pub fn create_2d(src_desc: CudnnTensorDesc<f32>, src_diff_desc: CudnnTensorDesc<f32>, dst_desc: CudnnTensorDesc<f32>, dst_diff_desc: CudnnTensorDesc<f32>, pool_width: usize, pool_height: usize, pool_stride_w: usize, pool_stride_h: usize, pool_pad_w: usize, pool_pad_h: usize, pooling_mode: cudnnPoolingMode_t) -> CudnnResult<CudnnPoolingOp> {
    let mut pooling_desc: cudnnPoolingDescriptor_t = null_mut();
    let status = unsafe { cudnnCreatePoolingDescriptor(&mut pooling_desc as *mut _) };
    let result = new_result((), status);
    if result.is_err() {
      return Err(status);
    }
    let status = unsafe { cudnnSetPooling2dDescriptor(
        pooling_desc,
        pooling_mode,
        cudnnNanPropagation_t::NotPropagateNan, // FIXME(20160330): ???
        pool_height as i32,
        pool_width as i32,
        pool_pad_h as i32,
        pool_pad_w as i32,
        pool_stride_h as i32,
        pool_stride_w as i32,
    ) };
    if result.is_err() {
      return Err(status);
    }
    Ok(CudnnPoolingOp{
      pooling_desc: pooling_desc,
      src_desc:         src_desc,
      src_diff_desc:    src_diff_desc,
      dst_desc:         dst_desc,
      dst_diff_desc:    dst_diff_desc,
    })
  }

  pub fn set_batch_size(&mut self, new_batch_size: usize) -> CudnnResult<()> {
    let res = self.src_desc.set_batch_size(new_batch_size);
    if res.is_err() {
      return res;
    }
    let res = self.src_diff_desc.set_batch_size(new_batch_size);
    if res.is_err() {
      return res;
    }
    let res = self.dst_desc.set_batch_size(new_batch_size);
    if res.is_err() {
      return res;
    }
    let res = self.dst_diff_desc.set_batch_size(new_batch_size);
    res
  }

  pub unsafe fn forward(&self, in_act: *const f32, out_act: *mut f32, handle: &CudnnHandle) -> CudnnResult<()> {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let status = unsafe { cudnnPoolingForward(
        handle.ptr,
        self.pooling_desc,
        &alpha as *const f32 as *const c_void,
        self.src_desc.ptr,
        in_act as *const c_void,
        &beta as *const f32 as *const c_void,
        self.dst_desc.ptr,
        out_act as *mut c_void,
    ) };
    new_result((), status)
  }

  pub unsafe fn backward(&self, in_act: *const f32, out_act: *const f32, out_delta: *const f32, in_delta: *mut f32, handle: &CudnnHandle) -> CudnnResult<()> {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let status = unsafe { cudnnPoolingBackward(
        handle.ptr,
        self.pooling_desc,
        &alpha as *const f32 as *const c_void,
        self.dst_desc.ptr,
        out_act as *const c_void,
        self.dst_diff_desc.ptr,
        out_delta as *const c_void,
        self.src_desc.ptr,
        in_act as *const c_void,
        &beta as *const f32 as *const c_void,
        self.src_diff_desc.ptr,
        in_delta as *mut c_void,
    ) };
    new_result((), status)
  }
}

pub struct CudnnTransformOp {
  src_desc: CudnnTensorDesc<f32>,
  dst_desc: CudnnTensorDesc<f32>,
}

impl CudnnTransformOp {
  pub fn new(src_desc: CudnnTensorDesc<f32>, dst_desc: CudnnTensorDesc<f32>) -> CudnnTransformOp {
    CudnnTransformOp{
      src_desc: src_desc,
      dst_desc: dst_desc,
    }
  }

  pub unsafe fn transform(&self, alpha: f32, src_data: *const f32, beta: f32, dst_data: *mut f32, handle: &CudnnHandle) -> CudnnResult<()> {
    let status = unsafe { cudnnTransformTensor(
        handle.ptr,
        &alpha as *const f32 as *const c_void,
        self.src_desc.ptr,
        src_data as *const c_void,
        &beta as *const f32 as *const c_void,
        self.dst_desc.ptr,
        dst_data as *mut c_void,
    ) };
    if status.is_err() {
      return Err(status);
    }
    Ok(())
  }
}

pub struct CudnnBatchNormOp {
  bn_mode:      cudnnBatchNormMode_t,
  x_desc:       CudnnTensorDesc<f32>,
  dx_desc:      CudnnTensorDesc<f32>,
  y_desc:       CudnnTensorDesc<f32>,
  dy_desc:      CudnnTensorDesc<f32>,
  stat_desc:    CudnnTensorDesc<f32>,
}

impl CudnnBatchNormOp {
  pub fn new(x_desc: CudnnTensorDesc<f32>, dx_desc: CudnnTensorDesc<f32>, y_desc: CudnnTensorDesc<f32>, dy_desc: CudnnTensorDesc<f32>, stat_desc: CudnnTensorDesc<f32>, batch_norm_mode: cudnnBatchNormMode_t) -> CudnnBatchNormOp {
    CudnnBatchNormOp{
      bn_mode:      batch_norm_mode,
      x_desc:       x_desc,
      dx_desc:      dx_desc,
      y_desc:       y_desc,
      dy_desc:      dy_desc,
      stat_desc:    stat_desc,
    }
  }

  pub fn set_batch_size(&mut self, new_batch_size: usize) -> CudnnResult<()> {
    let res = self.x_desc.set_batch_size(new_batch_size);
    if res.is_err() {
      return res;
    }
    let res = self.dx_desc.set_batch_size(new_batch_size);
    if res.is_err() {
      return res;
    }
    let res = self.y_desc.set_batch_size(new_batch_size);
    if res.is_err() {
      return res;
    }
    let res = self.dy_desc.set_batch_size(new_batch_size);
    if res.is_err() {
      return res;
    }
    Ok(())
  }

  pub unsafe fn forward_inference(&self,
      alpha: f32,
      x_data: *const f32,
      beta: f32,
      y_data: *mut f32,
      epsilon: f64,
      bn_scale: *const f32,
      bn_bias: *const f32,
      bn_cached_mean: *const f32,
      bn_cached_ivar: *const f32,
      handle: &CudnnHandle,
  ) -> CudnnResult<()> {
    let status = unsafe { cudnnBatchNormalizationForwardInference(
        handle.ptr,
        self.bn_mode,
        &alpha as *const f32 as *const _,
        &beta as *const f32 as *const _,
        self.x_desc.ptr,
        x_data as *const _,
        self.y_desc.ptr,
        y_data as *mut _,
        self.stat_desc.ptr,
        bn_scale as *const _,
        bn_bias as *const _,
        bn_cached_mean as *const _,
        bn_cached_ivar as *const _,
        epsilon,
    ) };
    if status.is_err() {
      return Err(status);
    }
    Ok(())
  }

  pub unsafe fn forward_training(&self,
      alpha: f32,
      x_data: *const f32,
      beta: f32,
      y_data: *mut f32,
      ema_factor: f64, // XXX(20160420): must be one on the very first call!
      epsilon: f64,
      bn_scale: *const f32,
      bn_bias: *const f32,
      bn_running_mean: *mut f32,
      bn_running_ivar: *mut f32,
      bn_cached_mean: *mut f32,
      bn_cached_ivar: *mut f32,
      handle: &CudnnHandle,
  ) -> CudnnResult<()> {
    let status = unsafe { cudnnBatchNormalizationForwardTraining(
        handle.ptr,
        self.bn_mode,
        &alpha as *const f32 as *const _,
        &beta as *const f32 as *const _,
        self.x_desc.ptr,
        x_data as *const _,
        self.y_desc.ptr,
        y_data as *mut _,
        self.stat_desc.ptr,
        bn_scale as *const _,
        bn_bias as *const _,
        ema_factor,
        bn_running_mean as *mut _,
        bn_running_ivar as *mut _,
        epsilon,
        bn_cached_mean as *mut _,
        bn_cached_ivar as *mut _,
    ) };
    if status.is_err() {
      return Err(status);
    }
    Ok(())
  }

  pub unsafe fn backward(&self,
      dx_alpha: f32,
      dx_beta:  f32,
      x_data: *const f32,
      dy_data: *const f32,
      dx_data: *mut f32,
      stat_alpha: f32,
      stat_beta:  f32,
      epsilon:  f64,
      bn_scale: *const f32,
      bn_scale_grad: *mut f32,
      bn_bias_grad: *mut f32,
      bn_cached_mean: *const f32,
      bn_cached_ivar: *const f32,
      handle: &CudnnHandle,
  ) -> CudnnResult<()> {
    let status = unsafe { cudnnBatchNormalizationBackward(
        handle.ptr,
        self.bn_mode,
        &dx_alpha as *const f32 as *const _,
        &dx_beta as *const f32 as *const _,
        &stat_alpha as *const f32 as *const _,
        &stat_beta as *const f32 as *const _,
        self.x_desc.ptr,
        x_data as *const _,
        self.dy_desc.ptr,
        dy_data as *const _,
        self.dx_desc.ptr,
        dx_data as *mut _,
        self.stat_desc.ptr,
        bn_scale as *const _,
        bn_scale_grad as *mut _,
        bn_bias_grad as *mut _,
        epsilon,
        bn_cached_mean as *const _,
        bn_cached_ivar as *const _,
    ) };
    if status.is_err() {
      return Err(status);
    }
    Ok(())
  }
}

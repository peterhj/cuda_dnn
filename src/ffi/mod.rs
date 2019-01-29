#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

pub use self::v::cudnn;

#[cfg(all(
    feature = "cudnn_7_4",
    any(feature = "cuda_9_0",
        feature = "cuda_9_2",
        feature = "cuda_10_0")
))]
mod v {
  pub mod cudnn {
    use cuda::ffi::driver_types::*;
    use cuda::ffi::library_types::*;
    include!("ffi/v7_4/_cudnn.rs");
  }

  const_assert_eq!(cudnn_major;   self::cudnn::CUDNN_MAJOR, 7);
  const_assert_eq!(cudnn_minor;   self::cudnn::CUDNN_MINOR, 4);
  const_assert_eq!(cudnn_patch;   self::cudnn::CUDNN_PATCHLEVEL, 2);
  const_assert_eq!(cudnn_version; self::cudnn::CUDNN_VERSION, 7402);
}

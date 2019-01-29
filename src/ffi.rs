#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

//use cuda::ffi::driver_types::{cudaStream_t};
//include!(concat!(env!("OUT_DIR"), "/cudnn_bind.rs"));

pub use self::v::cudnn;

mod v {
  pub mod cudnn {
    use cuda::ffi::driver_types::*;
    use cuda::ffi::library_types::*;
    include!("ffi/_cudnn.rs");
  }
}

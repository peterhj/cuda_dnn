extern crate cuda_dnn;

#[test]
fn check_runtime_version() {
  assert!(cuda_dnn::check_runtime_version());
}

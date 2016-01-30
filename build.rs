fn main() {
  //println!("cargo:rustc-flags=-L /usr/local/cuda/lib64 -l dylib=cudnn");
  //println!("cargo:rustc-flags=-l dylib=cudnn");

  println!("cargo:rustc-flags=-l dylib=cudnn");
}

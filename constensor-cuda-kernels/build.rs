/// Error messages
#[derive(Debug)]
pub enum Error {}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(any(feature = "cuda_half", feature = "cuda_bfloat"))]
    let compute_cap = compute_cap().unwrap();

    #[cfg(feature = "cuda_half")]
    if compute_cap < 53 {
        panic!("`half` is only supported for CUDA compute cap >= 5.3");
    }

    #[cfg(feature = "cuda_bfloat")]
    if compute_cap < 80 {
        panic!("`bfloat` is only supported for CUDA compute cap >= 8.0");
    }

    let builder = bindgen_cuda::Builder::default();
    #[cfg(feature = "half")]
    let builder = builder.arg("-DHALF");
    #[cfg(feature = "bfloat")]
    let builder = builder.arg("-DBFLOAT");

    println!("cargo:info={builder:?}");
    let bindings = builder.build_ptx().unwrap();
    bindings.write("src/lib.rs").unwrap();
}

#[cfg(any(feature = "cuda_half", feature = "cuda_bfloat"))]
fn compute_cap() -> Result<usize, Error> {
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");

    // Try to parse compute caps from env
    let compute_cap = if let Ok(compute_cap_str) = std::env::var("CUDA_COMPUTE_CAP") {
        println!("cargo:rustc-env=CUDA_COMPUTE_CAP={compute_cap_str}");
        compute_cap_str
            .parse::<usize>()
            .expect("Could not parse code")
    } else {
        // Use nvidia-smi to get the current compute cap
        let out = std::process::Command::new("nvidia-smi")
                .arg("--query-gpu=compute_cap")
                .arg("--format=csv")
                .output()
                .expect("`nvidia-smi` failed. Ensure that you have CUDA installed and that `nvidia-smi` is in your PATH.");
        let out = std::str::from_utf8(&out.stdout).expect("stdout is not a utf8 string");
        let mut lines = out.lines();
        assert_eq!(lines.next().expect("missing line in stdout"), "compute_cap");
        let cap = lines
            .next()
            .expect("missing line in stdout")
            .replace('.', "");
        let cap = cap.parse::<usize>().expect("cannot parse as int {cap}");
        println!("cargo:rustc-env=CUDA_COMPUTE_CAP={cap}");
        cap
    };

    // Grab available GPU codes from nvcc and select the highest one
    let (supported_nvcc_codes, max_nvcc_code) = {
        let out = std::process::Command::new("nvcc")
                .arg("--list-gpu-code")
                .output()
                .expect("`nvcc` failed. Ensure that you have CUDA installed and that `nvcc` is in your PATH.");
        let out = std::str::from_utf8(&out.stdout).expect("valid utf-8 nvcc output");

        let out = out.lines().collect::<Vec<&str>>();
        let mut codes = Vec::with_capacity(out.len());
        for code in out {
            let code = code.split('_').collect::<Vec<&str>>();
            if !code.is_empty() && code.contains(&"sm") {
                if let Ok(num) = code[1].parse::<usize>() {
                    codes.push(num);
                }
            }
        }
        codes.sort();
        let max_nvcc_code = *codes.last().expect("no gpu codes parsed from nvcc");
        (codes, max_nvcc_code)
    };

    // Check that nvcc supports the asked compute caps
    if !supported_nvcc_codes.contains(&compute_cap) {
        panic!(
            "nvcc cannot target gpu arch {compute_cap}. Available nvcc targets are {supported_nvcc_codes:?}."
        );
    }
    if compute_cap > max_nvcc_code {
        panic!(
            "CUDA compute cap {compute_cap} is higher than the highest gpu code from nvcc {max_nvcc_code}"
        );
    }

    Ok(compute_cap)
}

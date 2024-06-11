use std::{
    borrow::Cow,
    hash::{DefaultHasher, Hash, Hasher},
    ops::Deref,
    path::PathBuf,
    sync::Arc,
};
mod error;
use cudarc::{
    driver::{CudaFunction, CudaSlice, LaunchAsync, LaunchConfig},
    nvrtc::{CompileOptions, Ptx},
};
use error::{CudaError, WrapErr};

use crate::{
    cpu_storage::CpuStorage,
    storage::{BackendDevice, BackendStorage},
    DType, Op, Result,
};

#[derive(Clone)]
pub struct CudaDevice {
    device: Arc<cudarc::driver::CudaDevice>,
}

impl CudaDevice {
    pub(crate) fn new(ordinal: usize) -> Result<Self> {
        Ok(Self {
            device: cudarc::driver::CudaDevice::new(ordinal).w()?,
        })
    }

    pub(crate) fn get_or_load_func(&self, module_name: &str, ptx: Ptx) -> Result<CudaFunction> {
        if !self.has_func(module_name, module_name) {
            // Leaking the string here is a bit sad but we need a &'static str and this is only
            // done once per kernel name.
            let static_module_name = Box::leak(module_name.to_string().into_boxed_str());
            self.load_ptx(ptx, module_name, &[static_module_name])
                .map_err(|cuda| CudaError::Load {
                    cuda,
                    module_name: module_name.to_string(),
                })
                .w()?;
        }
        self.get_func(module_name, module_name)
            // Clippy recommends this `ok_or` rather than `ok_or_else` so hopefully the compiler is
            // able to only build the error value if needed.
            .ok_or(CudaError::MissingKernel {
                module_name: module_name.to_string(),
            })
            .w()
    }
}

impl Deref for CudaDevice {
    type Target = Arc<cudarc::driver::CudaDevice>;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

pub struct CudaStorage<T: DType> {
    slice: CudaSlice<T>,
    device: CudaDevice,
}

impl<T: DType> BackendStorage<T> for CudaStorage<T> {
    fn to_cpu_storage(&self) -> Result<Cow<CpuStorage<T>>> {
        let data = self.device.dtoh_sync_copy(&self.slice).w()?;
        Ok(Cow::Owned(CpuStorage(data)))
    }
}

#[derive(Debug)]
struct Name(usize);
impl Name {
    fn to_name(&self) -> String {
        format!("v{}", self.0)
    }
}

fn handle_node<T: DType>(
    current_name: &mut usize,
    header: &mut String,
    op: &Op<T>,
    graph: &[Op<T>],
) -> String {
    dbg!(&op);
    match op {
        Op::BinaryOp {
            l_id,
            r_id,
            operator,
        } => {
            let l_name = handle_node(current_name, header, &graph[**l_id], graph);
            let r_name = handle_node(current_name, header, &graph[**r_id], graph);
            format!("({l_name} {} {r_name})", operator.to_c_op())
        }
        Op::Fill { v } => {
            *current_name += 1;
            let name = Name(*current_name);
            *header += &format!("T {} = {v:?};\n", name.to_name());
            format!("({})", name.to_name())
        }
        Op::Arange { start, step } => {
            *current_name += 1;
            let name = Name(*current_name);
            *header += &format!(
                "T {} = static_cast<T>(i) * {step:?} + {start:?};\n",
                name.to_name()
            );
            format!("({})", name.to_name())
        }
    }
}

fn cuda_include_dir() -> Option<String> {
    // NOTE: copied from cudarc build.rs.
    let env_vars = [
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
        "CUDNN_LIB",
    ];
    #[allow(unused)]
    let env_vars = env_vars
        .into_iter()
        .map(std::env::var)
        .filter_map(std::result::Result::ok)
        .map(Into::<PathBuf>::into);

    let roots = [
        "/usr",
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/lib/cuda",
        "C:/Program Files/NVIDIA GPU Computing Toolkit",
        "C:/CUDA",
    ];

    println!("cargo:info={roots:?}");

    #[allow(unused)]
    let roots = roots.into_iter().map(Into::<PathBuf>::into);

    env_vars
        .chain(roots)
        .find(|path| path.join("include").join("cuda.h").is_file())
        .map(|x| x.to_str().unwrap().to_string())
}

impl BackendDevice for CudaDevice {
    type Storage<X: DType> = CudaStorage<X>;

    fn compile_and_run_graph<S: crate::Shape, T: DType>(
        &self,
        nodes: &[crate::Op<T>],
    ) -> Result<Self::Storage<T>> {
        let mut header = "".to_string();
        let body = handle_node(&mut 0, &mut header, nodes.last().unwrap(), nodes);

        // Module name is based on hash of body and header
        let mut hasher = DefaultHasher::new();
        body.hash(&mut hasher);
        header.hash(&mut hasher);
        let module_name = format!("jit_kernel_{}_{}", hasher.finish(), T::NAME);

        let template_kernel = format!(
            r#"
            typedef unsigned char uint8_t;
            typedef unsigned int uint32_t;
            typedef long long int int64_t;
            {}

            template <typename T>
            __device__ void {module_name}_kernel(T *buf, const size_t numel) {{
                for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;
                    i += blockDim.x * gridDim.x) {{
                    {header}
                    buf[i] = {body};
                }}
            }}
            
            extern "C" __global__ void {module_name}({} *buf, const size_t numel) {{
                {module_name}_kernel(buf, numel);
            }}

            "#,
            T::C_DEP.unwrap_or(""),
            T::C_NAME,
        );

        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            template_kernel,
            CompileOptions {
                use_fast_math: Some(true),
                include_paths: vec![cuda_include_dir().unwrap()],
                ..Default::default()
            },
        )
        .w()?;

        let n_elems = S::element_count();
        let data = unsafe { self.device.alloc::<T>(n_elems) }.w()?;

        let func = self.get_or_load_func(&module_name, ptx)?;

        let params = (&data, n_elems);
        let cfg = LaunchConfig::for_num_elems(n_elems as u32);
        unsafe { func.launch(cfg, params) }.w()?;
        Ok(CudaStorage {
            slice: data,
            device: self.clone(),
        })
    }
}

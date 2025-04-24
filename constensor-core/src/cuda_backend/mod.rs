use std::{
    borrow::Cow,
    cell::OnceCell,
    fs,
    hash::{DefaultHasher, Hash, Hasher},
    ops::Deref,
    path::{Path, PathBuf},
    sync::Arc,
};
mod error;
use cudarc::{
    driver::{CudaFunction, CudaModule, CudaSlice, LaunchConfig, PushKernelArg},
    nvrtc::{CompileOptions, Ptx},
};
use error::WrapErr;

use crate::{
    cpu_storage::CpuStorage,
    graph::GraphTensorId,
    storage::{BackendDevice, BackendStorage},
    DType, GraphNode, Op, Result,
};

#[derive(Clone)]
pub struct CudaDevice {
    context: Arc<cudarc::driver::CudaContext>,
    stream: Arc<cudarc::driver::CudaStream>,
    module: OnceCell<Arc<CudaModule>>,
}

impl CudaDevice {
    pub(crate) fn new(ordinal: usize) -> Result<Self> {
        let context = cudarc::driver::CudaContext::new(ordinal).w()?;
        let stream = context.new_stream().w()?;
        Ok(Self {
            context,
            stream,
            module: OnceCell::new(),
        })
    }

    pub(crate) fn stream(&self) -> Arc<cudarc::driver::CudaStream> {
        self.stream.clone()
    }

    pub(crate) fn get_or_load_func(&self, function_name: &str, ptx: Ptx) -> Result<CudaFunction> {
        let module = self
            .module
            .get_or_init(|| self.context.load_module(ptx).w().unwrap());
        module.load_function(function_name).w()
    }
}

impl Deref for CudaDevice {
    type Target = Arc<cudarc::driver::CudaStream>;

    fn deref(&self) -> &Self::Target {
        &self.stream
    }
}

pub struct CudaStorage<T: DType> {
    slice: CudaSlice<T>,
    device: CudaDevice,
}

impl<T: DType> BackendStorage<T> for CudaStorage<T> {
    fn to_cpu_storage(&self) -> Result<Cow<CpuStorage<T>>> {
        let data = self.device.stream().memcpy_dtov(&self.slice).w()?;
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

/// Can assume that the type T is available.
fn handle_node<T: DType>(
    current_name: &mut usize,
    header: &mut String,
    op: &GraphNode<T>,
    graph: &[GraphNode<T>],
) -> String {
    match &op.op {
        Op::BinaryOp {
            l_id,
            r_id,
            operator,
        } => {
            let l_name = handle_node(current_name, header, &graph[l_id.get()], graph);
            let r_name = handle_node(current_name, header, &graph[r_id.get()], graph);
            format!("({l_name} {} {r_name})", operator.as_c_op())
        }
        Op::UnaryOp { v_id, operator } => {
            let v_name = handle_node(current_name, header, &graph[v_id.get()], graph);
            operator.fill_in_c_op(v_name)
        }
        Op::Fill { v } => {
            *current_name += 1;
            let name = Name(*current_name);
            *header += &format!("T {} = {v:?};\n", name.to_name());
            format!("({})", name.to_name())
        }
        Op::Arange { start, step, stop } => {
            compile_error!("arange is not implemented for CUDA yet.");
            *current_name += 1;
            let name = Name(*current_name);
            *header += &format!(
                "T {} = static_cast<T>(i) * static_cast<T>({step:?}) + static_cast<T>({start:?});\n",
                name.to_name()
            );
            format!("({})", name.to_name())
        }
        Op::FusedMulAdd { a_id, b_id, c_id } => {
            let a_name = handle_node(current_name, header, &graph[a_id.get()], graph);
            let b_name = handle_node(current_name, header, &graph[b_id.get()], graph);
            let c_name = handle_node(current_name, header, &graph[c_id.get()], graph);
            #[cfg(feature = "slow_integral_fma_cuda")]
            if T::INTEGRAL {
                use crate::graph::BinaryOpType;
                let mul_op = BinaryOpType::Mul.to_c_op();
                let add_op = BinaryOpType::Add.to_c_op();
                format!("({a_name} {mul_op} {b_name} {add_op} {c_name})")
            } else {
                format!("( static_cast<T>(fma(static_cast<double>({a_name}), static_cast<double>({b_name}), static_cast<double>({c_name}))))")
            }
            #[cfg(not(feature = "slow_integral_fma_cuda"))]
            format!("( static_cast<T>(fma(static_cast<double>({a_name}), static_cast<double>({b_name}), static_cast<double>({c_name}))))")
        }
        Op::NoOp => unreachable!("no-op ops should never be reached."),
        Op::MatMul {
            l_id,
            r_id,
            o_id,
            k,
            alpha,
            beta,
        } => {
            todo!()
        }
    }
}

fn cuda_include_dir() -> Option<PathBuf> {
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

    #[allow(unused)]
    let roots = roots.into_iter().map(Into::<PathBuf>::into);

    env_vars
        .chain(roots)
        .find(|path| path.join("include").join("cuda.h").is_file())
}

fn compile_ptx(template_kernel: String) -> Result<Ptx> {
    cudarc::nvrtc::compile_ptx_with_opts(
        template_kernel,
        CompileOptions {
            use_fast_math: Some(true),
            include_paths: vec![cuda_include_dir()
                .unwrap()
                .join("include")
                .display()
                .to_string()],
            ..Default::default()
        },
    )
    .w()
}

impl CudaDevice {
    fn run_graph<T: DType>(&self, header: String, body: String) -> Result<CudaStorage<T>> {
        // Module name is based on hash of body and header
        let mut hasher = DefaultHasher::new();
        body.hash(&mut hasher);
        header.hash(&mut hasher);
        let function_name = format!("jit_kernel_{}_{}", hasher.finish(), T::NAME);

        let template_kernel = format!(
            r#"
            typedef unsigned char uint8_t;
            typedef unsigned int uint32_t;
            typedef long long int int64_t;
            {}

            template <typename T>
            __device__ void {function_name}_kernel(T *buf, const size_t numel) {{
                for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;
                    i += blockDim.x * gridDim.x) {{
                    {header}
                    buf[i] = {body};
                }}
            }}
            
            extern "C" __global__ void {function_name}({} *buf, const size_t numel) {{
                {function_name}_kernel(buf, numel);
            }}

            "#,
            T::C_DEP.unwrap_or(""),
            T::C_NAME,
        );

        let ptx = if let Some(home) = dirs::home_dir() {
            let path = format!(
                "{}/.cache/constensor/ptx/{function_name}.ptx",
                home.display()
            );
            if Path::new(&path).exists() {
                match fs::read_to_string(path) {
                    Ok(ptx) => Ptx::from_src(ptx),
                    Err(_) => compile_ptx(template_kernel)?,
                }
            } else {
                compile_ptx(template_kernel)?
            }
        } else {
            compile_ptx(template_kernel)?
        };

        let ptx_str = ptx.to_src();
        if let Some(home) = dirs::home_dir() {
            let path = format!(
                "{}/.cache/constensor/ptx/{function_name}.ptx",
                home.display()
            );
            let path = Path::new(&path);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(path, ptx_str)?;
        }

        let n_elems = S::element_count();
        let stream = self.stream();

        let data = unsafe { stream.alloc::<T>(n_elems) }.w()?;

        let func = self.get_or_load_func(&function_name, ptx)?;

        let cfg = LaunchConfig::for_num_elems(n_elems as u32);

        let mut builder = stream.launch_builder(&func);
        builder.arg(&data);
        builder.arg(&n_elems);
        unsafe { builder.launch(cfg).w()? };

        Ok(CudaStorage {
            slice: data,
            device: self.clone(),
        })
    }
}

impl BackendDevice for CudaDevice {
    type Storage<X: DType> = CudaStorage<X>;

    fn compile_and_run_graph<T: DType>(&self, nodes: &[GraphNode<T>]) -> Result<Self::Storage<T>> {
        let mut header = "".to_string();
        let body = handle_node(&mut 0, &mut header, nodes.last().unwrap(), nodes);
        self.run_graph::<T>(header, body)
    }
}

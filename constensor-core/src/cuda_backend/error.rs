use cudarc::nvrtc::CompileError;

/// cudarc related errors
#[derive(thiserror::Error, Debug)]
pub enum CudaError {
    #[error(transparent)]
    Cuda(#[from] cudarc::driver::DriverError),

    #[error("{cuda} when loading {module_name}")]
    Load {
        cuda: cudarc::driver::DriverError,
        module_name: String,
    },

    #[error("missing kernel '{module_name}'")]
    MissingKernel { module_name: String },

    #[error("error when compiling to ptx: {err}")]
    PtxCompileError { err: CompileError },
}

impl From<CudaError> for crate::Error {
    fn from(val: CudaError) -> Self {
        crate::Error::Cuda(Box::new(val)).bt()
    }
}

pub trait WrapErr<O> {
    fn w(self) -> std::result::Result<O, crate::Error>;
}

impl<O, E: Into<CudaError>> WrapErr<O> for std::result::Result<O, E> {
    fn w(self) -> std::result::Result<O, crate::Error> {
        self.map_err(|e| crate::Error::Cuda(Box::new(e.into())).bt())
    }
}

impl<O> WrapErr<O> for std::result::Result<O, CompileError> {
    fn w(self) -> std::result::Result<O, crate::Error> {
        self.map_err(|e| {
            crate::Error::Cuda(Box::new(CudaError::PtxCompileError { err: e }).into()).bt()
        })
    }
}

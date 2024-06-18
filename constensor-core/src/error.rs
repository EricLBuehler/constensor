#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Cuda(Box<dyn std::error::Error + Send + Sync>),

    #[error("Message: {0}")]
    Msg(String),

    #[error("{inner}\n{backtrace}")]
    WithBacktrace {
        inner: Box<Self>,
        backtrace: Box<std::backtrace::Backtrace>,
    },

    #[error("IO error: {0}")]
    IoError(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn bt(self) -> Self {
        let backtrace = std::backtrace::Backtrace::capture();
        match backtrace.status() {
            std::backtrace::BacktraceStatus::Disabled
            | std::backtrace::BacktraceStatus::Unsupported => self,
            _ => Self::WithBacktrace {
                inner: Box::new(self),
                backtrace: Box::new(backtrace),
            },
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Error::IoError(value.to_string())
    }
}

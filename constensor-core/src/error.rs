use std::{convert::Infallible, fmt::Display};

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

    /// Arbitrary errors wrapping.
    #[error(transparent)]
    Wrapped(Box<dyn std::error::Error + Send + Sync>),

    /// Arbitrary errors wrapping with context.
    #[error("{wrapped:?}\n{context:?}")]
    WrappedContext {
        wrapped: Box<dyn std::error::Error + Send + Sync>,
        context: String,
    },

    #[error("matmul is only supported for contiguous tensors lstride: {lhs_stride:?} rstride: {rhs_stride:?} ostride: {out_stride:?} mnk: {mnk:?}")]
    MatMulNonContiguous {
        lhs_stride: Vec<usize>,
        rhs_stride: Vec<usize>,
        out_stride: Vec<usize>,
        mnk: (usize, usize, usize),
    },
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Create a new error based on a printable error message.
    ///
    /// If the message implements `std::error::Error`, prefer using [`Error::wrap`] instead.
    pub fn msg<M: Display>(msg: M) -> Self {
        Self::Msg(msg.to_string()).bt()
    }

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

#[macro_export]
macro_rules! bail {
    ($msg:literal $(,)?) => {
        return Err($crate::Error::Msg(format!($msg).into()).bt())
    };
    ($err:expr $(,)?) => {
        return Err($crate::Error::Msg(format!($err).into()).bt())
    };
    ($fmt:expr, $($arg:tt)*) => {
        return Err($crate::Error::Msg(format!($fmt, $($arg)*).into()).bt())
    };
}

pub(crate) mod private {
    pub trait Sealed {}

    impl<T, E> Sealed for std::result::Result<T, E> where E: std::error::Error {}
    impl<T> Sealed for Option<T> {}
}

/// Attach more context to an error.
///
/// Inspired by [`anyhow::Context`].
pub trait Context<T, E>: private::Sealed {
    /// Wrap the error value with additional context.
    fn context<C>(self, context: C) -> std::result::Result<T, Error>
    where
        C: Display + Send + Sync + 'static;

    /// Wrap the error value with additional context that is evaluated lazily
    /// only once an error does occur.
    fn with_context<C, F>(self, f: F) -> std::result::Result<T, Error>
    where
        C: Display + Send + Sync + 'static,
        F: FnOnce() -> C;
}

impl<T, E> Context<T, E> for std::result::Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn context<C>(self, context: C) -> std::result::Result<T, Error>
    where
        C: Display + Send + Sync + 'static,
    {
        // Not using map_err to save 2 useless frames off the captured backtrace
        // in ext_context.
        match self {
            Ok(ok) => Ok(ok),
            Err(error) => Err(Error::WrappedContext {
                wrapped: Box::new(error),
                context: context.to_string(),
            }),
        }
    }

    fn with_context<C, F>(self, context: F) -> std::result::Result<T, Error>
    where
        C: Display + Send + Sync + 'static,
        F: FnOnce() -> C,
    {
        match self {
            Ok(ok) => Ok(ok),
            Err(error) => Err(Error::WrappedContext {
                wrapped: Box::new(error),
                context: context().to_string(),
            }),
        }
    }
}

impl<T> Context<T, Infallible> for Option<T> {
    fn context<C>(self, context: C) -> std::result::Result<T, Error>
    where
        C: Display + Send + Sync + 'static,
    {
        // Not using ok_or_else to save 2 useless frames off the captured
        // backtrace.
        match self {
            Some(ok) => Ok(ok),
            None => Err(Error::msg(context)),
        }
    }

    fn with_context<C, F>(self, context: F) -> std::result::Result<T, Error>
    where
        C: Display + Send + Sync + 'static,
        F: FnOnce() -> C,
    {
        match self {
            Some(ok) => Ok(ok),
            None => Err(Error::msg(context())),
        }
    }
}

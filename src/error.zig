//! Shared error vocabulary for the library.
//!
//! Only errors that are generic across modules live here. Precise,
//! module-specific failures (e.g. Singular, NotPositiveDefinite) are
//! declared in their own module's error set and merged with '|| Common'.
//!
//! Type errors are handled at comptime via @compileError and therefore
//! have no runtime error here.
pub const Common = error{
    IndexOutOfBounds,
    BadShape,
    SizeMismatch,
    Empty,
    TypeMismatch,
};

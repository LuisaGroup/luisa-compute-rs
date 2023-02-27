use std::path::Path;

use crate::prelude::*;
use exr::prelude::*;
use luisa_compute_api_types::PixelFormat;

use crate::resource::{Tex2d, IoTexel};

pub fn imread<T: IoTexel>(
    path: impl AsRef<Path>,
    format: PixelFormat,
) -> std::io::Result<Tex2d<T>> {
    todo!();
}

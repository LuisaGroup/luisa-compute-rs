use std::sync::Arc;

use crate::prelude::Device;
use luisa_compute_api_types as api;
pub(crate) struct AccelHandle {
    pub(crate) device: Device,
    pub(crate) handle: api::Accel,
}
impl Drop for AccelHandle {
    fn drop(&mut self) {
        self.device.inner.destory_accel(self.handle);
    }
}
pub struct Accel {
    pub(crate) handle: Arc<AccelHandle>,
}
pub(crate) struct MeshHandle {
    pub(crate) device: Device,
    pub(crate) handle: api::Mesh,
}

pub struct Mesh {
    pub(crate) handle: Arc<MeshHandle>,
}

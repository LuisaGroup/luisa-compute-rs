use super::*;
use std::hash::Hash;

pub(crate) trait Alignment: Default + Copy + Hash + Eq + 'static {
    const ALIGNMENT: usize;
}

macro_rules! alignment {
    ($T:ident, $align:literal) => {
        #[derive(Copy, Clone, Debug, Hash, Default, PartialEq, Eq)]
        #[repr(align($align))]
        pub struct $T;
        impl Alignment for $T {
            const ALIGNMENT: usize = $align;
        }
    };
}

alignment!(Align1, 1);
alignment!(Align2, 2);
alignment!(Align4, 4);
alignment!(Align8, 8);
alignment!(Align16, 16);
alignment!(Align32, 32);
alignment!(Align64, 64);
alignment!(Align128, 128);
alignment!(Align256, 256);

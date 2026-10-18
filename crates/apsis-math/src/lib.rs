//! Coordinate convention: target body to camera, +Z forward, +X right, +Y down.
#![no_std]
pub use nalgebra::{Quaternion, SMatrix, SVector, UnitQuaternion, Vector2, Vector3};
pub type V3 = Vector3<f64>;
pub type Q = UnitQuaternion<f64>;

#[derive(Clone, Debug)]
pub struct Pose {
    pub position: V3,
    pub attitude: Q,
}

impl Pose {
    pub fn transform(&self, point: V3) -> V3 {
        self.attitude * point + self.position
    }
    /// Left multiplicative attitude correction in camera coordinates.
    pub fn corrected(&self, delta: &SVector<f64, 6>) -> Self {
        Self {
            position: self.position + delta.fixed_rows::<3>(0),
            attitude: Q::from_scaled_axis(delta.fixed_rows::<3>(3).into_owned()) * self.attitude,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Camera {
    pub width: usize,
    pub height: usize,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}
impl Camera {
    pub fn new(width: usize, height: usize, focal: f64) -> Self {
        Self {
            width,
            height,
            fx: focal,
            fy: focal,
            cx: width as f64 / 2.0,
            cy: height as f64 / 2.0,
        }
    }
    pub fn project(&self, point: V3) -> Option<Vector2<f64>> {
        if self.width == 0
            || self.height == 0
            || ![self.fx, self.fy, self.cx, self.cy]
                .iter()
                .all(|v| v.is_finite())
            || self.fx <= 0.0
            || self.fy <= 0.0
            || !point.iter().all(|x| x.is_finite())
            || point.z <= 0.01
        {
            return None;
        }
        let pixel = Vector2::new(
            self.fx * point.x / point.z + self.cx,
            self.fy * point.y / point.z + self.cy,
        );
        pixel.iter().all(|v| v.is_finite()).then_some(pixel)
    }
}

pub fn target() -> [V3; 8] {
    core::array::from_fn(|i| {
        V3::new(
            if i & 1 == 0 { -0.05 } else { 0.05 },
            if i & 2 == 0 { -0.15 } else { 0.15 },
            if i & 4 == 0 { -0.05 } else { 0.05 },
        )
    })
}

/// Body corners plus an asymmetric docking plate on the -Z face, in meters.
pub fn landmarks() -> [V3; 12] {
    let body = target();
    core::array::from_fn(|i| {
        if i < 8 {
            body[i]
        } else {
            let j = i - 8;
            V3::new(
                if j & 1 == 0 { -0.032 } else { 0.038 },
                if j & 2 == 0 { -0.09 } else { 0.025 },
                -0.0505,
            )
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn projection_rejects_behind_camera() {
        let c = Camera::new(640, 480, 500.0);
        assert!(c.project(V3::new(0.0, 0.0, -1.0)).is_none());
        assert_eq!(
            c.project(V3::new(0.0, 0.0, 5.0)).unwrap(),
            Vector2::new(320.0, 240.0)
        );
    }
    #[test]
    fn quaternion_integration_preserves_norm() {
        let mut q = Q::identity();
        for _ in 0..10000 {
            q = Q::from_scaled_axis(V3::new(0.001, 0.002, 0.003)) * q;
        }
        assert!((q.norm() - 1.0).abs() < 1e-10);
    }
}

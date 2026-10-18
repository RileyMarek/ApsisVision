//! Coordinate convention: target body to camera, +Z forward, +X right, +Y down.
pub use nalgebra::{SMatrix, SVector, UnitQuaternion, Vector2, Vector3};
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
    pub focal: f64,
}
impl Camera {
    pub fn project(&self, point: V3) -> Option<Vector2<f64>> {
        if !point.iter().all(|x| x.is_finite()) || point.z <= 0.01 {
            return None;
        }
        Some(Vector2::new(
            self.focal * point.x / point.z + self.width as f64 / 2.0,
            self.focal * point.y / point.z + self.height as f64 / 2.0,
        ))
    }
}

pub fn target() -> [V3; 8] {
    std::array::from_fn(|i| {
        V3::new(
            if i & 1 == 0 { -0.8 } else { 0.8 },
            if i & 2 == 0 { -0.6 } else { 0.6 },
            if i & 4 == 0 { -0.5 } else { 0.5 },
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn projection_rejects_behind_camera() {
        let c = Camera {
            width: 640,
            height: 480,
            focal: 500.0,
        };
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

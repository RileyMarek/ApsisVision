//! Multiplicative error-state EKF: [position, velocity, attitude error, angular rate].
#![no_std]
use apsis_math::{Pose, SMatrix, SVector, Q, V3};
type M12 = SMatrix<f64, 12, 12>;

pub struct Filter {
    pub pose: Pose,
    pub velocity: V3,
    pub rate: V3,
    pub covariance: M12,
}
impl Filter {
    pub fn new(pose: Pose) -> Self {
        Self {
            pose,
            velocity: V3::zeros(),
            rate: V3::zeros(),
            covariance: M12::identity() * 0.1,
        }
    }
    pub fn predict(&mut self, dt: f64) {
        assert!(dt.is_finite() && dt > 0.0);
        self.pose.position += self.velocity * dt;
        self.pose.attitude = Q::from_scaled_axis(self.rate * dt) * self.pose.attitude;
        let mut f = M12::identity();
        let rotation = Q::from_scaled_axis(self.rate * dt).to_rotation_matrix();
        f.fixed_view_mut::<3, 3>(6, 6).copy_from(rotation.matrix());
        // Rate-to-attitude Jacobian in camera coordinates, integrated by central differences.
        for k in 0..3 {
            f[(k, k + 3)] = dt;
            let mut perturbation = V3::zeros();
            perturbation[k] = 1e-6;
            let plus = Q::from_scaled_axis((self.rate + perturbation) * dt);
            let minus = Q::from_scaled_axis((self.rate - perturbation) * dt);
            let column = (plus * minus.inverse()).scaled_axis() / 2e-6;
            f.fixed_view_mut::<3, 1>(6, 9 + k).copy_from(&column);
        }
        // White acceleration and angular acceleration, piecewise constant per interval.
        let mut g = SMatrix::<f64, 12, 6>::zeros();
        for k in 0..3 {
            g[(k, k)] = 0.5 * dt * dt;
            g[(k + 3, k)] = dt;
            g[(k + 6, k + 3)] = 0.5 * dt * dt;
            g[(k + 9, k + 3)] = dt;
        }
        self.covariance = f * self.covariance * f.transpose() + g * g.transpose() * 0.02;
    }
    /// Returns false for a rejected pose innovation (chi-square gate, 6 DoF).
    pub fn update(&mut self, measured: &Pose) -> bool {
        let mut innovation = SVector::<f64, 6>::zeros();
        innovation
            .fixed_rows_mut::<3>(0)
            .copy_from(&(measured.position - self.pose.position));
        innovation
            .fixed_rows_mut::<3>(3)
            .copy_from(&(measured.attitude * self.pose.attitude.inverse()).scaled_axis());
        if !innovation.iter().all(|v| v.is_finite()) {
            return false;
        }
        let mut h = SMatrix::<f64, 6, 12>::zeros();
        for k in 0..3 {
            h[(k, k)] = 1.0;
            h[(k + 3, k + 6)] = 1.0;
        }
        let r = SMatrix::<f64, 6, 6>::from_diagonal(&SVector::<f64, 6>::new(
            0.0036, 0.0036, 0.0036, 0.0004, 0.0004, 0.0004,
        ));
        let s = h * self.covariance * h.transpose() + r;
        let Some(chol) = s.cholesky() else {
            return false;
        };
        if innovation.dot(&chol.solve(&innovation)) > 22.46 {
            return false;
        }
        let k = chol.solve(&(h * self.covariance)).transpose();
        let delta = k * innovation;
        let mut correction = SVector::<f64, 6>::zeros();
        correction
            .fixed_rows_mut::<3>(0)
            .copy_from(&delta.fixed_rows::<3>(0));
        correction
            .fixed_rows_mut::<3>(3)
            .copy_from(&delta.fixed_rows::<3>(6));
        self.pose = self.pose.corrected(&correction);
        self.velocity += delta.fixed_rows::<3>(3);
        self.rate += delta.fixed_rows::<3>(9);
        let a = M12::identity() - k * h;
        self.covariance = a * self.covariance * a.transpose() + k * r * k.transpose();
        // Rebase the attitude error after injecting the multiplicative correction.
        let d = correction.fixed_rows::<3>(3).into_owned();
        let skew = SMatrix::<f64, 3, 3>::new(0.0, -d.z, d.y, d.z, 0.0, -d.x, -d.y, d.x, 0.0);
        let mut reset = M12::identity();
        reset
            .fixed_view_mut::<3, 3>(6, 6)
            .copy_from(&(SMatrix::<f64, 3, 3>::identity() + skew * 0.5));
        self.covariance = reset * self.covariance * reset.transpose();
        self.covariance = (self.covariance + self.covariance.transpose()) * 0.5;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dropout_grows_uncertainty_and_outlier_is_rejected() {
        let pose = Pose {
            position: V3::new(0.0, 0.0, 12.0),
            attitude: Q::identity(),
        };
        let mut f = Filter::new(pose.clone());
        for _ in 0..100 {
            f.predict(0.02);
            assert!(f.update(&pose));
        }
        let before = f.covariance.trace();
        for _ in 0..30 {
            f.predict(0.02);
        }
        assert!(f.covariance.trace() > before);
        let mut bad = pose.clone();
        bad.position.x = 100.0;
        assert!(!f.update(&bad));
        assert!(f.update(&pose));
        assert!(f.covariance.cholesky().is_some());
    }
}

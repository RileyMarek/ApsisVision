//! Multiplicative error-state EKF: [position, velocity, attitude error, angular rate].
#![no_std]
use apsis_math::{Pose, SMatrix, SVector, Q, V3};
pub type PoseCovariance = SMatrix<f64, 6, 6>;
pub fn default_measurement_covariance() -> PoseCovariance {
    PoseCovariance::from_diagonal(&SVector::<f64, 6>::new(
        0.000025, 0.000025, 0.000025, 0.0004, 0.0004, 0.0004,
    ))
}
fn valid_covariance(r: &PoseCovariance) -> bool {
    r.iter().all(|v| v.is_finite()) && (r - r.transpose()).amax() < 1e-10 && r.cholesky().is_some()
}
type M12 = SMatrix<f64, 12, 12>;

#[derive(Clone)]
pub struct Filter {
    pub pose: Pose,
    pub velocity: V3,
    pub rate: V3,
    pub covariance: M12,
    /// Most recent attempted innovation statistic; absent for malformed input.
    pub last_nis: Option<f64>,
    linear_psd: f64,
    angular_psd: f64,
}
impl Filter {
    pub fn new(pose: Pose) -> Self {
        Self {
            pose,
            velocity: V3::zeros(),
            rate: V3::zeros(),
            covariance: M12::identity() * 0.1,
            last_nis: None,
            linear_psd: 0.0004,
            angular_psd: 0.0004,
        }
    }
    /// Initialize observed pose blocks and their position/attitude correlation from R.
    /// Velocity and rate retain the independent 0.1 prior variances.
    pub fn new_with_covariance(pose: Pose, r: &PoseCovariance) -> Option<Self> {
        if !valid_covariance(r) {
            return None;
        }
        let mut f = Self::new(pose);
        for i in 0..6 {
            for j in 0..6 {
                let a = if i < 3 { i } else { i + 3 };
                let b = if j < 3 { j } else { j + 3 };
                f.covariance[(a, b)] = r[(i, j)];
            }
        }
        f.finite().then_some(f)
    }
    /// Continuous white acceleration PSDs: m^2/s^3 and rad^2/s^3. No automatic truth tuning.
    pub fn set_process_noise(&mut self, linear: f64, angular: f64) -> bool {
        if !linear.is_finite() || !angular.is_finite() || linear < 0.0 || angular < 0.0 {
            return false;
        }
        self.linear_psd = linear;
        self.angular_psd = angular;
        true
    }
    pub fn predict(&mut self, dt: f64) -> bool {
        if !dt.is_finite() || dt <= 0.0 || dt > 10.0 {
            return false;
        }
        let before = self.clone();
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
        // Continuous white acceleration spectral density; covariance scales with elapsed time,
        // rather than changing its physical meaning when prediction cadence changes.
        let mut process = M12::zeros();
        for k in 0..3 {
            for base in [0, 6] {
                let p = k + base;
                let v = p + 3;
                let psd = if base == 0 {
                    self.linear_psd
                } else {
                    self.angular_psd
                };
                process[(p, p)] = psd * dt * dt * dt / 3.0;
                process[(p, v)] = psd * dt * dt / 2.0;
                process[(v, p)] = process[(p, v)];
                process[(v, v)] = psd * dt;
            }
        }
        self.covariance = f * self.covariance * f.transpose() + process;
        if !self.finite() {
            *self = before;
            return false;
        }
        true
    }
    fn finite(&self) -> bool {
        self.pose
            .position
            .iter()
            .chain(self.pose.attitude.coords.iter())
            .chain(self.velocity.iter())
            .chain(self.rate.iter())
            .chain(self.covariance.iter())
            .all(|x| x.is_finite())
    }
    /// Returns false for a rejected pose innovation (chi-square gate, 6 DoF).
    pub fn update(&mut self, measured: &Pose) -> bool {
        self.update_with_noise_scale(measured, 1.0)
    }
    /// Conservative reacquisition weighting; a scale >=1 inflates measurement covariance.
    pub fn update_with_noise_scale(&mut self, measured: &Pose, noise_scale: f64) -> bool {
        self.update_with_covariance(measured, &default_measurement_covariance(), noise_scale)
    }
    pub fn update_with_covariance(
        &mut self,
        measured: &Pose,
        covariance: &PoseCovariance,
        noise_scale: f64,
    ) -> bool {
        self.last_nis = None;
        if !valid_covariance(covariance) {
            return false;
        }
        if !noise_scale.is_finite() || noise_scale < 1.0 {
            return false;
        }
        if !self.finite() {
            return false;
        }
        let before = self.clone();
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
        let r = covariance * noise_scale;
        let s = h * self.covariance * h.transpose() + r;
        let Some(chol) = s.cholesky() else {
            return false;
        };
        let distance = innovation.dot(&chol.solve(&innovation));
        self.last_nis = Some(distance);
        if !distance.is_finite() || distance > 22.46 {
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
        if !self.finite() || self.covariance.cholesky().is_none() {
            *self = before;
            return false;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_covariance_controls_gain_and_nis_and_invalid_noise_is_atomic() {
        let pose = Pose {
            position: V3::new(0.0, 0.0, 2.0),
            attitude: Q::identity(),
        };
        let r = default_measurement_covariance();
        let mut narrow = Filter::new_with_covariance(pose.clone(), &r).unwrap();
        let mut broad = narrow.clone();
        let mut measurement = pose.clone();
        measurement.position.x += 0.01;
        assert!(narrow.update_with_covariance(&measurement, &r, 1.0));
        assert!(broad.update_with_covariance(&measurement, &(r * 100.0), 1.0));
        assert!(narrow.pose.position.x > broad.pose.position.x * 10.0);
        assert!(narrow.last_nis.unwrap() > broad.last_nis.unwrap() * 10.0);
        let before = narrow.clone();
        let mut invalid = r;
        invalid[(0, 0)] = -1.0;
        assert!(!narrow.update_with_covariance(&measurement, &invalid, 1.0));
        assert_eq!(narrow.pose.position, before.pose.position);
        assert_eq!(narrow.covariance, before.covariance);
        invalid = r;
        invalid[(0, 1)] = 0.01;
        assert!(Filter::new_with_covariance(pose.clone(), &invalid).is_none());
        assert!(!narrow.set_process_noise(f64::NAN, 0.1));
        let mut correlated = r;
        correlated[(0, 3)] = 0.00001;
        correlated[(3, 0)] = 0.00001;
        let initialized = Filter::new_with_covariance(pose, &correlated).unwrap();
        assert_eq!(initialized.covariance[(0, 6)], correlated[(0, 3)]);
    }
    #[test]
    fn covariance_matches_elapsed_time_and_invalid_prediction_is_atomic() {
        let mut once = Filter::new(Pose {
            position: V3::new(0.0, 0.0, 1.6),
            attitude: Q::identity(),
        });
        let mut many = once.clone();
        assert!(once.predict(0.5));
        for _ in 0..25 {
            assert!(many.predict(0.02));
        }
        assert!((once.covariance - many.covariance).norm() < 1e-8);
        let before = once.clone();
        assert!(!once.predict(f64::NAN));
        assert!(!once.predict(-0.1));
        assert_eq!(once.covariance, before.covariance);
        assert_eq!(once.pose.position, before.pose.position);
    }
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

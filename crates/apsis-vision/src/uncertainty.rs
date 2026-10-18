//! Local pose uncertainty conditional on CORRECT, independent pixel correspondences.
//! Does not represent wrong identities, calibration bias or multiple pose modes.
use apsis_math::{Camera, Pose, SMatrix, Vector2, V3};
pub type PoseCovariance = SMatrix<f64, 6, 6>;
pub struct PoseUncertainty {
    pub covariance: PoseCovariance,
    /// Dimensionless information condition number after scaling translation by model radius.
    pub condition: f64,
    pub pixel_sigma: f64,
    pub inliers: usize,
}
fn jacobian(camera: Camera, rotated: V3, translated: V3) -> Option<SMatrix<f64, 2, 6>> {
    camera.project(translated)?;
    let z = translated.z;
    let projection = SMatrix::<f64, 2, 3>::new(
        camera.fx / z,
        0.0,
        -camera.fx * translated.x / (z * z),
        0.0,
        camera.fy / z,
        -camera.fy * translated.y / (z * z),
    );
    let mut transform = SMatrix::<f64, 3, 6>::zeros();
    transform
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&SMatrix::<f64, 3, 3>::identity());
    // Left attitude error rotates the body about its origin, not camera translation.
    transform
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&SMatrix::<f64, 3, 3>::new(
            0.0, rotated.z, -rotated.y, -rotated.z, 0.0, rotated.x, rotated.y, -rotated.x, 0.0,
        ));
    Some(projection * transform)
}
pub fn estimate(
    camera: Camera,
    model: &[V3],
    points: &[(usize, Vector2<f64>)],
    pose: &Pose,
    pixel_sigma_floor: f64,
) -> Option<PoseUncertainty> {
    if !(6..=32).contains(&points.len())
        || model.len() > 32
        || model.is_empty()
        || !pixel_sigma_floor.is_finite()
        || pixel_sigma_floor <= 0.0
    {
        return None;
    }
    let mut seen = [false; 32];
    let mut information = PoseCovariance::zeros();
    let mut squared = 0.0;
    let mut count = 0;
    for (id, observed) in points {
        if *id >= model.len() || seen[*id] || !observed.iter().all(|v| v.is_finite()) {
            return None;
        }
        seen[*id] = true;
        let rotated = pose.attitude * model[*id];
        let translated = rotated + pose.position;
        let residual = (camera.project(translated)? - observed).norm_squared();
        if residual >= 6.25 {
            continue;
        }
        let j = jacobian(camera, rotated, translated)?;
        information += j.transpose() * j;
        squared += residual;
        count += 1;
    }
    if count < 6 || !information.iter().all(|x| x.is_finite()) {
        return None;
    }
    let center = model.iter().fold(V3::zeros(), |s, p| s + p) / model.len() as f64;
    let radius = libm::sqrt(
        model
            .iter()
            .map(|p| (p - center).norm_squared())
            .sum::<f64>()
            / model.len() as f64,
    );
    if !radius.is_finite() || radius < 1e-6 {
        return None;
    }
    let mut scaling = PoseCovariance::identity();
    for i in 0..3 {
        scaling[(i, i)] = radius;
    }
    let eigen = (scaling * information * scaling)
        .try_symmetric_eigen(f64::EPSILON, 96)?
        .eigenvalues;
    let lo = eigen.min();
    let hi = eigen.max();
    if lo <= 0.0 || !hi.is_finite() || hi / lo > 1e8 {
        return None;
    }
    // Two residual coordinates per correspondence, six fitted pose parameters.
    let variance = (squared / (2 * count - 6) as f64).max(pixel_sigma_floor * pixel_sigma_floor);
    let covariance = information.cholesky()?.inverse() * variance;
    Some(PoseUncertainty {
        covariance: (covariance + covariance.transpose()) * 0.5,
        condition: hi / lo,
        pixel_sigma: libm::sqrt(variance),
        inliers: count,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    use apsis_math::{SVector, Q};
    #[test]
    fn known_correspondence_pixel_monte_carlo_has_consistent_local_pose_errors() {
        let camera = Camera::new(640, 480, 1500.0);
        let model = apsis_math::landmarks();
        let truth = Pose {
            position: V3::new(0.01, -0.02, 1.6),
            attitude: Q::from_scaled_axis(V3::new(0.2, 0.3, 0.1)),
        };
        let mut random = apsis_sim::Noise::new(932);
        let mut total = 0.0;
        for _ in 0..256 {
            let points: [_; 12] = core::array::from_fn(|i| {
                let pixel = camera.project(truth.transform(model[i])).unwrap();
                // Independent unit-variance bounded pixel errors; identities are known.
                (
                    i,
                    pixel + Vector2::new(random.uniform(), random.uniform()) * libm::sqrt(12.0),
                )
            });
            let fitted = crate::solve_model(camera, &model, &points, &truth)
                .unwrap()
                .0;
            let r = estimate(camera, &model, &points, &fitted, 1.0)
                .unwrap()
                .covariance;
            let mut error = SVector::<f64, 6>::zeros();
            error
                .fixed_rows_mut::<3>(0)
                .copy_from(&(fitted.position - truth.position));
            error
                .fixed_rows_mut::<3>(3)
                .copy_from(&(fitted.attitude * truth.attitude.inverse()).scaled_axis());
            total += error.dot(&r.cholesky().unwrap().solve(&error));
        }
        let mean = total / 256.0;
        assert!(
            (4.5..7.5).contains(&mean),
            "conditional local pose NEES {mean}"
        );
    }
    #[test]
    fn analytic_projection_jacobian_matches_central_differences() {
        let c = Camera::new(640, 480, 800.0);
        let p = Pose {
            position: V3::new(0.1, -0.2, 2.0),
            attitude: Q::from_scaled_axis(V3::new(0.2, -0.3, 0.1)),
        };
        let point = V3::new(0.04, 0.12, -0.05);
        let j = jacobian(c, p.attitude * point, p.transform(point)).unwrap();
        for i in 0..6 {
            let mut d = SVector::<f64, 6>::zeros();
            d[i] = 1e-6;
            let difference = (c.project(p.corrected(&d).transform(point)).unwrap()
                - c.project(p.corrected(&(-d)).transform(point)).unwrap())
                / 2e-6;
            assert!((difference - j.column(i)).norm() < 1e-6);
        }
    }
    #[test]
    fn range_noise_and_support_change_uncertainty_and_degeneracy_rejects() {
        let c = Camera::new(640, 480, 1500.0);
        let model = apsis_math::landmarks();
        let at = |z: f64, sigma: f64, n: usize| {
            let p = Pose {
                position: V3::new(0., 0., z),
                attitude: Q::from_scaled_axis(V3::new(0.2, 0.3, 0.1)),
            };
            let points: [_; 12] =
                core::array::from_fn(|i| (i, c.project(p.transform(model[i])).unwrap()));
            estimate(c, &model, &points[..n], &p, sigma).unwrap()
        };
        let near = at(1.5, 1., 12);
        let far = at(3., 1., 12);
        let noisy = at(1.5, 2., 12);
        assert!(far.covariance[(2, 2)] > near.covariance[(2, 2)] * 10.0);
        assert!((noisy.covariance - near.covariance * 4.0).norm() < 1e-10);
        assert!(at(1.5, 1., 6).covariance.trace() > near.covariance.trace());
        assert!(near.covariance.cholesky().is_some());
        let p = Pose {
            position: V3::new(0., 0., 2.),
            attitude: Q::identity(),
        };
        let line: [_; 8] = core::array::from_fn(|i| V3::new(i as f64 * 0.01, 0., 0.));
        let points: [_; 8] =
            core::array::from_fn(|i| (i, c.project(p.transform(line[i])).unwrap()));
        assert!(estimate(c, &line, &points, &p, 1.).is_none());
        assert!(estimate(c, &model, &points, &p, f64::NAN).is_none());
    }
}

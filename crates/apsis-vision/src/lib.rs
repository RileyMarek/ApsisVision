//! Grayscale corners, calibrated PnP, and known-convex-body boundary verification.
//! The intensity-coded detector below is retained only for synthetic marker fixtures.
#![no_std]
pub mod corners;
pub mod integrity;
pub mod silhouette;
pub mod uncertainty;
use apsis_math::{target, Camera, Pose, SMatrix, SVector, Vector2, V3};

/// Bounded marker observations; no allocator or operating system is required.
pub struct Observations {
    points: [(usize, Vector2<f64>); 8],
    len: usize,
}
impl core::ops::Deref for Observations {
    type Target = [(usize, Vector2<f64>)];
    fn deref(&self) -> &Self::Target {
        &self.points[..self.len]
    }
}

pub fn detect(image: &[u8], camera: Camera) -> Observations {
    let mut result = Observations {
        points: [(0, Vector2::zeros()); 8],
        len: 0,
    };
    if camera.width.checked_mul(camera.height) != Some(image.len()) {
        return result;
    }
    let mut sums = [(0.0, 0.0, 0); 8];
    for (index, &value) in image.iter().enumerate() {
        if value >= 32 && (value - 32).is_multiple_of(24) {
            let id = ((value - 32) / 24) as usize;
            if id < 8 {
                sums[id].0 += (index % camera.width) as f64;
                sums[id].1 += (index / camera.width) as f64;
                sums[id].2 += 1;
            }
        }
    }
    for (id, sum) in sums.iter().enumerate() {
        if sum.2 >= 4 {
            result.points[result.len] =
                (id, Vector2::new(sum.0 / sum.2 as f64, sum.1 / sum.2 as f64));
            result.len += 1;
        }
    }
    result
}

pub fn solve(camera: Camera, points: &[(usize, Vector2<f64>)], seed: &Pose) -> Option<(Pose, f64)> {
    solve_model(camera, &target(), points, seed)
}

pub fn solve_model(
    camera: Camera,
    model: &[V3],
    points: &[(usize, Vector2<f64>)],
    seed: &Pose,
) -> Option<(Pose, f64)> {
    if points.len() < 6
        || model.len() > 32
        || points
            .iter()
            .any(|(id, p)| *id >= model.len() || !p.iter().all(|v| v.is_finite()))
    {
        return None;
    }
    let mut seen = [false; 32];
    for (id, _) in points {
        if seen[*id] {
            return None;
        }
        seen[*id] = true;
    }
    // Collinear landmarks cannot constrain a general pose. Coplanar landmarks are
    // allowed for local warm-start fitting, with the remaining ambiguity documented.
    let origin = model[points[0].0];
    let baseline = points
        .iter()
        .map(|(id, _)| model[*id] - origin)
        .max_by(|a, b| a.norm_squared().total_cmp(&b.norm_squared()))?;
    if baseline.norm_squared() < 1e-12
        || points
            .iter()
            .all(|(id, _)| baseline.cross(&(model[*id] - origin)).norm_squared() < 1e-16)
    {
        return None;
    }
    let mut pose = seed.clone();
    for _ in 0..30 {
        let mut normal = SMatrix::<f64, 6, 6>::zeros();
        let mut gradient = SVector::<f64, 6>::zeros();
        for (id, observed) in points {
            let predicted = camera.project(pose.transform(model[*id]))?;
            let mut jac = SMatrix::<f64, 2, 6>::zeros();
            for k in 0..6 {
                let mut d = SVector::<f64, 6>::zeros();
                d[k] = 1e-6;
                let shifted = camera.project(pose.corrected(&d).transform(model[*id]))?;
                jac.set_column(k, &((shifted - predicted) / 1e-6));
            }
            normal += jac.transpose() * jac;
            gradient += jac.transpose() * (observed - predicted);
        }
        let delta = (normal + SMatrix::identity() * 1e-6)
            .lu()
            .solve(&gradient)?;
        if !delta.iter().all(|x| x.is_finite()) || delta.norm() > 5.0 {
            return None;
        }
        pose = pose.corrected(&delta);
        if delta.norm() < 1e-7 {
            break;
        }
    }
    let mut squared = 0.0;
    for (id, observed) in points {
        squared += (camera.project(pose.transform(model[*id]))? - observed).norm_squared();
    }
    let rms = libm::sqrt(squared / points.len() as f64);
    (rms < 3.0).then_some((pose, rms))
}

/// Deterministic bounded RANSAC over tentative geometric correspondences.
/// A local pose prior is still required; this is not global target recognition.
pub fn solve_robust(
    camera: Camera,
    model: &[V3],
    points: &[(usize, Vector2<f64>)],
    seed: &Pose,
) -> Option<(Pose, f64)> {
    if !(6..=32).contains(&points.len())
        || model.len() > 32
        || points.iter().any(|(id, _)| *id >= model.len())
    {
        return None;
    }
    let mut best = None;
    let mut best_count = 0;
    let mut best_error = f64::INFINITY;
    let mut random = 17_u64;
    for iteration in 0..49 {
        let mut subset = [(0, Vector2::zeros()); 6];
        let mut pool: [usize; 32] = core::array::from_fn(|i| i);
        for (step, point) in subset.iter_mut().enumerate() {
            random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
            let remaining = points.len() - step;
            let id = ((random >> 32) as usize) % remaining;
            *point = points[pool[id]];
            pool.swap(id, remaining - 1);
        }
        let fitted = if iteration == 0 {
            solve_model(camera, model, points, seed)
        } else {
            solve_model(camera, model, &subset, seed)
        };
        let Some((pose, _)) = fitted else {
            continue;
        };
        let mut count = 0;
        let mut error = 0.0;
        for (id, pixel) in points {
            if let Some(p) = camera.project(pose.transform(model[*id])) {
                let e = (p - pixel).norm_squared();
                if e < 6.25 {
                    count += 1;
                    error += e;
                }
            }
        }
        if count > best_count || (count == best_count && error < best_error) {
            best_count = count;
            best_error = error;
            best = Some(pose);
        }
        if count == points.len() {
            break;
        }
    }
    if best_count < 6 {
        return None;
    }
    let pose = best?;
    let mut inliers = [(0, Vector2::zeros()); 32];
    let mut count = 0;
    for point in points {
        if camera
            .project(pose.transform(model[point.0]))
            .is_some_and(|p| (p - point.1).norm_squared() < 6.25)
        {
            inliers[count] = *point;
            count += 1;
        }
    }
    solve_model(camera, model, &inliers[..count], &pose)
}

/// Robust fit plus local pixel-conditioned uncertainty. The caller must still
/// verify landmark identity, calibration and silhouette/independent image support.
pub fn solve_uncertain(
    camera: Camera,
    model: &[V3],
    points: &[(usize, Vector2<f64>)],
    seed: &Pose,
    pixel_sigma_floor: f64,
) -> Option<(Pose, f64, uncertainty::PoseUncertainty)> {
    let (pose, rms) = solve_robust(camera, model, points, seed)?;
    let uncertainty = uncertainty::estimate(camera, model, points, &pose, pixel_sigma_floor)?;
    Some((pose, rms, uncertainty))
}

/// Explicit visual admission precedes covariance estimation and the downstream NIS gate.
pub fn solve_admitted(
    camera: Camera,
    model: &[V3],
    points: &[(usize, Vector2<f64>)],
    seed: &Pose,
    pixel_sigma_floor: f64,
) -> Result<(Pose, f64, uncertainty::PoseUncertainty), integrity::Failure> {
    let (pose, rms) = solve_robust(camera, model, points, seed).ok_or(integrity::Failure::Fit)?;
    integrity::reprojection(camera, model, points, &pose)?;
    let uncertainty = uncertainty::estimate(camera, model, points, &pose, pixel_sigma_floor)
        .ok_or(integrity::Failure::Covariance)?;
    Ok((pose, rms, uncertainty))
}

#[cfg(test)]
mod tests {
    use super::*;
    use apsis_math::{Q, V3};
    #[test]
    fn robust_fit_rejects_outlier_and_collinear_geometry() {
        let c = Camera::new(640, 480, 1500.0);
        let model = apsis_math::landmarks();
        let truth = Pose {
            position: V3::new(0.0, 0.0, 1.6),
            attitude: Q::from_scaled_axis(V3::new(0.25, 0.35, 0.1)),
        };
        let mut points: [_; 12] =
            core::array::from_fn(|i| (i, c.project(truth.transform(model[i])).unwrap()));
        points[3].1.x += 60.0;
        let seed = Pose {
            position: V3::new(0.0, 0.0, 1.62),
            attitude: Q::from_scaled_axis(V3::new(0.2, 0.3, 0.1)),
        };
        let fitted = solve_robust(c, &model, &points, &seed).unwrap().0;
        assert!((fitted.position - truth.position).norm() < 1e-4);
        let line: [_; 12] = core::array::from_fn(|i| V3::new(i as f64 * 0.01, 0.0, 0.0));
        assert!(solve_model(c, &line, &points, &seed).is_none());
    }
    #[test]
    fn recovers_pose_from_pixels() {
        let c = Camera::new(640, 480, 1000.0);
        let truth = Pose {
            position: V3::new(0.02, -0.01, 1.6),
            attitude: Q::from_scaled_axis(V3::new(0.15, 0.2, -0.1)),
        };
        let image = apsis_sim::render(c, &truth, &mut apsis_sim::Noise::new(7), false);
        let points = detect(&image, c);
        let seed = Pose {
            position: V3::new(0.0, 0.0, 1.8),
            attitude: Q::identity(),
        };
        let (actual, rms) = solve(c, &points, &seed).unwrap();
        assert!((actual.position - truth.position).norm() < 0.12);
        assert!((actual.attitude * truth.attitude.inverse()).angle() < 0.04);
        assert!(rms < 1.0);
        assert!(solve(c, &points[..4], &seed).is_none());
        assert!(detect(&[], c).is_empty());
        assert!(solve(c, &[points[0]; 6], &seed).is_none());
    }
}

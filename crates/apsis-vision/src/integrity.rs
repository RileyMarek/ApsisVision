//! Admission evidence independent of pose covariance. Correct correspondence is
//! still an assumption: a small residual alone cannot resolve a symmetric mode.
use apsis_math::{landmarks, Camera, Pose, Vector2, Q, V3};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Failure {
    Fit,
    Reprojection,
    Covariance,
    Boundary,
    Ambiguous,
}
#[derive(Clone, Copy, Debug)]
pub struct Reprojection {
    pub mean_px: f64,
    pub rms_px: f64,
    pub worst_px: f64,
    pub points: usize,
}
/// Every supplied tentative correspondence participates, including RANSAC outliers.
/// No R, filter state, pose prior or scoring truth can relax the 3 px mean limit.
pub fn reprojection(
    camera: Camera,
    model: &[V3],
    points: &[(usize, Vector2<f64>)],
    pose: &Pose,
) -> Result<Reprojection, Failure> {
    if !(6..=32).contains(&points.len()) || model.len() > 32 {
        return Err(Failure::Reprojection);
    }
    let mut seen = [false; 32];
    let mut sum = 0.0;
    let mut squared = 0.0;
    let mut worst = 0.0_f64;
    for (id, observed) in points {
        if *id >= model.len() || seen[*id] || !observed.iter().all(|v| v.is_finite()) {
            return Err(Failure::Reprojection);
        }
        seen[*id] = true;
        let predicted = camera
            .project(pose.transform(model[*id]))
            .ok_or(Failure::Reprojection)?;
        let error = (predicted - observed).norm();
        if !error.is_finite() {
            return Err(Failure::Reprojection);
        }
        sum += error;
        squared += error * error;
        worst = worst.max(error);
    }
    let mean = sum / points.len() as f64;
    if mean > 3.0 {
        return Err(Failure::Reprojection);
    }
    Ok(Reprojection {
        mean_px: mean,
        rms_px: libm::sqrt(squared / points.len() as f64),
        worst_px: worst,
        points: points.len(),
    })
}

#[derive(Clone, Copy, Debug)]
pub struct ModeEvidence {
    pub primary_supported: usize,
    pub runner_up_supported: usize,
    /// Heuristic independent-sample Bernoulli score gap, NOT calibrated log odds.
    pub log_support_gap: f64,
}
fn sample(image: &[u8], camera: Camera, p: Vector2<f64>) -> Option<f64> {
    if !p.iter().all(|v| v.is_finite())
        || p.x < 1.0
        || p.y < 1.0
        || p.x >= (camera.width - 1) as f64
        || p.y >= (camera.height - 1) as f64
    {
        return None;
    }
    let (x, y) = (p.x as usize, p.y as usize);
    let mut sum = 0u32;
    for j in y - 1..=y + 1 {
        for i in x - 1..=x + 1 {
            sum += image[j * camera.width + i] as u32;
        }
    }
    Some(sum as f64 / 9.0)
}
/// Independent image support for the DEMO'S asymmetric bright-rim docking plate.
/// This is not a Tango CAD model or a general spacecraft photometric likelihood.
fn plate_support(image: &[u8], camera: Camera, pose: &Pose) -> usize {
    let model = landmarks();
    if (pose.attitude * V3::new(0.0, 0.0, -1.0)).dot(&pose.transform(model[8])) >= 0.0 {
        return 0;
    }
    let mut p = [Vector2::zeros(); 4];
    for (j, id) in [8, 9, 11, 10].iter().enumerate() {
        let Some(pixel) = camera.project(pose.transform(model[*id])) else {
            return 0;
        };
        p[j] = pixel;
    }
    let center = (p[0] + p[1] + p[2] + p[3]) / 4.0;
    let mut support = 0;
    for edge in 0..4 {
        let a = p[edge];
        let b = p[(edge + 1) % 4];
        let length = (b - a).norm();
        if length < 12.0 {
            return 0;
        }
        let mut inward = Vector2::new(-(b - a).y, (b - a).x) / length;
        if inward.dot(&(center - (a + b) / 2.0)) < 0.0 {
            inward = -inward;
        }
        for j in 0..9 {
            let point = a + (b - a) * ((j as f64 + 0.5) / 9.0);
            let supported = (-2..=2).any(|shift| {
                let rim = point + inward * shift as f64;
                let (Some(bright), Some(inner), Some(outer)) = (
                    sample(image, camera, rim),
                    sample(image, camera, rim + inward * 8.0),
                    sample(image, camera, rim - inward * 4.0),
                ) else {
                    return false;
                };
                bright > inner + 35.0 && bright > outer + 35.0
            });
            support += usize::from(supported);
        }
    }
    support
}
/// Compare the eight proper rotations preserving this square-section 3U bus.
/// Do not invert the camera pose: each alternative keeps translation and applies a
/// BODY-frame symmetry. Ties, missing plate evidence and a better rival abstain.
pub fn cubesat_modes(image: &[u8], camera: Camera, pose: &Pose) -> Result<ModeEvidence, Failure> {
    if camera.width < 8
        || camera.height < 8
        || camera.width.checked_mul(camera.height) != Some(image.len())
    {
        return Err(Failure::Ambiguous);
    }
    crate::silhouette::verify(image, camera, &apsis_math::target(), pose)
        .ok_or(Failure::Boundary)?;
    let primary = plate_support(image, camera, pose);
    let mut runner = 0;
    for flip in 0..2 {
        for turn in 0..4 {
            if flip == 0 && turn == 0 {
                continue;
            }
            let rotation =
                Q::from_scaled_axis(V3::new(
                    0.0,
                    turn as f64 * core::f64::consts::FRAC_PI_2,
                    0.0,
                )) * Q::from_scaled_axis(V3::new(flip as f64 * core::f64::consts::PI, 0.0, 0.0));
            let alternative = Pose {
                position: pose.position,
                attitude: pose.attitude * rotation,
            };
            runner = runner.max(plate_support(image, camera, &alternative));
        }
    }
    let gap = (primary as f64 - runner as f64) * libm::log(0.85 / 0.15);
    if primary < 24 || gap < core::f64::consts::LN_10 {
        return Err(Failure::Ambiguous);
    }
    Ok(ModeEvidence {
        primary_supported: primary,
        runner_up_supported: runner,
        log_support_gap: gap,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_correspondences_enter_the_hard_pixel_gate() {
        let camera = Camera::new(640, 480, 1500.0);
        let model = landmarks();
        let p = Pose {
            position: V3::new(0.0, 0.0, 1.6),
            attitude: Q::identity(),
        };
        let mut observations: [_; 12] =
            core::array::from_fn(|i| (i, camera.project(p.transform(model[i])).unwrap()));
        assert_eq!(
            reprojection(camera, &model, &observations, &p)
                .unwrap()
                .mean_px,
            0.0
        );
        observations[0].1.x += 60.0;
        assert_eq!(
            reprojection(camera, &model, &observations, &p).unwrap_err(),
            Failure::Reprojection
        );
        assert!(
            crate::solve_robust(camera, &model, &observations, &p).is_some(),
            "RANSAC can fit the subset; admission must still fail"
        );
    }
    #[test]
    fn plate_distinguishes_body_symmetry_and_erasure_abstains() {
        let camera = Camera::new(640, 480, 1500.0);
        let p = apsis_sim::truth(0.0);
        let image = apsis_sim::scene::render(
            camera,
            &p,
            0.0,
            apsis_sim::scene::Conditions::default(),
            &mut apsis_sim::Noise::new(42),
        );
        let evidence = cubesat_modes(&image.gray, camera, &p).unwrap();
        assert!(evidence.primary_supported > evidence.runner_up_supported);
        let wrong = Pose {
            position: p.position,
            attitude: p.attitude * Q::from_scaled_axis(V3::new(0.0, 0.0, core::f64::consts::PI)),
        };
        assert_eq!(
            cubesat_modes(&image.gray, camera, &wrong).unwrap_err(),
            Failure::Ambiguous
        );
        let no_plate = apsis_sim::scene::render(
            camera,
            &p,
            0.0,
            apsis_sim::scene::Conditions::default(),
            &mut apsis_sim::Noise::new(42),
        );
        // A plate-free side view supplies a body outline but no asymmetric evidence.
        let hidden = Pose {
            position: p.position,
            attitude: Q::from_scaled_axis(V3::new(0.0, core::f64::consts::FRAC_PI_2, 0.0)),
        };
        let side = apsis_sim::scene::render(
            camera,
            &hidden,
            0.0,
            apsis_sim::scene::Conditions::default(),
            &mut apsis_sim::Noise::new(42),
        );
        assert!(matches!(
            cubesat_modes(&side.gray, camera, &hidden),
            Err(Failure::Ambiguous) | Err(Failure::Boundary)
        ));
        let identity = Pose {
            position: V3::new(0.0, 0.0, 1.6),
            attitude: Q::identity(),
        };
        let mut symmetric = image.gray.clone();
        symmetric.fill(0);
        let low = camera
            .project(identity.transform(V3::new(-0.05, -0.15, -0.05)))
            .unwrap();
        let high = camera
            .project(identity.transform(V3::new(0.05, 0.15, -0.05)))
            .unwrap();
        let (x0, y0, x1, y1) = (
            low.x as usize,
            low.y as usize,
            high.x as usize,
            high.y as usize,
        );
        for y in y0..=y1 {
            for x in x0..=x1 {
                symmetric[y * camera.width + x] =
                    if x - x0 < 3 || x1 - x < 3 || y - y0 < 3 || y1 - y < 3 {
                        230
                    } else {
                        20
                    };
            }
        }
        assert!(
            crate::silhouette::verify(&symmetric, camera, &apsis_math::target(), &identity)
                .is_some()
        );
        assert_eq!(
            cubesat_modes(&symmetric, camera, &identity).unwrap_err(),
            Failure::Ambiguous,
            "A supported symmetric outline supplies no unique attitude evidence"
        );
        assert!(cubesat_modes(&no_plate.gray[..10], camera, &p).is_err());
    }
}

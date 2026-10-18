//! Independent pixel support for a known opaque convex body's projected boundary.
//! This assumption is explicit: arbitrary keypoints are not a silhouette model.
use apsis_math::{Camera, Pose, Vector2, V3};

#[derive(Clone, Copy, Debug)]
pub struct BoundarySupport {
    pub supported: usize,
    pub samples: usize,
}
impl BoundarySupport {
    /// Fraction of sampled boundary locations with positive image evidence.
    pub fn fraction(self) -> f64 {
        self.supported as f64 / self.samples.max(1) as f64
    }
}

fn cross(a: Vector2<f64>, b: Vector2<f64>, c: Vector2<f64>) -> f64 {
    let u = b - a;
    let v = c - a;
    u.x * v.y - u.y * v.x
}
fn intensity(image: &[u8], camera: Camera, p: Vector2<f64>) -> Option<f64> {
    if !p.iter().all(|v| v.is_finite())
        || p.x < 1.0
        || p.y < 1.0
        || p.x >= (camera.width - 1) as f64
        || p.y >= (camera.height - 1) as f64
    {
        return None;
    }
    let x = p.x as usize;
    let y = p.y as usize;
    let mut sum = 0;
    for row in y - 1..=y + 1 {
        for col in x - 1..=x + 1 {
            sum += image[row * camera.width + col] as u32;
        }
    }
    Some(sum as f64 / 9.0)
}

/// Verify that each projected convex-hull edge separates body pixels from dark background.
/// Nine samples per edge, a +/-8 px raster/fit tolerance, and 3x3 intensity means.
/// Saturation, shadows, cropping, or occlusion may reject a correct pose: abstention
/// is preferred to accepting a corner fit without independent boundary evidence.
/// Fixed storage; supports 3..32 convex-body vertices. No pose prior or truth input.
pub fn verify(
    image: &[u8],
    camera: Camera,
    vertices: &[V3],
    pose: &Pose,
) -> Option<BoundarySupport> {
    if !(3..=32).contains(&vertices.len())
        || camera.width < 8
        || camera.height < 8
        || camera.width.checked_mul(camera.height) != Some(image.len())
    {
        return None;
    }
    let mut projected = [Vector2::zeros(); 32];
    for (i, v) in vertices.iter().enumerate() {
        projected[i] = camera.project(pose.transform(*v))?;
    }
    // Bounded insertion sort, then monotone-chain hull (duplicate projections allowed).
    let n = vertices.len();
    for i in 1..n {
        let mut j = i;
        while j > 0
            && (projected[j]
                .x
                .total_cmp(&projected[j - 1].x)
                .then(projected[j].y.total_cmp(&projected[j - 1].y)))
            .is_lt()
        {
            projected.swap(j, j - 1);
            j -= 1;
        }
    }
    let mut hull = [Vector2::zeros(); 64];
    let mut len = 0;
    for p in &projected[..n] {
        while len >= 2 && cross(hull[len - 2], hull[len - 1], *p) <= 0.0 {
            len -= 1;
        }
        hull[len] = *p;
        len += 1;
    }
    let lower = len;
    for p in projected[..n - 1].iter().rev() {
        while len > lower && cross(hull[len - 2], hull[len - 1], *p) <= 0.0 {
            len -= 1;
        }
        hull[len] = *p;
        len += 1;
    }
    if len < 4 {
        return None;
    }
    len -= 1;
    let mut result = BoundarySupport {
        supported: 0,
        samples: 0,
    };
    for edge in 0..len {
        let a = hull[edge];
        let delta = hull[(edge + 1) % len] - a;
        let length = delta.norm();
        if length < 8.0 {
            return None;
        }
        let inside = Vector2::new(-delta.y, delta.x) / length;
        let mut supported = 0;
        for i in 0..9 {
            let center = a + delta * ((i as f64 + 0.5) / 9.0);
            let evidence = (-4..=4).any(|offset| {
                let p = center + inside * (offset * 2) as f64;
                let (Some(inner), Some(outer)) = (
                    intensity(image, camera, p + inside * 4.0),
                    intensity(image, camera, p - inside * 4.0),
                ) else {
                    return false;
                };
                let rim = intensity(image, camera, p)
                    .unwrap_or(0.0)
                    .max(intensity(image, camera, p + inside).unwrap_or(0.0));
                let body = inner.max(rim);
                body > outer + 12.0 && outer < body * 0.5 + 8.0
            });
            if evidence {
                supported += 1;
            }
        }
        // All edges must be supported; a local bright false patch cannot validate a body.
        if supported < 5 {
            return None;
        }
        result.supported += supported;
        result.samples += 9;
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use apsis_math::{target, Q};
    #[test]
    fn rejects_interior_texture_fit_and_missing_boundary() {
        let camera = Camera::new(640, 480, 1500.0);
        let truth = apsis_sim::truth(5.0);
        let mut image = apsis_sim::scene::render(
            camera,
            &truth,
            5.0,
            apsis_sim::scene::Conditions::default(),
            &mut apsis_sim::Noise::new(42),
        );
        assert!(verify(&image.gray, camera, &target(), &truth).is_some());
        let wrong = Pose {
            position: V3::new(0.0, 0.0, 2.3),
            attitude: truth.attitude,
        };
        assert!(verify(&image.gray, camera, &target(), &wrong).is_none());
        assert!(verify(&[], camera, &target(), &truth).is_none());
        let behind = Pose {
            position: V3::new(0.0, 0.0, -1.0),
            attitude: Q::identity(),
        };
        assert!(verify(&image.gray, camera, &target(), &behind).is_none());
        image.gray.fill(0);
        assert!(verify(&image.gray, camera, &target(), &truth).is_none());
        // Bright corner patches alone must not masquerade as a supported outline.
        for vertex in target() {
            let p = camera.project(truth.transform(vertex)).unwrap();
            let x = p.x as usize;
            let y = p.y as usize;
            for row in y - 2..=y + 2 {
                for col in x - 2..=x + 2 {
                    image.gray[row * camera.width + col] = 255;
                }
            }
        }
        assert!(verify(&image.gray, camera, &target(), &truth).is_none());
    }
}

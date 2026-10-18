//! Harris corners from grayscale pixels, followed by prior-guided correspondence.
//! No intensity IDs, simulation geometry visibility, or truth pose are inputs.
use apsis_math::{Camera, Pose, Vector2, V3};

#[derive(Clone, Copy, Default)]
pub struct Feature {
    pub pixel: Vector2<f64>,
    pub score: f64,
}
pub struct Features {
    items: [Feature; 96],
    len: usize,
}
impl core::ops::Deref for Features {
    type Target = [Feature];
    fn deref(&self) -> &Self::Target {
        &self.items[..self.len]
    }
}
pub struct Matches {
    items: [(usize, Vector2<f64>); 32],
    len: usize,
}
impl core::ops::Deref for Matches {
    type Target = [(usize, Vector2<f64>)];
    fn deref(&self) -> &Self::Target {
        &self.items[..self.len]
    }
}
pub fn detect(image: &[u8], camera: Camera) -> Features {
    let mut result = Features {
        items: [Feature::default(); 96],
        len: 0,
    };
    if camera.width < 7
        || camera.height < 7
        || camera.width.checked_mul(camera.height) != Some(image.len())
    {
        return result;
    }
    let w = camera.width;
    for y in 3..camera.height - 3 {
        for x in 3..w - 3 {
            let (mut xx, mut yy, mut xy) = (0.0, 0.0, 0.0);
            for dy in -1_isize..=1 {
                for dx in -1_isize..=1 {
                    let i = (y as isize + dy) as usize * w + (x as isize + dx) as usize;
                    let gx = image[i + 1] as f64 - image[i - 1] as f64;
                    let gy = image[i + w] as f64 - image[i - w] as f64;
                    xx += gx * gx;
                    yy += gy * gy;
                    xy += gx * gy;
                }
            }
            let score = xx * yy - xy * xy - 0.045 * (xx + yy) * (xx + yy);
            if score < 5e8 {
                continue;
            }
            let feature = Feature {
                pixel: Vector2::new(x as f64, y as f64),
                score,
            };
            if let Some(i) = result
                .iter()
                .position(|f| (f.pixel - feature.pixel).norm_squared() < 36.0)
            {
                if score > result.items[i].score {
                    result.items[i] = feature;
                }
            } else if result.len < 96 {
                result.items[result.len] = feature;
                result.len += 1;
            } else {
                let i = (0..96)
                    .min_by(|a, b| result.items[*a].score.total_cmp(&result.items[*b].score))
                    .unwrap();
                if score > result.items[i].score {
                    result.items[i] = feature;
                }
            }
        }
    }
    result
}
/// Greedy one-to-one nearest-neighbor matching within an explicit pixel gate.
pub fn associate(
    camera: Camera,
    model: &[V3],
    features: &Features,
    prior: &Pose,
    gate: f64,
) -> Matches {
    associate_visible(camera, model, features, prior, gate, &[true; 32])
}

/// Optional visibility is derived from the estimated mesh pose, never simulator truth.
pub fn associate_visible(
    camera: Camera,
    model: &[V3],
    features: &Features,
    prior: &Pose,
    gate: f64,
    visible: &[bool],
) -> Matches {
    let mut result = Matches {
        items: [(0, Vector2::zeros()); 32],
        len: 0,
    };
    if model.len() > 32 || visible.len() < model.len() || !gate.is_finite() || gate <= 0.0 {
        return result;
    }
    let mut used_models = [false; 32];
    let mut used_features = [false; 96];
    for _ in 0..model.len() {
        let mut best = None;
        let mut distance = gate * gate;
        for (m, point) in model.iter().enumerate() {
            if used_models[m] || !visible[m] {
                continue;
            }
            let Some(predicted) = camera.project(prior.transform(*point)) else {
                continue;
            };
            for (f, feature) in features.iter().enumerate() {
                if used_features[f] {
                    continue;
                }
                let d = (feature.pixel - predicted).norm_squared();
                if d < distance {
                    distance = d;
                    best = Some((m, f));
                }
            }
        }
        let Some((m, f)) = best else {
            break;
        };
        result.items[result.len] = (m, features[f].pixel);
        result.len += 1;
        used_models[m] = true;
        used_features[f] = true;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn finds_uncoded_image_corners_and_rejects_blank() {
        let camera = Camera::new(64, 64, 50.0);
        let mut image = [0_u8; 4096];
        for y in 16..48 {
            for x in 16..48 {
                image[y * 64 + x] = 180;
            }
        }
        let features = detect(&image, camera);
        assert!(features.len() >= 4);
        for (x, y) in [(16.0, 16.0), (47.0, 16.0), (16.0, 47.0), (47.0, 47.0)] {
            assert!(features
                .iter()
                .any(|f| (f.pixel - Vector2::new(x, y)).norm() < 3.0));
        }
        assert!(detect(&[0; 4096], camera).is_empty());
        assert!(detect(&[], camera).is_empty());
    }
}

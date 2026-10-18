//! CPU rasterized 3U body with directional diffuse/specular light and visible-surface texture.
//! This is a visual stress model, not a radiometrically calibrated space camera.
use crate::Noise;
use apsis_math::{landmarks, target, Camera, Pose, Vector2, V3};

pub struct Image {
    pub gray: Vec<u8>,
    pub rgb: Vec<u8>,
}
#[derive(Clone, Copy)]
pub struct Conditions {
    pub light_gain: f64,
    pub sun_angle_rad: f64,
    pub noise: f64,
    pub occlusion: f64,
    pub false_blobs: usize,
    pub glare: f64,
}
impl Default for Conditions {
    fn default() -> Self {
        Self {
            light_gain: 1.0,
            sun_angle_rad: 0.0,
            noise: 1.0,
            occlusion: 0.0,
            false_blobs: 0,
            glare: 1.0,
        }
    }
}

pub fn render(
    camera: Camera,
    pose: &Pose,
    time: f64,
    conditions: Conditions,
    rng: &mut Noise,
) -> Image {
    let w = camera.width;
    let h = camera.height;
    let mut rgb = vec![0_u8; w * h * 3];
    let mut depth = vec![f64::INFINITY; w * h];
    for i in 0..65 {
        let x = (i * 7919 + 37) % w;
        let y = (i * 3571 + 19) % h;
        let k = (y * w + x) * 3;
        rgb[k..k + 3].fill((35 + i % 80) as u8);
    }
    let body = target();
    let faces = [
        [0, 2, 6, 4],
        [1, 5, 7, 3],
        [0, 4, 5, 1],
        [2, 3, 7, 6],
        [0, 1, 3, 2],
        [4, 6, 7, 5],
    ];
    let sun = apsis_math::Q::from_scaled_axis(V3::new(0.0, conditions.sun_angle_rad, 0.0))
        * V3::new(-0.6, -0.4, -0.8).normalize();
    for face in faces {
        let pts = face.map(|i| pose.transform(body[i]));
        let normal = (pts[2] - pts[0]).cross(&(pts[1] - pts[0])).normalize();
        let center = (pts[0] + pts[1] + pts[2] + pts[3]) / 4.0;
        if normal.dot(&center) >= 0.0 {
            continue;
        }
        let halfway = (sun - center.normalize()).normalize();
        let diffuse = normal.dot(&sun).max(0.0);
        let specular = normal.dot(&halfway).max(0.0).powi(80) * conditions.glare * 360.0;
        raster(camera, pts, &mut rgb, &mut depth, |u, v| {
            let edge = !(0.035..=0.965).contains(&u) || !(0.025..=0.975).contains(&v);
            let grid = (u * 3.0).fract() < 0.045 || (v * 8.0).fract() < 0.05;
            if edge {
                let b = (105.0 + 110.0 * diffuse + specular).clamp(0.0, 255.0) as u8;
                [b, b, b]
            } else {
                let b =
                    (if grid { 48.0 } else { 12.0 } + 50.0 * diffuse + specular).clamp(0.0, 255.0);
                [(b * 0.45) as u8, (b * 0.65) as u8, b as u8]
            }
        });
    }
    let plate = landmarks();
    let ids = [8, 9, 11, 10];
    let pts = ids.map(|i| pose.transform(plate[i]));
    let front = pose.attitude * V3::new(0.0, 0.0, -1.0);
    if front.dot(&pts[0]) < 0.0 {
        raster(camera, pts, &mut rgb, &mut depth, |u, v| {
            let border = !(0.07..=0.93).contains(&u) || !(0.05..=0.95).contains(&v);
            let disk = ((u - 0.5) * 1.4).powi(2) + (v - 0.5).powi(2) < 0.10;
            let b = if border {
                235
            } else if disk {
                90
            } else {
                18
            };
            [b, b, b]
        });
    }
    if conditions.occlusion > 0.0 {
        let projected: Vec<_> = body
            .iter()
            .filter_map(|p| camera.project(pose.transform(*p)))
            .collect();
        let lo = projected.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
        let hi = projected
            .iter()
            .map(|p| p.x)
            .fold(f64::NEG_INFINITY, f64::max);
        let end = (lo + (hi - lo) * conditions.occlusion) as usize;
        for y in 0..h {
            for x in (lo.max(0.0) as usize)..end.min(w) {
                rgb[(y * w + x) * 3..(y * w + x) * 3 + 3].fill(0);
            }
        }
    }
    for _ in 0..conditions.false_blobs {
        let x = (w as f64 * (0.3 + (rng.uniform() + 0.5) * 0.4)) as usize;
        let y = (h as f64 * (0.25 + (rng.uniform() + 0.5) * 0.5)) as usize;
        for dy in 0..5 {
            for dx in 0..5 {
                if x + dx < w && y + dy < h {
                    let k = ((y + dy) * w + x + dx) * 3;
                    rgb[k..k + 3].fill(245);
                }
            }
        }
    }
    // A narrow directional flare sweeps the panels; the detector sees the affected pixels.
    let flare = ((time * 1.8).sin().max(0.0)).powi(35) * conditions.glare * 180.0;
    let mut gray = vec![0; w * h];
    for i in 0..w * h {
        let perturb = rng.uniform() * conditions.noise * 3.46;
        let flare_here = if depth[i].is_finite()
            && ((i % w) as f64 - camera.cx - (time * 2.0).sin() * 35.0).abs() < 8.0
        {
            flare
        } else {
            0.0
        };
        for c in 0..3 {
            rgb[i * 3 + c] = (rgb[i * 3 + c] as f64 * conditions.light_gain + perturb + flare_here)
                .clamp(0.0, 255.0) as u8;
            // Simplified camera black level, keeping baseline dark frames compact.
            if rgb[i * 3 + c] < 3 {
                rgb[i * 3 + c] = 0;
            }
        }
        gray[i] = (0.2126 * rgb[i * 3] as f64
            + 0.7152 * rgb[i * 3 + 1] as f64
            + 0.0722 * rgb[i * 3 + 2] as f64) as u8;
    }
    Image { gray, rgb }
}

fn raster(
    camera: Camera,
    points: [V3; 4],
    rgb: &mut [u8],
    depth: &mut [f64],
    color: impl Fn(f64, f64) -> [u8; 3],
) {
    let Some(pixels) = points
        .iter()
        .map(|p| camera.project(*p))
        .collect::<Option<Vec<_>>>()
    else {
        return;
    };
    let uv = [
        Vector2::new(0.0, 0.0),
        Vector2::new(1.0, 0.0),
        Vector2::new(1.0, 1.0),
        Vector2::new(0.0, 1.0),
    ];
    for tri in [[0, 1, 2], [0, 2, 3]] {
        let [a, b, c] = tri.map(|i| pixels[i]);
        let det = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
        if det.abs() < 1e-8 {
            continue;
        }
        let minx = a.x.min(b.x).min(c.x).floor().max(0.0) as usize;
        let maxx = a.x.max(b.x).max(c.x).ceil().min(camera.width as f64 - 1.0) as usize;
        let miny = a.y.min(b.y).min(c.y).floor().max(0.0) as usize;
        let maxy = a.y.max(b.y).max(c.y).ceil().min(camera.height as f64 - 1.0) as usize;
        for y in miny..=maxy {
            for x in minx..=maxx {
                let p = Vector2::new(x as f64 + 0.5, y as f64 + 0.5);
                let wa = ((b.y - c.y) * (p.x - c.x) + (c.x - b.x) * (p.y - c.y)) / det;
                let wb = ((c.y - a.y) * (p.x - c.x) + (a.x - c.x) * (p.y - c.y)) / det;
                let wc = 1.0 - wa - wb;
                if wa < 0.0 || wb < 0.0 || wc < 0.0 {
                    continue;
                }
                let weights = [
                    wa / points[tri[0]].z,
                    wb / points[tri[1]].z,
                    wc / points[tri[2]].z,
                ];
                let inverse = weights.iter().sum::<f64>();
                let z = 1.0 / inverse;
                let index = y * camera.width + x;
                if z < depth[index] {
                    let coord = (uv[tri[0]] * weights[0]
                        + uv[tri[1]] * weights[1]
                        + uv[tri[2]] * weights[2])
                        / inverse;
                    depth[index] = z;
                    rgb[index * 3..index * 3 + 3].copy_from_slice(&color(coord.x, coord.y));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_visual_failure_control_changes_sensor_pixels() {
        let camera = Camera::new(160, 120, 250.0);
        let pose = crate::truth(0.9);
        let base = render(
            camera,
            &pose,
            0.9,
            Conditions::default(),
            &mut Noise::new(42),
        );
        for changed in [
            Conditions {
                noise: 10.0,
                ..Conditions::default()
            },
            Conditions {
                occlusion: 0.5,
                ..Conditions::default()
            },
            Conditions {
                false_blobs: 5,
                ..Conditions::default()
            },
            Conditions {
                glare: 3.0,
                ..Conditions::default()
            },
        ] {
            let image = render(camera, &pose, 0.9, changed, &mut Noise::new(42));
            assert_ne!(image.gray, base.gray);
        }
    }
    #[test]
    fn pixels_respond_to_occlusion_and_noise_reproducibly() {
        let camera = Camera::new(160, 120, 250.0);
        let p = crate::truth(0.0);
        let a = render(camera, &p, 0.0, Conditions::default(), &mut Noise::new(7));
        let b = render(camera, &p, 0.0, Conditions::default(), &mut Noise::new(7));
        assert_eq!(a.gray, b.gray);
        let c = render(
            camera,
            &p,
            0.0,
            Conditions {
                occlusion: 0.8,
                ..Conditions::default()
            },
            &mut Noise::new(7),
        );
        assert_ne!(a.gray, c.gray);
        assert!(a.gray.iter().any(|p| *p > 200));
    }
}

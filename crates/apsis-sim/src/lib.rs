//! Synthetic point emitters, intentionally without surface occlusion or lighting.
use apsis_math::{target, Camera, Pose, Q, V3};

pub fn truth(time: f64) -> Pose {
    Pose {
        position: V3::new(
            0.25 * (time * 0.2).sin(),
            0.15 * (time * 0.3).cos(),
            12.0 - time * 0.025,
        ),
        attitude: Q::from_scaled_axis(V3::new(0.018, 0.025, 0.035) * time),
    }
}

pub struct Noise(u64);
impl Noise {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 11) as f64 / ((1_u64 << 53) as f64) - 0.5
    }
}

pub fn render(camera: Camera, pose: &Pose, rng: &mut Noise, dropout: bool) -> Vec<u8> {
    let mut image = vec![0; camera.width * camera.height];
    if dropout {
        return image;
    }
    for (id, point) in target().iter().enumerate() {
        if let Some(mut pixel) = camera.project(pose.transform(*point)) {
            pixel.x += rng.uniform() * 0.8;
            pixel.y += rng.uniform() * 0.8;
            let (x, y) = (pixel.x.round() as isize, pixel.y.round() as isize);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (u, v) = (x + dx, y + dy);
                    if u >= 0 && v >= 0 && u < camera.width as isize && v < camera.height as isize {
                        image[v as usize * camera.width + u as usize] = 32 + id as u8 * 24;
                    }
                }
            }
        }
    }
    image
}

//! Intensity-coded synthetic marker centroids and iterative calibrated PnP.
use apsis_math::{target, Camera, Pose, SMatrix, SVector, Vector2};

pub fn detect(image: &[u8], camera: Camera) -> Vec<(usize, Vector2<f64>)> {
    if image.len() != camera.width * camera.height { return Vec::new(); }
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
    sums.iter().enumerate().filter(|(_, s)| s.2 >= 4).map(|(id, s)|
        (id, Vector2::new(s.0 / s.2 as f64, s.1 / s.2 as f64))).collect()
}

pub fn solve(camera: Camera, points: &[(usize, Vector2<f64>)], seed: &Pose) -> Option<(Pose, f64)> {
    if points.len() < 6 || points.iter().any(|(id,p)| *id >= 8 || !p.iter().all(|v| v.is_finite())) { return None; }
    let mut seen = [false; 8];
    for (id, _) in points {
        if seen[*id] { return None; }
        seen[*id] = true;
    }
    let model = target();
    let mut pose = seed.clone();
    for _ in 0..30 {
        let mut normal = SMatrix::<f64,6,6>::zeros();
        let mut gradient = SVector::<f64,6>::zeros();
        for (id, observed) in points {
            let predicted = camera.project(pose.transform(model[*id]))?;
            let mut jac = SMatrix::<f64,2,6>::zeros();
            for k in 0..6 {
                let mut d = SVector::<f64,6>::zeros(); d[k] = 1e-6;
                let shifted = camera.project(pose.corrected(&d).transform(model[*id]))?;
                jac.set_column(k, &((shifted - predicted) / 1e-6));
            }
            normal += jac.transpose() * jac;
            gradient += jac.transpose() * (observed - predicted);
        }
        let delta = (normal + SMatrix::identity() * 1e-6).lu().solve(&gradient)?;
        if !delta.iter().all(|x| x.is_finite()) || delta.norm() > 5.0 { return None; }
        pose = pose.corrected(&delta);
        if delta.norm() < 1e-7 { break; }
    }
    let mut squared = 0.0;
    for (id, observed) in points {
        squared += (camera.project(pose.transform(model[*id]))? - observed).norm_squared();
    }
    let rms = (squared / points.len() as f64).sqrt();
    (rms < 3.0).then_some((pose, rms))
}

#[cfg(test)]
mod tests {
    use super::*;
    use apsis_math::{V3,Q};
    #[test]
    fn recovers_pose_from_pixels() {
        let c = Camera {width:640,height:480,focal:500.0};
        let truth = Pose {position:V3::new(0.4,-0.2,10.0),attitude:Q::from_scaled_axis(V3::new(0.15,0.2,-0.1))};
        let image = apsis_sim::render(c,&truth,&mut apsis_sim::Noise::new(7),false);
        let points = detect(&image,c);
        let seed = Pose {position:V3::new(0.0,0.0,12.0),attitude:Q::identity()};
        let (actual,rms) = solve(c,&points,&seed).unwrap();
        assert!((actual.position-truth.position).norm()<0.12);
        assert!((actual.attitude*truth.attitude.inverse()).angle()<0.04);
        assert!(rms<1.0);
        assert!(solve(c,&points[..4],&seed).is_none());
        assert!(detect(&[],c).is_empty());
        assert!(solve(c,&[points[0];6],&seed).is_none());
    }
}

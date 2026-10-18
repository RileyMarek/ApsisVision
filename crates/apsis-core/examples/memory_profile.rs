//! Reproducible host footprint and constrained-stack smoke test, not a WCET/stack peak proof.
use apsis_core::{
    guidance::{Guidance, Input},
    math::{landmarks, Camera, Pose, Q, V3},
    navigation::{Navigator, Rejection},
    vision::{corners, silhouette, solve_robust},
    Filter,
};
fn workload() {
    let camera = Camera::new(64, 64, 60.0);
    let mut pixels = [0_u8; 4096];
    for y in 12..52 {
        for x in 12..52 {
            pixels[y * 64 + x] = 220;
        }
    }
    let mut nav = Navigator::default();
    let mut guidance = Guidance::default();
    let pose = Pose {
        position: V3::new(0.0, 0.0, 1.6),
        attitude: Q::identity(),
    };
    let points = landmarks().map(|p| camera.project(pose.transform(p)).unwrap());
    let points: [_; 12] = core::array::from_fn(|i| (i, points[i]));
    let boundary = [
        V3::new(-0.5, -0.5, 0.0),
        V3::new(0.5, -0.5, 0.0),
        V3::new(0.5, 0.5, 0.0),
        V3::new(-0.5, 0.5, 0.0),
    ];
    for i in 0..200 {
        assert!(std::hint::black_box(silhouette::verify(
            std::hint::black_box(&pixels),
            camera,
            &boundary,
            &pose,
        ))
        .is_some());
        let features = corners::detect(std::hint::black_box(&pixels), camera);
        std::hint::black_box(corners::associate(
            camera,
            &landmarks(),
            &features,
            &pose,
            15.0,
        ));
        let solved = solve_robust(camera, &landmarks(), &points, &pose)
            .unwrap()
            .0;
        let t = i as f64 * 0.02;
        nav.predict_to(t).unwrap();
        nav.observe(t, Some(&solved), Rejection::None);
        std::hint::black_box(nav.health());
        std::hint::black_box(guidance.update(Input {
            pose: &solved,
            velocity: V3::zeros(),
            angular_rate: V3::zeros(),
            valid: nav.health().valid,
            timestamp_s: t,
            sigma_position_m: 0.01,
            in_fov: true,
        }));
    }
}
fn main() {
    const REQUESTED_STACK: usize = 128 * 1024;
    std::thread::Builder::new()
        .stack_size(REQUESTED_STACK)
        .spawn(workload)
        .unwrap()
        .join()
        .unwrap();
    println!("{{\"filter_bytes\":{},\"navigator_bytes\":{},\"feature_buffer_bytes\":{},\"match_buffer_bytes\":{},\"image_bytes\":4096,\"requested_host_thread_stack_bytes\":{},\"workload_completed\":true}}",std::mem::size_of::<Filter>(),std::mem::size_of::<Navigator>(),std::mem::size_of::<corners::Features>(),std::mem::size_of::<corners::Matches>(),REQUESTED_STACK);
    eprintln!("Storage sizes exclude temporary stack frames. The OS may round the thread stack. Completion is not a high-water measurement or embedded stack guarantee.");
}

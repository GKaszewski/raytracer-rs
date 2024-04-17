use clap::Parser;
use gltf::{buffer::Data, mesh::Primitive};
use std::sync::Arc;

use image::RgbImage;
use nalgebra::Vector3;
use rayon::iter::{ParallelBridge, ParallelIterator};

fn random_vec3() -> Vector3<f32> {
    Vector3::new(
        rand::random::<f32>(),
        rand::random::<f32>(),
        rand::random::<f32>(),
    )
}

fn random_vec3_range(min: f32, max: f32) -> Vector3<f32> {
    Vector3::new(
        rand::random::<f32>() * (max - min) + min,
        rand::random::<f32>() * (max - min) + min,
        rand::random::<f32>() * (max - min) + min,
    )
}

fn random_vec3_in_unit_sphere() -> Vector3<f32> {
    loop {
        let p = random_vec3_range(-1.0, 1.0);
        if p.norm_squared() < 1.0 {
            return p;
        }
    }
}

fn random_vec3_unit() -> Vector3<f32> {
    random_vec3_in_unit_sphere().normalize()
}

fn random_vec3_on_hemisphere(normal: Vector3<f32>) -> Vector3<f32> {
    let in_unit_sphere = random_vec3_in_unit_sphere();
    if in_unit_sphere.dot(&normal) > 0.0 {
        in_unit_sphere
    } else {
        -in_unit_sphere
    }
}

fn linear_to_gamma(value: f32) -> f32 {
    if value > 0.0 {
        value.sqrt()
    } else {
        0.0
    }
}

fn random_color() -> Vector3<f32> {
    Vector3::new(
        rand::random::<f32>(),
        rand::random::<f32>(),
        rand::random::<f32>(),
    )
}

fn near_zero(vec: Vector3<f32>) -> bool {
    let s = 1e-8;
    vec.x.abs() < s && vec.y.abs() < s && vec.z.abs() < s
}

fn reflect(v: Vector3<f32>, n: Vector3<f32>) -> Vector3<f32> {
    v - 2.0 * v.dot(&n) * n
}

fn refract(uv: Vector3<f32>, n: Vector3<f32>, etai_over_etat: f32) -> Vector3<f32> {
    let cos_theta = f32::min((-uv).dot(&n), 1.0);
    let r_out_perp = etai_over_etat * (uv + cos_theta * n);
    let r_out_parallel = -f32::sqrt(f32::abs(1.0 - r_out_perp.norm_squared())) * n;
    r_out_perp + r_out_parallel
}

struct Interval {
    min: f32,
    max: f32,
}

impl Interval {
    fn new(min: f32, max: f32) -> Self {
        Interval { min, max }
    }

    fn size(&self) -> f32 {
        self.max - self.min
    }

    fn contains(&self, x: f32) -> bool {
        self.min <= x && x <= self.max
    }

    fn surrounds(&self, x: f32) -> bool {
        self.min <= x && x <= self.max
    }

    fn empty() -> Self {
        Interval {
            min: std::f32::INFINITY,
            max: -std::f32::INFINITY,
        }
    }

    fn universe() -> Self {
        Interval {
            min: -std::f32::INFINITY,
            max: std::f32::INFINITY,
        }
    }

    fn clamp(&self, x: f32) -> f32 {
        if x < self.min {
            self.min
        } else if x > self.max {
            self.max
        } else {
            x
        }
    }
}

struct HitData {
    t: f32,
    point: Vector3<f32>,
    normal: Vector3<f32>,
    front_face: bool,
    material: Arc<dyn Material>,
}

impl HitData {
    fn set_face_normal(&mut self, ray: &Ray, outward_normal: Vector3<f32>) {
        self.front_face = ray.direction.dot(&outward_normal) < 0.0;
        self.normal = if self.front_face {
            outward_normal
        } else {
            -outward_normal
        };
    }
}

trait Hitable: Sync {
    fn hit(&self, ray: &Ray, ray_t: &Interval) -> Option<HitData>;
}

trait Material: Sync + Send {
    fn scatter(&self, ray: &Ray, hit_data: &HitData) -> Option<(Vector3<f32>, Ray)>;
}

struct MaterialLambertian {
    albedo: Vector3<f32>,
}

impl Material for MaterialLambertian {
    fn scatter(&self, _ray: &Ray, hit_data: &HitData) -> Option<(Vector3<f32>, Ray)> {
        let mut scatter_direction = hit_data.normal + random_vec3_unit();
        if near_zero(scatter_direction) {
            scatter_direction = hit_data.normal;
        }

        let scattered = Ray {
            origin: hit_data.point,
            direction: scatter_direction,
        };
        Some((self.albedo, scattered))
    }
}

struct MaterialMetal {
    albedo: Vector3<f32>,
    fuzz: f32,
}

impl Material for MaterialMetal {
    fn scatter(&self, ray: &Ray, hit_data: &HitData) -> Option<(Vector3<f32>, Ray)> {
        let mut reflected = reflect(ray.direction.normalize(), hit_data.normal);
        reflected = reflected.normalize() + (self.fuzz * random_vec3_in_unit_sphere());
        let scattered = Ray {
            origin: hit_data.point,
            direction: reflected,
        };
        if scattered.direction.dot(&hit_data.normal) > 0.0 {
            Some((self.albedo, scattered))
        } else {
            None
        }
    }
}

struct MaterialDielectric {
    refraction_index: f32,
}

impl Material for MaterialDielectric {
    fn scatter(&self, ray: &Ray, hit_data: &HitData) -> Option<(Vector3<f32>, Ray)> {
        let attenuation = Vector3::new(1.0, 1.0, 1.0);
        let refraction_ratio = if hit_data.front_face {
            1.0 / self.refraction_index
        } else {
            self.refraction_index
        };

        let unit_direction = ray.direction.normalize();
        let cos_theta = f32::min(-unit_direction.dot(&hit_data.normal), 1.0);
        let sin_theta = f32::sqrt(1.0 - cos_theta * cos_theta);
        let cannot_refract = refraction_ratio * sin_theta > 1.0;
        let mut direction = Vector3::new(0.0, 0.0, 0.0);

        if cannot_refract || Self::reflectance(cos_theta, refraction_ratio) > rand::random::<f32>()
        {
            direction = reflect(unit_direction, hit_data.normal);
        } else {
            direction = refract(unit_direction, hit_data.normal, refraction_ratio);
        }

        let scattered = Ray {
            origin: hit_data.point,
            direction,
        };

        Some((attenuation, scattered))
    }
}

impl MaterialDielectric {
    fn reflectance(cosine: f32, ref_idx: f32) -> f32 {
        let r0 = ((1.0 - ref_idx) / (1.0 + ref_idx)).powi(2);
        r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
    }
}

struct Sphere {
    center: Vector3<f32>,
    radius: f32,
    material: Arc<dyn Material>,
}

impl Hitable for Sphere {
    fn hit(&self, ray: &Ray, ray_t: &Interval) -> Option<HitData> {
        let origin_to_center = self.center - ray.origin;
        let a = ray.direction.dot(&ray.direction);
        let h = ray.direction.dot(&origin_to_center);
        let c = origin_to_center.dot(&origin_to_center) - self.radius * self.radius;
        let discriminant = h * h - a * c;
        if discriminant < 0.0 {
            return None;
        }

        let sqrt_discriminant = discriminant.sqrt();
        let mut root = (h - sqrt_discriminant) / a;
        if !ray_t.surrounds(root) {
            root = (h + sqrt_discriminant) / a;
            if !ray_t.surrounds(root) {
                return None;
            }
        }

        let point = ray.at(root);
        let outward_normal = (point - self.center) / self.radius;
        let mut hit_data = HitData {
            t: root,
            point,
            normal: Vector3::new(0.0, 0.0, 0.0),
            front_face: false,
            material: self.material.clone(),
        };
        hit_data.set_face_normal(ray, outward_normal);
        Some(hit_data)
    }
}

struct Cube {
    min: Vector3<f32>,
    max: Vector3<f32>,
    material: Arc<dyn Material>,
}

impl Hitable for Cube {
    fn hit(&self, ray: &Ray, ray_t: &Interval) -> Option<HitData> {
        let mut t_min = ray_t.min;
        let mut t_max = ray_t.max;

        for i in 0..3 {
            let inv_d = 1.0 / ray.direction[i];
            let mut t0 = (self.min[i] - ray.origin[i]) * inv_d;
            let mut t1 = (self.max[i] - ray.origin[i]) * inv_d;
            if inv_d < 0.0 {
                std::mem::swap(&mut t0, &mut t1);
            }
            t_min = f32::max(t0, t_min);
            t_max = f32::min(t1, t_max);
            if t_max <= t_min {
                return None;
            }
        }

        let t = if ray_t.surrounds(t_min) { t_min } else { t_max };

        let point = ray.at(t);
        let outward_normal = Vector3::new(0.0, 0.0, 0.0);
        let mut hit_data = HitData {
            t,
            point,
            normal: Vector3::new(0.0, 0.0, 0.0),
            front_face: false,
            material: self.material.clone(),
        };
        hit_data.set_face_normal(ray, outward_normal);
        Some(hit_data)
    }
}

#[derive(Clone)]
struct Triangle {
    vertices: [Vector3<f32>; 3],
    material: Arc<dyn Material>,
}

impl Hitable for Triangle {
    fn hit(&self, ray: &Ray, ray_t: &Interval) -> Option<HitData> {
        let [v0, v1, v2] = self.vertices;
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let h = ray.direction.cross(&edge2);
        let a = edge1.dot(&h);
        if a > -0.00001 && a < 0.00001 {
            return None;
        }

        let f = 1.0 / a;
        let s = ray.origin - v0;
        let u = f * s.dot(&h);
        if u < 0.0 || u > 1.0 {
            return None;
        }

        let q = s.cross(&edge1);
        let v = f * ray.direction.dot(&q);
        if v < 0.0 || u + v > 1.0 {
            return None;
        }

        let t = f * edge2.dot(&q);
        if !ray_t.surrounds(t) {
            return None;
        }

        let point = ray.at(t);
        let outward_normal = edge1.cross(&edge2).normalize();
        let front_face = ray.direction.dot(&outward_normal) < 0.0;
        let normal = if front_face {
            outward_normal
        } else {
            -outward_normal
        };
        let mut hit_data = HitData {
            t,
            point,
            normal,
            front_face,
            material: self.material.clone(),
        };
        hit_data.set_face_normal(ray, outward_normal);
        Some(hit_data)
    }
}

#[derive(Clone)]
struct Mesh {
    triangles: Vec<Triangle>,
}

impl Mesh {
    fn new(vertices: Vec<Vector3<f32>>, indices: Vec<u32>, material: Arc<dyn Material>) -> Self {
        let mut triangles = Vec::new();
        for chunk in indices.chunks(3) {
            if chunk.len() == 3 {
                let v0 = vertices[chunk[0] as usize];
                let v1 = vertices[chunk[1] as usize];
                let v2 = vertices[chunk[2] as usize];
                triangles.push(Triangle {
                    vertices: [v0, v1, v2],
                    material: material.clone(),
                });
            }
        }

        Mesh { triangles }
    }
}

impl Hitable for Mesh {
    fn hit(&self, ray: &Ray, ray_t: &Interval) -> Option<HitData> {
        let mut closest_so_far = ray_t.max;
        let mut hit_data = None;
        for triangle in &self.triangles {
            if let Some(data) = triangle.hit(ray, &Interval::new(ray_t.min, closest_so_far)) {
                closest_so_far = data.t;
                hit_data = Some(data);
            }
        }
        hit_data
    }
}

struct HitableList {
    hitables: Vec<Box<dyn Hitable>>,
}

impl HitableList {
    fn new() -> Self {
        HitableList {
            hitables: Vec::new(),
        }
    }

    fn add(&mut self, hitable: Box<dyn Hitable>) {
        self.hitables.push(hitable);
    }

    fn clear(&mut self) {
        self.hitables.clear();
    }
}

impl Hitable for HitableList {
    fn hit(&self, ray: &Ray, ray_t: &Interval) -> Option<HitData> {
        let mut closest_so_far = ray_t.max;
        let mut hit_data = None;
        for hitable in &self.hitables {
            if let Some(data) = hitable.hit(ray, &Interval::new(ray_t.min, closest_so_far)) {
                closest_so_far = data.t;
                hit_data = Some(data);
            }
        }
        hit_data
    }
}

struct Ray {
    origin: Vector3<f32>,
    direction: Vector3<f32>,
}

impl Ray {
    fn at(&self, t: f32) -> Vector3<f32> {
        self.origin + t * self.direction
    }

    fn color(&self, world: &HitableList) -> Vector3<f32> {
        if let Some(hit_data) = world.hit(self, &Interval::new(0.0, std::f32::INFINITY)) {
            return 0.5 * (hit_data.normal + Vector3::new(1.0, 1.0, 1.0));
        }

        let unit_direction = self.direction.normalize();
        let a = 0.5 * (unit_direction.y + 1.0);
        (1.0 - a) * Vector3::new(1.0, 1.0, 1.0) + a * Vector3::new(0.5, 0.7, 1.0)
    }
}

struct Camera {
    aspect_ratio: f32,
    image_width: u32,
    image_height: u32,
    center: Vector3<f32>,
    pixel00_loc: Vector3<f32>,
    pixel_delta_u: Vector3<f32>,
    pixel_delta_v: Vector3<f32>,
    samples_per_pixel: u32,
    pixel_samples_scale: f32,
    max_depth: u32,
    vfov: f32,
    look_from: Vector3<f32>,
    look_at: Vector3<f32>,
    up: Vector3<f32>,
}

impl Camera {
    fn intitialize(&mut self) {
        self.image_height = (self.image_width as f32 / self.aspect_ratio) as u32;
        self.image_height = if self.image_height < 1 {
            1
        } else {
            self.image_height
        };

        self.pixel_samples_scale = 1.0 / self.samples_per_pixel as f32;

        self.center = self.look_from;
        let focal_length = (self.look_from - self.look_at).norm();
        let theta = self.vfov * std::f32::consts::PI / 180.0;
        let h = f32::tan(theta / 2.0);
        let viewport_height = 2.0 * h * focal_length;
        let viewport_width = viewport_height * (self.image_width as f32 / self.image_height as f32);

        let w = (self.look_from - self.look_at).normalize();
        let u = self.up.cross(&w).normalize();
        let v = w.cross(&u);

        let viewport_u = u * viewport_width;
        let viewport_v = -v * viewport_height;

        self.pixel_delta_u = viewport_u / self.image_width as f32;
        self.pixel_delta_v = viewport_v / self.image_height as f32;

        let viewport_upper_left =
            self.center - (focal_length * w) - (0.5 * viewport_u) - (0.5 * viewport_v);

        self.pixel00_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v);
    }

    fn ray_color(&self, ray: &Ray, world: &HitableList, depth: u32) -> Vector3<f32> {
        if depth == 0 {
            return Vector3::new(0.0, 0.0, 0.0);
        }

        if let Some(hit_data) = world.hit(ray, &Interval::new(0.001, std::f32::INFINITY)) {
            let scatter = hit_data.material.scatter(ray, &hit_data);
            if let Some((attenuation, scattered)) = scatter {
                return attenuation.component_mul(&self.ray_color(&scattered, world, depth - 1));
            }

            return Vector3::new(0.0, 0.0, 0.0);
        }

        let unit_direction = ray.direction.normalize();
        let a = 0.5 * (unit_direction.y + 1.0);
        (1.0 - a) * Vector3::new(1.0, 1.0, 1.0) + a * Vector3::new(0.5, 0.7, 1.0)
    }

    fn get_ray(&self, u: f32, v: f32) -> Ray {
        let offset = self.sample_square();
        let pixel_sample = self.pixel00_loc
            + ((u + offset.x) * self.pixel_delta_u)
            + ((v + offset.y) * self.pixel_delta_v);
        let ray_origin = self.center;
        let ray_direction = pixel_sample - ray_origin;

        Ray {
            origin: ray_origin,
            direction: ray_direction,
        }
    }

    fn sample_square(&self) -> Vector3<f32> {
        Vector3::new(
            rand::random::<f32>() - 0.5,
            rand::random::<f32>() - 0.5,
            0.0,
        )
    }

    fn render(&self, img: &mut RgbImage, world: &HitableList) {
        img.enumerate_rows_mut().par_bridge().for_each(|(y, row)| {
            for (x, y, pixel) in row {
                let mut pixel_color = Vector3::new(0.0, 0.0, 0.0);
                for _ in 0..self.samples_per_pixel {
                    let ray = self.get_ray(x as f32, y as f32);
                    pixel_color += self.ray_color(&ray, world, self.max_depth);
                }
                pixel_color *= self.pixel_samples_scale;

                let r = linear_to_gamma(pixel_color.x);
                let g = linear_to_gamma(pixel_color.y);
                let b = linear_to_gamma(pixel_color.z);

                let intensity = Interval::new(0.000, 0.999);
                let r_byte = 256.0 * intensity.clamp(r);
                let g_byte = 256.0 * intensity.clamp(g);
                let b_byte = 256.0 * intensity.clamp(b);

                *pixel = image::Rgb([r_byte as u8, g_byte as u8, b_byte as u8]);
            }
            //println!("Scanlines remaining: {}", self.image_height - y);
        });
    }
}

fn load_gltf_mesh(path: &str) -> Vec<Mesh> {
    let (gltf, buffers, _) = gltf::import(path).expect("Failed to load gltf file");
    let mut meshes = Vec::new();

    for mesh in gltf.meshes() {
        for primitive in mesh.primitives() {
            if let Some(mesh) = process_primitive(primitive, &buffers) {
                meshes.push(mesh);
            }
        }
    }

    meshes
}

fn process_primitive(primitive: Primitive, buffers: &[Data]) -> Option<Mesh> {
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()].0));

    let positions = reader
        .read_positions()?
        .map(Vector3::from)
        .map(|pos| Vector3::new(pos[0], pos[1], pos[2]))
        .collect::<Vec<_>>();

    println!("Positions: {:?}", positions);

    let indices = reader
        .read_indices()
        .map(|indices| indices.into_u32().collect::<Vec<_>>())?;

    let material = Arc::new(MaterialMetal {
        albedo: random_color(),
        fuzz: 0.4,
    });

    Some(Mesh::new(positions, indices, material))
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "400")]
    width: u32,
    #[arg(short, long, default_value = "20.0")]
    vfov: f32,
    #[arg(short, long, default_value = "10")]
    samples: u32,
    #[arg(short, long, default_value = "50")]
    max_depth: u32,
}

fn monkey_scene() -> HitableList {
    let monkey_mesh = load_gltf_mesh("monke.glb");
    let ground_material = Arc::new(MaterialLambertian {
        albedo: Vector3::new(1.0, 0.5, 0.5),
    });

    let mut world = HitableList::new();

    world.add(Box::new(Sphere {
        center: Vector3::new(0.0, -105.0, 0.0),
        radius: 100.0,
        material: ground_material.clone(),
    }));
    for mesh in monkey_mesh {
        println!("Adding mesh");
        world.add(Box::new(mesh));
    }

    world
}

fn balls_scene() -> HitableList {
    let ground_material = Arc::new(MaterialLambertian {
        albedo: Vector3::new(1.0, 0.5, 0.5),
    });
    let center_material = Arc::new(MaterialLambertian {
        albedo: Vector3::new(0.1, 0.2, 0.5),
    });
    let left_material = Arc::new(MaterialDielectric {
        refraction_index: 1.5,
    });
    let right_material = Arc::new(MaterialMetal {
        albedo: Vector3::new(0.8, 0.6, 0.2),
        fuzz: 1.0,
    });
    let bubble_material = Arc::new(MaterialDielectric {
        refraction_index: 1.0 / 1.5,
    });

    let mut world = HitableList::new();
    world.add(Box::new(Sphere {
        center: Vector3::new(0.0, -100.5, -1.0),
        radius: 100.0,
        material: ground_material.clone(),
    }));
    world.add(Box::new(Sphere {
        center: Vector3::new(0.0, 0.0, -1.0),
        radius: 0.5,
        material: center_material.clone(),
    }));
    world.add(Box::new(Sphere {
        center: Vector3::new(-1.0, 0.0, -1.0),
        radius: 0.5,
        material: left_material.clone(),
    }));
    world.add(Box::new(Sphere {
        center: Vector3::new(-1.0, 0.0, -1.0),
        radius: -0.45,
        material: right_material.clone(),
    }));
    world.add(Box::new(Sphere {
        center: Vector3::new(1.0, 0.0, -1.0),
        radius: 0.5,
        material: bubble_material.clone(),
    }));

    world
}

fn main() {
    let args = Args::parse();
    let width = args.width;
    let vfov = args.vfov;
    let samples = args.samples;
    let max_depth = args.max_depth;

    let mut cam = Camera {
        aspect_ratio: 16.0 / 9.0,
        image_width: width,
        image_height: 0,
        center: Vector3::new(0.0, 0.0, 0.0),
        pixel00_loc: Vector3::new(0.0, 0.0, 0.0),
        pixel_delta_u: Vector3::new(0.0, 0.0, 0.0),
        pixel_delta_v: Vector3::new(0.0, 0.0, 0.0),
        samples_per_pixel: samples,
        pixel_samples_scale: 0.0,
        max_depth,
        vfov,
        look_from: Vector3::new(2.0, 2.0, 5.0),
        look_at: Vector3::new(0.0, 0.0, -1.0),
        up: Vector3::new(0.0, 1.0, 0.0),
    };
    cam.intitialize();

    let mut img = RgbImage::new(cam.image_width, cam.image_height);
    let world = monkey_scene();

    println!("Start writing image");
    let start = std::time::Instant::now();
    cam.render(&mut img, &world);

    println!("Done writing image");
    let duration = start.elapsed();
    println!("Time elapsed: {:?}", duration);

    img.save("output.png").unwrap();
}

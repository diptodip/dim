// #![no_std]
#![feature(core_float_math)]
#![feature(portable_simd)]
use core::iter::zip;
use core::ops::Index;
use core::ops::IndexMut;
use core::simd::f32x8;
use core::simd::mask32x8;
use core::simd::num::SimdFloat;

use rayon::prelude::*;

#[derive(PartialEq)]
pub struct Image<T>
where
    T: Clone + Copy,
{
    pub height: u32,
    pub width: u32,
    pub data: Vec<T>,
}

impl<T> Image<T>
where
    T: Clone + Copy,
{
    pub fn new(height: u32, width: u32, fill: T) -> Self {
        Image {
            height,
            width,
            data: vec![fill; height as usize * width as usize],
        }
    }

    pub fn from_slice(slice: &mut [T], height: u32, width: u32) -> Self {
        Image {
            height,
            width,
            data: unsafe {
                Vec::from_raw_parts(
                    slice.as_mut_ptr(),
                    height as usize * width as usize,
                    height as usize * width as usize,
                )
            },
        }
    }

    pub fn from_raw_parts(ptr: *mut T, height: u32, width: u32) -> Self {
        Image {
            height,
            width,
            data: unsafe {
                Vec::from_raw_parts(
                    ptr,
                    height as usize * width as usize,
                    height as usize * width as usize,
                )
            },
        }
    }

    pub fn from_vec(vec: Vec<T>, height: u32, width: u32) -> Self {
        Image {
            height,
            width,
            data: vec,
        }
    }

    pub fn to_view<'image>(&'image self) -> ImageView<'image, T> {
        ImageView::new(&self.data, self.height, self.width)
    }

    pub fn pad(
        mut self,
        height_top: u32,
        height_bottom: u32,
        width_left: u32,
        width_right: u32,
        fill: T,
    ) -> Self {
        let height = self.height + height_top + height_bottom;
        let width = self.width + width_left + width_right;
        let mut data = vec![fill; height as usize * width as usize];
        for y in height_top..(height_top + self.height) {
            for x in width_left..(width_left + self.width) {
                data[(y * width + x) as usize] =
                    self[((y - height_top) as usize, (x - width_left) as usize)];
            }
        }
        self.height = height;
        self.width = width;
        self.data = data;
        self
    }

    pub fn crop(
        mut self,
        height_top: u32,
        height_bottom: u32,
        width_left: u32,
        width_right: u32,
    ) -> Self {
        let height = self.height - (height_top + height_bottom);
        let width = self.width - (width_left + width_right);
        let mut data = vec![self[(0, 0)]; height as usize * width as usize];
        for y in height_top..(height_top + height) {
            for x in width_left..(width_left + width) {
                data[((y - height_top) * width + (x - width_left)) as usize] =
                    self[(y as usize, x as usize)];
            }
        }
        self.height = height;
        self.width = width;
        self.data = data;
        self
    }
}

impl<T> Index<(usize, usize)> for Image<T>
where
    T: Clone + Copy,
{
    type Output = T;
    fn index(&self, (y, x): (usize, usize)) -> &T {
        let index = y * self.width as usize + x;
        return &self.data[index];
    }
}

impl<T> IndexMut<(usize, usize)> for Image<T>
where
    T: Clone + Copy,
{
    fn index_mut(&mut self, (y, x): (usize, usize)) -> &mut T {
        let index = y * self.width as usize + x;
        return &mut self.data[index];
    }
}

#[derive(PartialEq)]
pub struct ImageView<'image, T>
where
    T: Clone + Copy,
{
    pub height: u32,
    pub width: u32,
    pub data: &'image [T],
}

impl<'image, T> ImageView<'image, T>
where
    T: Clone + Copy,
{
    pub fn new(data: &'image [T], height: u32, width: u32) -> Self {
        ImageView {
            height,
            width,
            data,
        }
    }
}

impl<'image, T> Index<(usize, usize)> for ImageView<'image, T>
where
    T: Clone + Copy,
{
    type Output = T;
    fn index(&self, (y, x): (usize, usize)) -> &T {
        let index = y * self.width as usize + x;
        return &self.data[index];
    }
}

pub trait Interpolate<T>
where
    T: Clone + Copy,
{
    fn interpolate(&self, image: &ImageView<T>) -> Image<T>;
}

pub struct RollingBallDownsample {
    pub downsample_factor: u32,
    pub pad_height_top: u32,
    pub pad_height_bottom: u32,
    pub pad_width_left: u32,
    pub pad_width_right: u32,
}

impl RollingBallDownsample {
    pub fn new(downsample_factor: u32) -> Self {
        Self {
            downsample_factor,
            pad_height_top: 0,
            pad_height_bottom: 0,
            pad_width_left: 0,
            pad_width_right: 0,
        }
    }

    pub fn with_padding(
        downsample_factor: u32,
        pad_height_top: u32,
        pad_height_bottom: u32,
        pad_width_left: u32,
        pad_width_right: u32,
    ) -> Self {
        Self {
            downsample_factor,
            pad_height_top,
            pad_height_bottom,
            pad_width_left,
            pad_width_right,
        }
    }

    pub fn with_symmetric_padding(downsample_factor: u32, pad: u32) -> Self {
        Self {
            downsample_factor,
            pad_height_top: pad,
            pad_height_bottom: pad,
            pad_width_left: pad,
            pad_width_right: pad,
        }
    }
}

impl Interpolate<f32> for RollingBallDownsample {
    fn interpolate(&self, image: &ImageView<f32>) -> Image<f32> {
        let downsampled_height =
            (image.height + self.downsample_factor - 1) / self.downsample_factor;
        let downsampled_width = (image.width + self.downsample_factor - 1) / self.downsample_factor;
        let padded_downsampled_height =
            downsampled_height + self.pad_height_top + self.pad_height_bottom;
        let padded_downsampled_width =
            downsampled_width + self.pad_width_left + self.pad_width_right;
        let mut downsampled =
            Image::<f32>::new(padded_downsampled_height, padded_downsampled_width, 0.0);
        for y_downsampled in self.pad_height_top..(self.pad_height_top + downsampled_height) {
            for x_downsampled in self.pad_width_left..(self.pad_width_left + downsampled_width) {
                let mut min = f32::MAX;
                let mut y = self.downsample_factor * (y_downsampled - self.pad_height_top);
                let mut x;
                let mut i = 0;
                let mut j;
                while y < image.height && i < self.downsample_factor {
                    j = 0;
                    x = self.downsample_factor * (x_downsampled - self.pad_width_left);
                    while x < image.width && j < self.downsample_factor {
                        let value = image[(y as usize, x as usize)];
                        if value < min {
                            min = value;
                        }
                        j += 1;
                        x += 1;
                    }
                    i += 1;
                    y += 1;
                }
                downsampled[(y_downsampled as usize, x_downsampled as usize)] = min;
            }
        }
        return downsampled;
    }
}

pub struct RollingBallUpsample {
    pub upsample_factor: u32,
    pub upsampled_height: u32,
    pub upsampled_width: u32,
    pub pad_height_top: u32,
    pub pad_height_bottom: u32,
    pub pad_width_left: u32,
    pub pad_width_right: u32,
}

impl RollingBallUpsample {
    pub fn new(upsample_factor: u32, upsampled_height: u32, upsampled_width: u32) -> Self {
        Self {
            upsample_factor,
            upsampled_height,
            upsampled_width,
            pad_height_top: 0,
            pad_height_bottom: 0,
            pad_width_left: 0,
            pad_width_right: 0,
        }
    }

    pub fn with_symmetric_cropping(
        upsample_factor: u32,
        upsampled_height: u32,
        upsampled_width: u32,
        crop: u32,
    ) -> Self {
        Self {
            upsample_factor,
            upsampled_height,
            upsampled_width,
            pad_height_top: crop,
            pad_height_bottom: crop,
            pad_width_left: crop,
            pad_width_right: crop,
        }
    }
}

impl Interpolate<f32> for RollingBallUpsample {
    fn interpolate(&self, image: &ImageView<f32>) -> Image<f32> {
        fn interpolation_arrays(
            image_indices: &mut [usize],
            weights: &mut [f32],
            image_length: usize,
            upsample_factor: u32,
        ) {
            for i in 0..weights.len() {
                let mut image_index =
                    ((i as i32 - upsample_factor as i32 / 2) / upsample_factor as i32) as usize;
                if image_index >= image_length - 1 {
                    image_index -= 2;
                }
                image_indices[i] = image_index;
                let distance =
                    (i as f32 + 0.5) / upsample_factor as f32 - (image_index as f32 + 0.5);
                weights[i] = 1.0 - distance;
            }
        }
        let mut upsampled = Image::<f32>::new(self.upsampled_height, self.upsampled_width, 0.0);
        let mut image_indices_y = vec![0; self.upsampled_height as usize];
        let mut weights_y = vec![0.0; self.upsampled_height as usize];
        interpolation_arrays(
            &mut image_indices_y,
            &mut weights_y,
            image.height as usize - (self.pad_height_top + self.pad_height_bottom) as usize,
            self.upsample_factor,
        );
        for i in image_indices_y.iter_mut() {
            *i += self.pad_height_top as usize;
        }
        let mut image_indices_x = vec![0; self.upsampled_width as usize];
        let mut weights_x = vec![0.0; self.upsampled_width as usize];
        interpolation_arrays(
            &mut image_indices_x,
            &mut weights_x,
            image.width as usize - (self.pad_width_left + self.pad_width_right) as usize,
            self.upsample_factor,
        );
        for i in image_indices_x.iter_mut() {
            *i += self.pad_width_left as usize;
        }
        let mut line0 = vec![0.0; self.upsampled_width as usize];
        let mut line1 = vec![0.0; self.upsampled_width as usize];
        for x in 0..self.upsampled_width as usize {
            line1[x] = image.data[image_indices_x[x]] * weights_x[x]
                + image.data[image_indices_x[x] + 1] * (1.0 - weights_x[x]);
        }
        let mut y_image_line0 = -1;
        for y in 0..self.upsampled_height as usize {
            if y_image_line0 < image_indices_y[y] as i32 {
                line0.swap_with_slice(&mut line1);
                y_image_line0 += 1;
                let image_row_after_line0_index = (image_indices_y[y] + 1) * image.width as usize;
                for x in 0..self.upsampled_width as usize {
                    line1[x] = image.data[image_row_after_line0_index + image_indices_x[x]]
                        * weights_x[x]
                        + image.data[image_row_after_line0_index + image_indices_x[x] + 1]
                            * (1.0 - weights_x[x]);
                }
            }
            let weight = weights_y[y];
            for x in 0..self.upsampled_width as usize {
                upsampled[(y, x)] = line0[x] * weight + line1[x] * (1.0 - weight);
            }
        }
        return upsampled;
    }
}

pub struct RollingBall<T> {
    pub downsample_factor: u32,
    pub radius: f32,
    pub kernel_width: usize,
    pub kernel: Vec<T>,
}

impl<T> RollingBall<T>
where
    T: From<f32>,
{
    pub fn new(radius: f32) -> RollingBall<T> {
        let (downsample_factor, arc_trim_percentage) = if radius <= 10.0 {
            (1, 0.24)
        } else if radius <= 30.0 {
            (2, 0.24)
        } else if radius <= 100.0 {
            (4, 0.32)
        } else {
            (8, 0.40)
        };
        RollingBall::with_downsample_and_arc_trim(radius, downsample_factor, arc_trim_percentage)
    }

    pub fn with_downsample_and_arc_trim(
        radius: f32,
        downsample_factor: u32,
        arc_trim_percentage: f32,
    ) -> RollingBall<T> {
        let downsampled_ball_radius = radius / downsample_factor as f32;
        let downsampled_ball_radius_sq = downsampled_ball_radius * downsampled_ball_radius;
        let trim = (arc_trim_percentage * downsampled_ball_radius) as isize;
        let half_width = downsampled_ball_radius as isize - trim;
        let kernel_width = 2 * half_width + 1;
        let mut kernel = Vec::with_capacity(kernel_width as usize * kernel_width as usize);
        for y in 0..kernel_width {
            for x in 0..kernel_width {
                let mut z = downsampled_ball_radius_sq
                    - (x - half_width).pow(2) as f32
                    - (y - half_width).pow(2) as f32;
                z = if z > 0.0 {
                    core::f32::math::sqrt(z)
                } else {
                    0.0
                };
                kernel.push(z.into());
            }
        }
        RollingBall {
            downsample_factor,
            radius,
            kernel_width: kernel_width as usize,
            kernel,
        }
    }

    pub fn with_no_downsample(radius: f32) -> RollingBall<T> {
        RollingBall::with_downsample_and_arc_trim(radius, 1, 0.24)
    }
}

impl<T> Index<(usize, usize)> for RollingBall<T> {
    type Output = T;
    fn index(&self, (y, x): (usize, usize)) -> &T {
        let index = y * self.kernel_width as usize + x;
        return &self.kernel[index];
    }
}

pub trait RemoveBackground<T>
where
    T: Clone + Copy + Send + Sync,
{
    fn estimate_background(&self, image: &ImageView<T>) -> Image<T>;

    fn estimate_background_stack(&self, image_stack: &[ImageView<T>]) -> Vec<T>
    where
        Self: Sync,
    {
        (0..image_stack.len())
            .into_par_iter()
            .map(|index| self.estimate_background(&image_stack[index]).data)
            .flatten()
            .collect()
    }
}

impl RemoveBackground<f32> for RollingBall<f32> {
    fn estimate_background(&self, image: &ImageView<f32>) -> Image<f32> {
        fn estimate_background_scalar(
            ball: &RollingBall<f32>,
            image: &ImageView<f32>,
        ) -> Image<f32> {
            let radius = (ball.kernel_width / 2) as i32;
            let downsampled = RollingBallDownsample::with_symmetric_padding(
                ball.downsample_factor,
                radius as u32,
            )
            .interpolate(image);
            let mut background = Image::new(downsampled.height, downsampled.width, f32::MIN);
            let height = background.height as i32;
            let width = background.width as i32;
            for y in radius..(height - radius) {
                let y_start = (y - radius) as usize;
                let y_end = (y + radius) as usize;
                for x in radius..(width - radius) {
                    let x_start = (x - radius) as usize;
                    let mut z = f32::MAX;
                    for (image_row, ball_row) in
                        zip(y_start as usize..=y_end as usize, 0..ball.kernel_width)
                    {
                        let image_start = image_row * downsampled.width as usize + x_start;
                        let image_end = image_start + ball.kernel_width;
                        let ball_start = ball_row * ball.kernel_width;
                        let ball_end = ball_start + ball.kernel_width;
                        for (i, k) in zip(
                            &downsampled.data[image_start..image_end],
                            &ball.kernel[ball_start..ball_end],
                        ) {
                            let z_reduced = i - k;
                            if z > z_reduced {
                                z = z_reduced;
                            }
                        }
                    }
                    for (background_row, ball_row) in
                        zip(y_start as usize..=y_end as usize, 0..ball.kernel_width)
                    {
                        let background_start = background_row * background.width as usize + x_start;
                        let background_end = background_start + ball.kernel_width;
                        let ball_start = ball_row * ball.kernel_width;
                        let ball_end = ball_start + ball.kernel_width;
                        for (b, k) in zip(
                            &mut background.data[background_start..background_end],
                            &ball.kernel[ball_start..ball_end],
                        ) {
                            let z_min = z + k;
                            if *b < z_min {
                                *b = z_min;
                            }
                        }
                    }
                }
            }
            RollingBallUpsample::with_symmetric_cropping(
                ball.downsample_factor,
                image.height,
                image.width,
                radius as u32,
            )
            .interpolate(&background.to_view())
        }

        fn estimate_background_large_kernel_avx2(
            ball: &RollingBall<f32>,
            image: &ImageView<f32>,
        ) -> Image<f32> {
            let radius = (ball.kernel_width / 2) as i32;
            let downsampled = RollingBallDownsample::with_symmetric_padding(
                ball.downsample_factor,
                radius as u32,
            )
            .interpolate(image);
            let mut background = Image::new(downsampled.height, downsampled.width, f32::MIN);
            let height = background.height as i32;
            let width = background.width as i32;
            let zero_splatted = f32x8::splat(0.0);
            let mask_true = mask32x8::splat(true);
            for y in radius..(height - radius) {
                let y_start = (y - radius) as usize;
                let y_end = (y + radius) as usize;
                for x in radius..(width - radius) {
                    let x_start = (x - radius) as usize;
                    let mut z = f32::MAX;
                    for (image_row, ball_row) in
                        zip(y_start as usize..=y_end as usize, 0..ball.kernel_width)
                    {
                        let image_start = image_row * downsampled.width as usize + x_start;
                        let ball_start = ball_row * ball.kernel_width;
                        let num_blocks = ball.kernel_width / 8;
                        let blocks_end = num_blocks * 8;
                        unsafe {
                            for block in 0..num_blocks {
                                let i = f32x8::load_select_unchecked(
                                    &downsampled.data
                                        [image_start + block * 8..image_start + block * 8 + 8],
                                    mask_true,
                                    zero_splatted,
                                );
                                let k = f32x8::load_select_unchecked(
                                    &ball.kernel
                                        [ball_start + block * 8..ball_start + block * 8 + 8],
                                    mask_true,
                                    zero_splatted,
                                );
                                let z_reduced = (i - k).reduce_min();
                                if z > z_reduced {
                                    z = z_reduced;
                                }
                            }
                            for remainder in 0..ball.kernel_width - blocks_end {
                                let i = downsampled.data[image_start + blocks_end + remainder];
                                let k = ball.kernel[ball_start + blocks_end + remainder];
                                let z_reduced = i - k;
                                if z > z_reduced {
                                    z = z_reduced;
                                }
                            }
                        }
                    }
                    let z_splatted = f32x8::splat(z);
                    for (background_row, ball_row) in
                        zip(y_start as usize..=y_end as usize, 0..ball.kernel_width)
                    {
                        let background_start = background_row * background.width as usize + x_start;
                        let ball_start = ball_row * ball.kernel_width;
                        let num_blocks = ball.kernel_width / 8;
                        let blocks_end = num_blocks * 8;
                        unsafe {
                            for block in 0..num_blocks {
                                let mut b = f32x8::load_select_unchecked(
                                    &background.data[background_start + block * 8
                                        ..background_start + block * 8 + 8],
                                    mask_true,
                                    zero_splatted,
                                );
                                let k = f32x8::load_select_unchecked(
                                    &ball.kernel
                                        [ball_start + block * 8..ball_start + block * 8 + 8],
                                    mask_true,
                                    zero_splatted,
                                );
                                let z_min = z_splatted + k;
                                b = b.simd_max(z_min);
                                b.store_select_unchecked(
                                    &mut background.data[background_start + block * 8
                                        ..background_start + block * 8 + 8],
                                    mask_true,
                                );
                            }
                            for remainder in 0..ball.kernel_width - blocks_end {
                                let z_min = z + ball.kernel[ball_start + blocks_end + remainder];
                                if background.data[background_start + blocks_end + remainder]
                                    < z_min
                                {
                                    background.data[background_start + blocks_end + remainder] =
                                        z_min;
                                }
                            }
                        }
                    }
                }
            }
            RollingBallUpsample::with_symmetric_cropping(
                ball.downsample_factor,
                image.height,
                image.width,
                radius as u32,
            )
            .interpolate(&background.to_view())
        }

        #[cfg(target_arch = "x86_64")]
        {
            if self.kernel_width / 2 >= 8 && is_x86_feature_detected!("avx2") {
                estimate_background_large_kernel_avx2(self, image)
            } else {
                estimate_background_scalar(self, image)
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            estimate_background_scalar(self, image)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate image;
    extern crate std;
    use std::path::PathBuf;

    #[test]
    fn it_works() {
        let fname = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources/test.tif");
        let image = image::open(fname).expect("Test image file should exist");
        let height = image.height();
        let width = image.width();
        let mut image = Image::from_vec(image.to_luma32f().into_raw(), height, width);
        for p in image.data.iter_mut() {
            *p *= 65335.0;
        }
        let rolling_ball = RollingBall::with_downsample_and_arc_trim(50.0, 4, 0.32);
        let _background = rolling_ball.estimate_background(&image.to_view());
    }
}

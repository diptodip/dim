#[pyo3::pymodule]
mod dim {
    use dim::RemoveBackground;
    use numpy::ndarray::{Array2, Array3};
    use numpy::{
        IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods,
    };
    use pyo3::prelude::*;

    #[pyclass]
    struct RollingBall {
        _rolling_ball: dim::RollingBall<f32>,
    }

    #[pymethods]
    impl RollingBall {
        #[new]
        fn new(radius: f32, downsample_factor: u32, arc_trim_percentage: f32) -> Self {
            RollingBall {
                _rolling_ball: dim::RollingBall::<f32>::with_downsample_and_arc_trim(
                    radius,
                    downsample_factor,
                    arc_trim_percentage,
                ),
            }
        }

        #[getter]
        fn get_radius(&self) -> PyResult<f32> {
            Ok(self._rolling_ball.radius)
        }

        #[getter]
        fn get_downsample_factor(&self) -> PyResult<u32> {
            Ok(self._rolling_ball.downsample_factor)
        }

        #[getter]
        fn get_kernel_width(&self) -> PyResult<usize> {
            Ok(self._rolling_ball.kernel_width)
        }

        fn estimate_background<'py>(
            &self,
            py: Python<'py>,
            image: PyReadonlyArray2<'py, f32>,
        ) -> Bound<'py, PyArray2<f32>> {
            let shape = image.shape().to_vec();
            let image_view = dim::ImageView::<f32>::new(
                image.as_slice().expect("Image should be available"),
                shape[0] as u32,
                shape[1] as u32,
            );
            let background = Array2::from_shape_vec(
                (shape[0], shape[1]),
                self._rolling_ball.estimate_background(&image_view).data,
            )
            .expect("Computed background should be same shape as image");
            background.into_pyarray(py)
        }

        fn estimate_background_stack<'py>(
            &self,
            py: Python<'py>,
            image_stack: PyReadonlyArray3<'py, f32>,
        ) -> Bound<'py, PyArray3<f32>> {
            let shape = image_stack.shape().to_vec();
            let image_stack = image_stack.as_array();
            let mut image_view_stack = Vec::with_capacity(shape[0]);
            for i in 0..shape[0] {
                image_view_stack.push(dim::ImageView::<f32>::new(
                    &image_stack.as_slice().expect("Image stack should exist")
                        [i * shape[1] * shape[2]..(i + 1) * shape[1] * shape[2]],
                    shape[1] as u32,
                    shape[2] as u32,
                ));
            }
            let background_stack = Array3::from_shape_vec(
                (shape[0], shape[1], shape[2]),
                self._rolling_ball
                    .estimate_background_stack(&image_view_stack),
            )
            .expect("Computed background should be same shape as image");
            background_stack.into_pyarray(py)
        }
    }
}

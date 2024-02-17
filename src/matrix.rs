use std::f32;
use std::ops::{Add, Index, IndexMut, SubAssign, AddAssign};
extern crate rand;
use rand::distributions::{Distribution, Uniform};
use rand::Rng;

pub(crate) struct Vector {
    pub(crate) data: Vec<f32>,
}

impl Vector {
    pub(crate) fn new(size: usize) -> Self {
        Vector { data: vec![0.0; size] }
    }

    pub(crate) fn full(size: usize, value: f32) -> Self {
        Vector { data: vec![value; size] }
    }

    pub(crate) fn clone(&self) -> Vector {
        Vector {
            data: self.data.clone(),
        }
    }

    pub(crate) fn from_vec(data: Vec<f32>) -> Self {
        Vector { data }
    }

    pub(crate) fn from_slice(data: &[f32]) -> Self {
        Vector { data: data.to_vec() }
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    pub(crate) fn print(&self) {
        for &val in &self.data {
            print!("{} ", val);
        }
        println!();
    }

    fn print_size(&self) {
        println!("Size: {}", self.data.len());
    }

    pub(crate) fn mul(&self, c: f32) -> Vector {
        let mut result = Vector::new(self.len());
        for i in 0..self.len() {
            result[i] = self[i] * c;
        }
        result
    }
}

impl std::ops::Index<usize> for Vector {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl std::ops::IndexMut<usize> for Vector {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl std::ops::AddAssign<&Vector> for Vector {
    fn add_assign(&mut self, other: &Vector) {
        assert_eq!(self.len(), other.len(), "Vectors must be of the same length");
        for i in 0..self.len() {
            self[i] += other[i];
        }
    }
}

impl Add for &Vector {
    type Output = Vector;

    fn add(self, other: Self) -> Vector {
        assert_eq!(self.len(), other.len(), "Vectors must be of the same length");
        let mut result = Vector::new(self.len());
        for i in 0..self.len() {
            result[i] = self[i] + other[i];
        }
        result
    }
}

impl SubAssign for Vector {
    fn sub_assign(&mut self, other: Self) {
        assert_eq!(self.len(), other.len(), "Vectors must be of the same length");
        for i in 0..self.len() {
            self[i] -= other[i];
        }
    }
}

pub(crate) struct Matrix {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    pub(crate) fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    pub(crate) fn zeros_like(other: &Self) -> Self {
        Matrix::new(other.rows, other.cols)
    }

    pub(crate) fn set(&mut self, y: usize, x: usize, val: f32) {
        self.data[y * self.cols + x] = val;
    }

    pub(crate) fn rand(rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..rows * cols).map(|_| rng.gen()).collect();
        Matrix { data, rows, cols }
    }

    // Initialize a matrix with values using the Glorot uniform distribution
    pub(crate) fn glorot_uniform(rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let limit = (6.0f32 / (rows as f32 + cols as f32)).sqrt();
        let uniform_dist = Uniform::new(-limit, limit);
        let data: Vec<f32> = (0..rows * cols).map(|_| uniform_dist.sample(&mut rng)).collect();
        Matrix { data, rows, cols }
    }

    // Initialize a matrix with all elements set to a specific value
    fn full(rows: usize, cols: usize, val: f32) -> Self {
        let data = vec![val; rows * cols];
        Matrix { data, rows, cols }
    }

    // Create a new matrix with the same size as another but filled with a specific value
    pub(crate) fn full_like(other: &Matrix, val: f32) -> Self {
        let data = vec![val; other.rows * other.cols];
        Matrix { data, rows: other.rows, cols: other.cols }
    }

    pub(crate) fn from_vector(vector: &Vector) -> Self {
        Matrix {
            data: vector.data.clone(),
            rows: 1,
            cols: vector.data.len(),
        }
    }

    pub(crate) fn from_vectors(vectors: &Vec<Vector>) -> Self {
        let rows = vectors.len();
        let cols = vectors[0].len();
        let mut data = Vec::new();
        for vector in vectors {
            data.extend(vector.data.clone());
        }
        Matrix { data, rows, cols }
    }

    pub(crate) fn row_as_vector(&self, row_index: usize) -> Vector {
        assert!(row_index < self.rows, "Row index out of bounds.");
        let start = row_index * self.cols;
        let end = start + self.cols;
        Vector {
            data: self.data[start..end].to_vec(),
        }
    }

    pub(crate) fn transpose(&self) -> Matrix {
        let mut transposed = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                transposed.data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        transposed
    }

    pub(crate) fn clone(&self) -> Matrix {
        Matrix {
            data: self.data.clone(),
            rows: self.rows,
            cols: self.cols,
        }
    }

    pub(crate) fn print(&self) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                print!("{} ", self.data[row * self.cols + col]);
            }
            println!();
        }
    }

    fn print_shape(&self) {
        println!("Shape: {}, {}", self.rows, self.cols);
    }

    pub(crate) fn matmul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "Incompatible matrix dimensions");
        let mut result = Matrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self[(i, k)] * other[(k, j)];
                }
                result[(i, j)] = sum;
            }
        }
        result
    }

    // Constant multiple
    pub(crate) fn mul(&self, c: f32) -> Matrix {
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] * c;
        }
        result
    }

    pub(crate) fn elementwise_mul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] * other.data[i];
        }
        result
    }

    pub(crate) fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] + other.data[i];
        }
        result
    }

    // ReLU activation function
    pub(crate) fn relu(&mut self) {
        for i in 0..self.data.len() {
            self.data[i] = self.data[i].max(0.0);
        }

    }

    pub(crate) fn tanh(&mut self) {
        for i in 0..self.data.len() {
            self.data[i] = self.data[i].tanh();
        }
    }

    pub(crate) fn sigmoid(&mut self) {
        for i in 0..self.data.len() {
            self.data[i] = 1.0 / (1.0 + f32::exp(-self.data[i]));
        }
    }

    // Softmax function
    pub(crate) fn softmax(&mut self) {
        for i in 0..self.rows {
            let row_start = i * self.cols;
            let row = &mut self.data[row_start..row_start + self.cols];
            let max_val = *row.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            let exp_sum: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
            for val in row.iter_mut() {
                *val = (*val - max_val).exp() / exp_sum;
            }
        }
    }

    pub(crate) fn softmax_bwd(&self, out_derivative: &Matrix) -> Matrix {
        assert_eq!(self.rows, out_derivative.rows);
        assert_eq!(self.cols, out_derivative.cols);

        let mut c = Matrix::zeros_like(self);
        for i in 0..self.rows {
            for j in 0..self.cols {
                let mut sum = out_derivative[(i, j)];
                for k in 0..self.cols {
                    sum -= self[(i, k)] * out_derivative[(i, k)];
                }
                c[(i, j)] = sum * self[(i, j)];
            }
        }
        c
    }

    // Apply ReLU derivative mask
    pub(crate) fn relu_der_mask(&mut self, activations: &Matrix) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                if activations[(i, j)] <= 0.0 {
                    self[(i, j)] = 0.0;
                }
            }
        }
    }

    pub(crate) fn apply<F>(&mut self, f: F) -> &mut Self
        where
            F: Fn(f32) -> f32,
    {
        for x in &mut self.data {
            *x = f(*x);
        }
        self
    }

    pub(crate) fn elementwise_sub(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] - other.data[i];
        }
        result
    }

    pub(crate) fn add_vec(&self, other: &Vector) -> Matrix {
        assert_eq!(self.cols, other.len());
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[(i, j)] = self[(i, j)] + other[j];
            }
        }
        result
    }

    pub(crate) fn sum_batch(&self) -> Vector {
        let mut result = Vector::new(self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[j] += self[(i, j)];
            }
        }
        result
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f32;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        &self.data[row * self.cols + col]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (row, col) = index;
        &mut self.data[row * self.cols + col]
    }
}

impl SubAssign for Matrix {
    fn sub_assign(&mut self, other: Self) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for i in 0..self.data.len() {
            self.data[i] -= other.data[i];
        }
    }
}

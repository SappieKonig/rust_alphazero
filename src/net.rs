use crate::matrix::{Matrix, Vector};


pub(crate) struct Output {
    pub(crate) policy: Vector,
    pub(crate) value: Vector,
}

impl Output {
    fn new(policy: Vector, value: Vector) -> Self {
        Output { policy, value }
    }
}

pub(crate) struct Net {
    w1: Matrix,
    b1: Vector,
    w2: Matrix,
    b2: Vector,
    w_p: Matrix,
    b_p: Vector,
    w_v: Matrix,
    b_v: Vector,
}

impl Net {
    pub(crate) fn new() -> Self {
        Net {
            w1: Matrix::glorot_uniform(18, 32),
            b1: Vector::new(32),
            w2: Matrix::glorot_uniform(32, 32),
            b2: Vector::new(32),
            w_p: Matrix::glorot_uniform(32, 9),
            b_p: Vector::new(9),
            w_v: Matrix::glorot_uniform(32, 2),
            b_v: Vector::new(2),
        }
    }

    pub(crate) fn forward(&self, input: &Vector) -> Output {
        let input_matrix = Matrix::from_vector(input);
        let mut x1 = input_matrix.matmul(&self.w1).add_vec(&self.b1);
        x1.relu();
        let mut x2 = x1.matmul(&self.w2).add_vec(&self.b2);
        x2.relu();
        let mut policy_matrix = x2.matmul(&self.w_p).add_vec(&self.b_p);
        policy_matrix.sigmoid();
        let mut value_matrix = x2.matmul(&self.w_v).add_vec(&self.b_v);
        value_matrix.softmax();

        Output {
            policy: policy_matrix.row_as_vector(0),
            value: value_matrix.row_as_vector(0),
        }
    }

    pub(crate) fn backward(&mut self, input: &Matrix, target_value: &Matrix, target_policy: &Matrix, moves_played: &Vec<usize>, lr: f32) {
        let mut x1 = input.matmul(&self.w1).add_vec(&self.b1);
        x1.relu();
        let mut x2 = x1.matmul(&self.w2).add_vec(&self.b2);
        x2.relu();
        let mut p = x2.matmul(&self.w_p).add_vec(&self.b_p);
        p.sigmoid();
        let mut v = x2.matmul(&self.w_v).add_vec(&self.b_v);
        v.softmax();

        // Backward pass
        // Derivative of tanh: 1 - tanh^2
        let mut p_clone = p.clone();
        let dsigmoid_dp = p_clone.elementwise_mul(&Matrix::full_like(&p_clone, 1.).elementwise_sub(&p_clone));
        let dbp = dsigmoid_dp.sum_batch();
        let dp_loss = p.elementwise_sub(target_policy).mul(2.).elementwise_mul(&dsigmoid_dp); // Element-wise multiplication
        // mask out moves that were not played
        let mut mask = Matrix::zeros_like(&dp_loss);
        for (i, n) in moves_played.iter().enumerate() {
            mask.set(i, n.clone(), 1.);
        }
        let dWp = x2.transpose().matmul(&dp_loss); // Gradient of W_p

        // Derivative of softmax applied in softmax_bwd (assuming it's implemented correctly)
        let dv_dl = v.elementwise_sub(target_value).mul(2.);
        let dsoft_dl = v.softmax_bwd(&dv_dl); // Applying softmax backward directly
        let dWv = x2.transpose().matmul(&dsoft_dl); // Gradient of W_v

        // Propagate through ReLU and second linear layer
        let mut dx2 = dp_loss.matmul(&self.w_p.transpose()).add(&dsoft_dl.matmul(&self.w_v.transpose()));
        dx2.relu_der_mask(&x2);
        let dW2 = x1.transpose().matmul(&dx2); // Gradient of W2

        // Propagate through first ReLU and linear layer
        let mut dx1 = dx2.matmul(&self.w2.transpose());
        dx1.relu_der_mask(&x1); // Apply ReLU derivative conditionally
        let dW1 = input.transpose().matmul(&dx1); // Gradient of W1

        // Update weights with gradients
        self.w1 -= dW1.mul(lr);
        self.w2 -= dW2.mul(lr);
        self.w_p -= dWp.mul(lr);
        self.b_p -= dbp.mul(lr);
        self.w_v -= dWv.mul(lr);
    }
}

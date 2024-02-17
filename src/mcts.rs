use crate::game::TicTacToe;
use std::cell::RefCell;
use std::rc::Rc;
use crate::matrix::Vector;
use crate::net::Net;


struct Node {
    state: TicTacToe,
    policy: Vec<f32>,
    value: Vector,
    visits: i32,
    legal_moves: Vec<usize>,
    children: Vec<Option<Rc<RefCell<Node>>>>,
}

impl Node {
    fn new(state: TicTacToe, policy: Vec<f32>, value: Vector) -> Self {
        let legal_moves = state.get_legal_moves();
        let children = vec![None; legal_moves.len()];
        Node {
            state,
            policy,
            value,
            visits: 1,
            legal_moves,
            children,
        }
    }

    fn from_net(state: TicTacToe, net: &Net) -> Self {
        let output = net.forward(&state.get_nn_input());
        let legal_moves = state.get_legal_moves();
        let mut policy = vec![0.0; legal_moves.len()];
        for (i, &m) in legal_moves.iter().enumerate() {
            policy[i] = output.policy[m];
        }
        let children = vec![None; legal_moves.len()];
        Node {
            state,
            policy,
            value: output.value,
            visits: 1,
            legal_moves,
            children,
        }
    }

    fn select(&self) -> (Option<Rc<RefCell<Node>>>, usize) {
        let constant: f32 = 0.1;
        let mut max_uct = f32::NEG_INFINITY;
        let mut move_index = 0;
        let mut best_child = None;
        for (i, (child, p)) in self.children.iter().zip(self.policy.iter()).enumerate() {
            let mut score: f32 = 0.0;
            if let Some(c) = child {
                score = p + c.borrow().value[(self.state.turn - 1) as usize] + constant * (self.visits as f32).sqrt();
                score /= (c.borrow().visits + 1) as f32;
            }
            else {
                score = p + constant * (self.visits as f32).sqrt();
            }
            if score > max_uct {
                max_uct = score;
                best_child = child.clone();
                move_index = i;
            }
        }
        (best_child, move_index)
    }

    fn get_move_probs(&self) -> Vec<f32> {
        let mut sum_visits: i32 = 0;
        for c in &self.children {
            if let Some(ref child) = c {
                sum_visits += child.borrow().visits;
            }
        }
        self.children.iter().map(|c| {
            if let Some(ref child) = c {
                child.borrow().visits as f32 / sum_visits as f32
            }
            else {
                0.0
            }
        }).collect()
    }
}


pub(crate) struct MCTS<'a> {
    root: Rc<RefCell<Node>>,
    net: &'a Net,
}

impl<'a> MCTS<'a> {
    pub(crate) fn new(net: &'a Net, state: TicTacToe) -> Self {
        let root = Rc::new(RefCell::new(Node::from_net(state, &net)));
        MCTS { root, net }
    }

    pub(crate) fn move_root(&mut self, action: usize) {
        let new_root: Rc<RefCell<Node>>;
        if let Some(ref child) = self.root.borrow().children[action] {
            new_root = child.clone();
        }
        else {
            let mut new_state = self.root.borrow().state.clone();
            new_state.move_piece(self.root.borrow().legal_moves[action]);
            new_root = Rc::new(RefCell::new(Node::from_net(new_state, &self.net)));
        }
        self.root = new_root;
    }

    pub(crate) fn simulate(&self, n_simulations: i32) -> Vec<f32> {
        for _ in 0..n_simulations {
            let path = self.search();
            let leaf = path.last().unwrap().clone();
            if leaf.borrow().state.is_over() {
                self.backup(path, leaf.borrow().state.get_reward());
            }
            else {
                let value = self.expand(leaf.clone());
                self.backup(path, value);
            }
        }
        self.root.borrow().get_move_probs()
    }

    fn expand(&self, node: Rc<RefCell<Node>>) -> Vector {
        let move_index: usize;
        (_, move_index) = node.borrow().select();
        let mut new_state = node.borrow().state.clone();
        let best_move = node.borrow().legal_moves[move_index];
        new_state.move_piece(best_move);
        let new_node = Rc::new(RefCell::new(Node::from_net(new_state, &self.net)));
        node.borrow_mut().children[move_index] = Some(new_node.clone());
        let x = new_node.borrow().value.clone(); x
    }

    fn backup(&self, path: Vec<Rc<RefCell<Node>>>, value: Vector) {
        for node in path {
            node.borrow_mut().visits += 1;
            node.borrow_mut().value += &value;
        }
    }

    fn search(&self) -> Vec<Rc<RefCell<Node>>> {
        let mut current = self.root.clone();
        let mut path = vec![current.clone()];

        loop {
            let (child, _) = current.borrow().select();
            if let Some(ref c) = child {
                if c.borrow().state.is_over() {
                    break;
                }
                path.push(c.clone());
            }
            else {
                break;
            }
            current = child.unwrap();
        }

        path
    }


}
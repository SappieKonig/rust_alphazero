mod mcts;
mod matrix;
mod game;
mod net;

use matrix::{Matrix, Vector};
use net::Net;
use game::TicTacToe;
use mcts::MCTS;
use std::time::Instant;
use rand::Rng;


struct IncompleteRecord {
    state_nn_input: Vector,
    move_played: usize,
    player: i8,
}


struct Record {
    state_nn_input: Vector,
    result: Vector,
    move_played: usize,
    policy_target: f32,
}

impl Record {
    fn new(incomplete_record: IncompleteRecord, result: Vector) -> Self {
        let policy_target = result[(incomplete_record.player - 1) as usize];
        Record {
            state_nn_input: incomplete_record.state_nn_input,
            result,
            move_played: incomplete_record.move_played,
            policy_target,
        }
    }
}


fn select_with_temp(move_probs: Vec<f32>, t: f32) -> usize {
    let inv_t = 1.0 / t;
    let mut sum = 0.0;
    for p in &move_probs {
        sum += p.powf(inv_t);
    }
    let mut cdf = 0.0;
    let r = rand::random::<f32>();
    for (i, p) in move_probs.iter().enumerate() {
        cdf += p.powf(inv_t) / sum;
        if cdf > r {
            return i;
        }
    }
    move_probs.len() - 1
}


fn select_max(move_probs: Vec<f32>) -> usize {
    let mut max: f32 = 0.;
    let mut best_move = 0;
    for (i, p) in move_probs.iter().enumerate() {
        if p.clone() > max {
            max = p.clone();
            best_move = i;
        }
    }
    best_move
}


fn play_game(net: &Net, t: f32) -> Vec<Record> {
    let mut incomplete_data: Vec<IncompleteRecord> = Vec::new();
    let mut state = TicTacToe::new();
    let mut mcts = MCTS::new(net, state.clone());
    while !state.is_over() {
        let move_index = select_with_temp(mcts.simulate(100), t);
        let _move = state.get_legal_moves()[move_index];
        incomplete_data.push(IncompleteRecord {
            state_nn_input: state.get_nn_input(),
            move_played: _move,
            player: state.turn,
        });
        state.move_piece(_move);
        mcts.move_root(move_index);
    }
    let mut data: Vec<Record> = Vec::new();
    let result = state.get_reward();
    for record in incomplete_data {
        data.push(Record::new(record, result.clone()));
    }
    data
}


fn play_games(net: &Net, t: f32, n_games: usize) -> Vec<Record> {
    let mut all_data: Vec<Record> = Vec::new();
    for _ in 0..n_games {
        let data = play_game(net, t);
        all_data.extend(data);
    }
    all_data
}


fn shuffle_data(data: &mut Vec<Record>) {
    for i in (1..data.len()).rev() {
        let j = rand::random::<usize>() % (i + 1);
        data.swap(i, j);
    }
}


fn batch(data: &Vec<Record>, start: usize, stop: usize) -> (Matrix, Matrix, Matrix, Vec<usize>) {
    let mut inp: Vec<Vector> = Vec::new();
    let mut res: Vec<Vector> = Vec::new();
    let mut target_p: Vec<Vector> = Vec::new();
    let mut moves_played: Vec<usize> = Vec::new();
    for i in start..stop {
        inp.push(data[i].state_nn_input.clone());
        res.push(data[i].result.clone());
        target_p.push(Vector::full(9, data[i].policy_target));
        moves_played.push(data[i].move_played);
    }
    let input = Matrix::from_vectors(&inp);
    let target_value = Matrix::from_vectors(&res);
    let target_policy = Matrix::from_vectors(&target_p);
    (input, target_value, target_policy, moves_played)
}

fn against_random(net: &Net, n_games: usize) -> Vec<usize> {
    let mut result = vec![0; 3];
    for i in 0..n_games {
        let mut state = TicTacToe::new();
        let mut mcts = MCTS::new(net, state.clone());
        while !state.is_over() {
            let mut _move;
            let mut move_index;
            if ((state.get_turn() - 1) % 2) as usize == (i % 2) {
                move_index = select_max(mcts.simulate(100));
                _move = state.get_legal_moves()[move_index];
            } else {
                let legal_moves = state.get_legal_moves();
                let mut rng = rand::thread_rng();
                move_index = rng.gen_range(0..legal_moves.len());
                _move = legal_moves[move_index]
            }
            state.move_piece(_move);
            mcts.move_root(move_index);
        }
        let reward = state.get_reward()[i % 2];
        if reward == 1. {
            result[0] += 1
        } else if reward == 0. {
            result[2] += 1
        } else {
            result[1] += 1
        }
    }
    result
}


fn train_net(net: &mut Net, data: &mut Vec<Record>, lr: f32) {
    shuffle_data(data);
    let batch_size = 32;
    let n_batches = data.len() / batch_size;
    for i in 0..n_batches {
        let (input, target_value, target_policy, moves_played) = batch(data, i * batch_size, (i + 1) * batch_size);
        net.backward(&input, &target_value, &target_policy, &moves_played, lr);
    }
}


fn main() {
    let start = Instant::now();
    let mut net = Net::new();
    let n_epochs = 100;
    for _ in 0..n_epochs {
        let out = net.forward(&TicTacToe::new().get_nn_input());
        out.policy.print();
        for n in against_random(&net, 100) {
            print!("{} ", n);
        }
        println!();
        let mut data = play_games(&net, 1.0, 100);
        train_net(&mut net, &mut data, 0.01);
    }
    let duration = start.elapsed();
    println!("Time elapsed in expensive_function() is: {:?}", duration);
    // println!("{:?}", data.len());
    // play_game(Net::new(), 1.0);
    // let state = TicTacToe::new();
    // let net = Net::new();
    // let mcts = MCTS::new(net, state.clone());
    // print!("{:?}", mcts.simulate(100))
}
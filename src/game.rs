use crate::matrix::Vector;


pub(crate) struct TicTacToe {
    board: Vec<i8>,
    pub(crate) turn: i8,
}

impl TicTacToe {
    // Constructor for a new game
    pub(crate) fn new() -> Self {
        TicTacToe {
            board: vec![0; 9], // Initialize a 9-element vector with zeroes
            turn: 1,
        }
    }

    pub(crate) fn clone(&self) -> Self {
        Self {
            board: self.board.clone(),
            turn: self.turn,
        }
    }

    // Constructor with existing board and turn
    fn with_state(board: Vec<i8>, turn: i8) -> Self {
        Self { board, turn }
    }

    pub(crate) fn get_turn(&self) -> i8 {
        self.turn
    }

    // Display the game board
    pub(crate) fn show(&self) {
        for i in 0..3 {
            for j in 0..3 {
                let square = self.board[i * 3 + j];
                let symbol = match square {
                    0 => ".",
                    1 => "X",
                    2 => "O",
                    _ => " ",
                };
                print!("{} ", symbol);
            }
            println!();
        }
    }

    // Make a move
    pub(crate) fn move_piece(&mut self, move_position: usize) {
        if move_position < self.board.len() {
            self.board[move_position] = self.turn as i8;
            self.turn = 3 - self.turn;
        }
    }

    // Check if the current player has won
    fn has_won(&self) -> bool {
        let check_line = |a, b, c| {
            self.board[a] != 0 && self.board[a] == self.board[b] && self.board[a] == self.board[c]
        };
        for i in 0..3 {
            if check_line(i, i + 3, i + 6) || check_line(i * 3, i * 3 + 1, i * 3 + 2) {
                return true;
            }
        }
        if check_line(0, 4, 8) || check_line(2, 4, 6) {
            return true;
        }
        false
    }

    // Check if the game is over
    pub(crate) fn is_over(&self) -> bool {
        if self.has_won() {
            return true;
        }
        !self.board.contains(&0)
    }

    pub(crate) fn get_legal_moves(&self) -> Vec<usize> {
        self.board
            .iter()
            .enumerate()
            .filter(|(_, &v)| v == 0)
            .map(|(i, _)| i)
            .collect()
    }

    pub(crate) fn get_nn_input(&self) -> Vector {
        let mut input = Vector::new(18);
        for i in 0..9 {
            if self.board[i] == self.turn {
                input.data[i * 2] = 1.0;
            }
            else if self.board[i] == 3 - self.turn {
                input.data[i * 2 + 1] = 1.0;
            }
        }
        input
    }

    pub(crate) fn get_reward(&self) -> Vector {
        if self.has_won() {
            if self.turn == 1 {
                return Vector::from_slice(&[0., 1.0]);
            }
            else {
                return Vector::from_slice(&[1.0, 0.]);
            }
        }
        Vector::from_slice(&[0.5, 0.5])
    }
}


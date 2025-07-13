from cc.core import Board, Move

class HumanPlayer:
    def select_move(self, state: Board, moves: list[Move]) -> tuple[Move, None]:
        '''
        allows a human player to select a move from the available moves.
        '''
        # show the player the board annotated with the available moves
        print(state.visualize_move_ends(moves))

        # print the available moves
        print(f"Available moves: {len(moves)}")
        for i, move in enumerate(moves):
            print(f'{i}: {move}')
        
        # get the user's input
        while True:
            user_input = input("Enter a number: ")
            try:
                move_idx = int(user_input)
                assert 0 <= move_idx < len(moves)
                break
            except ValueError:
                print("Invalid input. Please enter a valid integer.")
        
        # echo the user's choice
        print(f"Player chooses move: {move_idx}")

        # return the selected move
        return moves[move_idx], None

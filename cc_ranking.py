from ChineseCheckers import Game, Board, Move, Point, Player, Tile

import logging
logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.CRITICAL)

def rank(board: Board) -> int:
	'''
	Copied from CCheckers.cpp.
	'''
	r = 0
	l1s, l2s = board.num_pieces()
'''
	for i in range(len(s.board)):
        if l1s + l2s <= 0:
            break
		
        if s.board[i] == Player.PLAYER_O:
            l2s -= 1
        if s.board[i] == Player.PLAYER_X:
            if l2s > 0:
                r = r + multinomial(NUM_SPOTS - i - 1, l1s, l2s - 1)
            l1s -= 1
        else:
            if l2s > 0:
                r = r + multinomial(NUM_SPOTS - i - 1, l1s, l2s - 1)
            if l1s > 0:
                r = r + multinomial(NUM_SPOTS - i - 1, l1s - 1, l2s)
    
	// LSB stores the player to move
	return (r << 1) + s.toMove;
	'''

def sort_list_by_listed_positions(list1, list2):
    '''
    Takes two lists of the same length.
    sorts the second list according to the the first list.
    returns two lists,
    - the first is the sorted indexes of the second list
    - the second is the sorted values of the second list
    '''
    result1 = [None] * len(list1)
    result2 = [None] * len(list2)

    for i, pos in enumerate(list1):
        result1[pos] = i
        result2[pos] = list2[i]
    
    return result1, result2

def generate_rect_board_lists(width, height):
    '''
    Generates the localRectToBoard and localBoardToRect lists from ChineseCheckers.h for a board of a given size.
    '''
    in_order = []
    localRectToBoard = []
    localBoardToRect = []
    board_index = 0

    for d in range(width + height - 1):  # sum of indices (x + y)
        logging.debug(f"Processing diagonal {d}")
        # For each diagonal, we need to find the valid (x, y) pairs
        for x in range(d + 1):
            # Calculate y based on the current x and d
            y = d - x
            if x < width and y < height:
                logging.debug(f"\tMapping rect_index ({x}, {y}) to board_index {board_index}")
                rect_index = y * width + x  # row-major index from grid[x][y]
                localBoardToRect.append(rect_index)
                in_order.append(board_index)
                board_index += 1
            else:
                logging.debug(f"\tSkipping rect_index ({x}, {y}) as it is out of bounds")
    
    localRectToBoard, _ = sort_list_by_listed_positions(localBoardToRect, in_order)

    return localRectToBoard, localBoardToRect

def grid_to_CCState_order(board: list[list[Tile]]) -> list[int]:
    '''
    Converts a grid of tiles to a 1D list of integers, following the format used in CCState.
    '''
    width, height = len(board[0]), len(board)
    ccstate_order = []

    for d in range(width + height - 1):  # sum of indices (x + y)
        logging.debug(f"Processing diagonal {d}")
        # For each diagonal, we need to find the valid (x, y) pairs
        for x in range(d + 1):
            # Calculate y based on the current x and d
            y = d - x
            if x < width and y < height:
                logging.debug(f"\Appending tile ({x}, {y})")
                tile = 0
                if board[y][x] == Tile.PLAYER_X:
                    tile = 1
                elif board[y][x] == Tile.PLAYER_O:
                    tile = 2
                ccstate_order.append(tile)
            else:
                logging.debug(f"\tSkipping tile ({x}, {y}) as it is out of bounds")

    return ccstate_order

def convert_board_to_CCState_old(board: Board) -> tuple[list[int], int]:
    # todo: I could probably do this quicker by just iterating through the diagonals than creating all the lists.
    height = len(board.board)
    width = len(board.board[0])
    flat_board = [board.board[x][y] for y in range(height) for x in range(width)]
    _, board = sort_list_by_listed_positions(generate_rect_board_lists(width, height)[0], flat_board)
    toMove = 0 if board.current_player == Player.PLAYER_X else 1
    return board, toMove

def convert_Board_to_CCState(board: Board) -> tuple[list[int], int]:
    '''
    Converts a Board object to a 1D list of integers, following the format used in CCState.
    '''
    new_board = grid_to_CCState_order(board.board)
    toMove = 0 if board.current_player == Player.PLAYER_X else 1
    return new_board, toMove

def check_CCState(board_size, num_pieces):
    '''
    Checks the CCState for a given board size and number of pieces.
    '''
    game = Game(board_size=board_size, num_pieces=num_pieces)
    board, toMove = convert_Board_to_CCState(game.board)
    print(f"CCState for board size {board_size} and {num_pieces} pieces:")
    print(f"toMove: {toMove}")
    #print(f"CCState: {board}")
    for i in range(len(board)):
        if board[i] == 1:
            print(f"Player X at index {i}")
        elif board[i] == 2:
            print(f"Player O at index {i}")
    print("---")

def check_beginning_ccstates():
    check_CCState(11, 10)
    check_CCState(9, 10)
    check_CCState(9, 8)
    check_CCState(9, 7)
    check_CCState(9, 6)
    check_CCState(9, 5)
    check_CCState(9, 4)
    check_CCState(7, 6)
    check_CCState(7, 5)
    check_CCState(7, 4)
    check_CCState(7, 3)
    check_CCState(7, 2)
    check_CCState(7, 1)
    check_CCState(6, 6)
    check_CCState(6, 4)
    check_CCState(5, 6)
    check_CCState(4, 6)
    check_CCState(4, 4)
    check_CCState(4, 3)
    check_CCState(4, 2)
    check_CCState(3, 1)
    check_CCState(3, 2)
    check_CCState(2, 1)

if __name__ == "__main__":
    check_beginning_ccstates()
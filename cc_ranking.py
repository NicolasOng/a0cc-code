from ChineseCheckers import Game, Board, Move, Point, Player, Tile

import logging
logging.basicConfig(level=logging.DEBUG)
#logging.basicConfig(level=logging.CRITICAL)

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

def convert_board_to_CCState(board: Board) -> tuple[list[int], int]:
    height = len(board.board)
    width = len(board.board[0])
    flat_board = [board.board[x][y] for y in range(height) for x in range(width)]
    _, board = sort_list_by_listed_positions(generate_rect_board_lists(width, height)[0], flat_board)
    toMove = 0 if board.current_player == Player.PLAYER_X else 1
    return board, toMove

if __name__ == "__main__":
    print(generate_rect_board_lists(7, 7))
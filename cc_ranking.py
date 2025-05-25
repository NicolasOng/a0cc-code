import math
from ChineseCheckers import Game, Board, Move, Point, Player, Tile, board_to_home_size

import logging
logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.CRITICAL)

class RankingBase:
    '''
    RankingBase is a class that provides methods for ranking and unranking.
    It is focused on the Ranking methods found in the CCheckers class in C++.
    '''
    def __init__(self, num_spots: int, num_players: int, num_pieces: int):
        self.num_spots = num_spots
        self.num_players = num_players
        self.num_pieces = num_pieces
        self.the_sums = self.init_binomial_sums(num_pieces, num_spots)
        self.binomials = RankingBase.init_binomials(num_spots, num_players, num_pieces)
    
    @staticmethod
    def bi(n: int, k: int) -> int:
        '''
        Equivalent to CCheckers::bi(unsigned int n, unsigned int k) const in C++.
        Computes the binomial coefficient (n choose k) using iterative multiplication.
        '''
        num = 1
        bound = n - k
        while n > bound:
            num *= n
            n -= 1
        den = 1
        while k > 1:
            den *= k
            k -= 1
        return num // den
    
    @staticmethod
    def init_binomials(num_spots: int, num_players: int, num_pieces: int) -> list[int]:
        '''
        Initializes and returns the binomials as a flat list of binomial coefficients.
        Equivalent to CCheckers::initBinomial().
        binomials[x*(num_players*num_pieces+1)+y] = bi(x, y)
        '''
        size = (num_spots + 1) * (num_players * num_pieces + 1)
        binomials = [0] * size
        for x in range(num_spots + 1):
            for y in range(num_players * num_pieces + 1):
                binomials[x * (num_players * num_pieces + 1) + y] = RankingBase.bi(x, y)
        return binomials
    
    def binomial(self, n: int, k: int) -> int:
        '''
        Equivalent to CCheckers::binomial(unsigned int n, unsigned int k) const in C++.
        Returns the precomputed binomial coefficient from the binomials array.
        binomials[n*(1+num_players*num_pieces)+k]
        '''
        idx = n * (1 + self.num_players * self.num_pieces) + k
        return self.binomials[idx]
    
    def init_binomial_sums(self, num_pieces: int, num_spots: int) -> list[int]:
        '''
        Initializes and returns the_sums as a flat list of cumulative binomial coefficients.
        Equivalent to CCheckers::initBinomialSums() for theSums.
        '''
        size = (num_pieces + 1) * (num_spots + 1)
        the_sums = [0] * size
        for x in range(num_pieces + 1):
            result = 0
            for y in range(num_spots + 1):
                result += self.binomial(y, x)
                the_sums[x * (num_spots + 1) + y] = result
        return the_sums
    
    def binomial_sum(self, n1: int, n2: int, k: int) -> int:
        '''
        Equivalent to CCheckers::binomialSum(unsigned int n1, unsigned int n2, unsigned int k) const in C++.
        '''
        idx1 = k * (self.num_spots + 1) + n1
        idx2 = k * (self.num_spots + 1) + n2
        return self.the_sums[idx1] - self.the_sums[idx2]

    @staticmethod
    def multinomial(n: int, k1: int, k2: int):
        '''
        Copied from CCheckers.cpp.
        multinomial(n, k1, k2, k3) = \frac{n!}{k1! k2! k3!},
        where k3 is n - (k1 + k2).
        first calculates n!/k3!, then multiplies by 1/(k2! k3!)
        n & k1 & k2 & k3 > 0. k1 & k2 <= 20. 
        '''
        k3 = n - (k1 + k2)

        num = 1
        for i in range(k3 + 1, n + 1):
            num *= i
        
        table = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600,
            6227020800, 87178291200, 1307674368000, 20922789888000, 355687428096000,
            6402373705728000, 121645100408832000, 2432902008176640000]

        den = table[k1] * table[k2]

        return num // den
    
    @staticmethod
    def get_max_rank(num_spots: int, num_pieces: int) -> int:
        '''
        Returns the maximum rank for a given board size and number of pieces, equivalent to CCheckers::getMaxRank().
        '''
        return 2 * RankingBase.multinomial(num_spots, num_pieces, num_pieces)

    @staticmethod
    def rank(board: list[int], to_move: int, num_pieces: int) -> int:
        '''
        Copied from CCheckers.cpp.
        in CCheckers, NUM_PIECES & NUM_SPOTS are defined globally. Here, they are passed in or calculated.
        '''
        r = 0
        l1s, l2s = num_pieces, num_pieces
        num_spots = len(board)
        for i in range(len(board)):
            if l1s + l2s <= 0:
                break
            
            if board[i] == 2:
                l2s -= 1
            elif board[i] == 1:
                if l2s > 0:
                    r = r + RankingBase.multinomial(num_spots - i - 1, l1s, l2s - 1)
                l1s -= 1
            else:
                if l2s > 0:
                    r = r + RankingBase.multinomial(num_spots - i - 1, l1s, l2s - 1)
                if l1s > 0:
                    r = r + RankingBase.multinomial(num_spots - i - 1, l1s - 1, l2s)
        
        return (r << 1) + to_move
    
    @staticmethod
    def unrank(rank: int, num_spots: int, num_pieces: int) -> tuple[bool, list[int], int]:
        '''
        Copied from CCheckers.cpp.
        in CCheckers.cpp, NUM_SPOTS and NUM_PIECES was defined globally.
        here, they need to be passed in.
        '''
        board = [0] * num_spots

        to_move = rank & 0x1
        rank >>= 1

        l1s, l2s = num_pieces, num_pieces
        for i in range(num_spots):
            if l1s + l2s <= 0:
                break
            
            value1 = RankingBase.multinomial(num_spots - i - 1, l1s - 1, l2s) if l1s > 0 else 0
            value2 = RankingBase.multinomial(num_spots - i - 1, l1s, l2s - 1) if l2s > 0 else 0

            # this block of code guarantees that the element at the ith index gets either 2, 1, 0
            if rank < value2:
                # trying to place too many 2s
                if l2s <= 0: return False, board, to_move
                board[i] = 2
                l2s -= 1
            elif rank < value1 + value2:
                # trying to place too many 1s
                if l1s <= 0: return False, board, to_move
                board[i] = 1
                rank -= value2
                l1s -= 1
            else:
                board[i] = 0
                rank -= value1 + value2

        return True, board, to_move

    def rank_player(self, s, who: int, num_spots: int, num_pieces: int) -> int:
        '''
        Equivalent to CCheckers::rankPlayer(const CCState &s, int who) const in C++.
        s: a CCState-like object with attribute pieces[who][i]
        who: player index (0 or 1)
        num_spots: total number of spots on the board
        num_pieces: number of pieces per player
        '''
        r2 = 0
        last = num_spots - 1
        for x in range(num_pieces):
            idx = num_pieces - 1 - x
            piece_pos = s.pieces[who][idx]
            tmp = self.binomial_sum(last, num_spots - piece_pos - 1, idx)
            r2 += tmp
            last = num_spots - piece_pos - 1 - 1
        return r2

class CCDefaultRank:
    @staticmethod
    def get_max_rank(num_spots: int, num_pieces: int) -> int:
        return RankingBase.get_max_rank(num_spots, num_pieces)
    
    @staticmethod
    def rank(board: list[int], to_move: int, num_pieces: int) -> int:
        return RankingBase.rank(board, to_move, num_pieces)
    
    @staticmethod
    def unrank(rank: int, num_spots: int, num_pieces: int) -> tuple[bool, list[int], int]:
        return RankingBase.unrank(rank, num_spots, num_pieces)

class CCLocalRank12:
    @staticmethod
    def rank(cc, s) -> int:
        '''
        Equivalent to CCLocalRank12::rank(const CCState &s) const in C++.
        cc: an object with methods rankPlayer, rankPlayerRelative, getMaxSinglePlayerRankRelative
        s: a CCState-like object with attribute toMove
        '''
        r0 = cc.rankPlayer(s, 0)
        r1 = cc.rankPlayerRelative(s, 1, 0)
        assert r0 < cc.getMaxSinglePlayerRank()
        assert r1 >= 0
        assert r1 < cc.getMaxSinglePlayerRankRelative()
        return (r0 * cc.getMaxSinglePlayerRankRelative() + r1) * 2 + s.toMove
    
    @staticmethod
    def unrank(cc, r: int, s) -> bool:
        '''
        Equivalent to CCLocalRank12::unrank(int64_t r, CCState &s) const in C++.
        cc: an object with methods unrankPlayer, unrankPlayerRelative, getMaxSinglePlayerRankRelative
        r: the rank to unrank
        s: a CCState-like object to modify (should have attribute toMove)
        '''
        toMove = int(r & 1)
        r >>= 1
        r0 = r // cc.getMaxSinglePlayerRankRelative()
        r1 = r % cc.getMaxSinglePlayerRankRelative()
        cc.unrankPlayer(r0, s, 0)
        cc.unrankPlayerRelative(r1, s, 1, 0)
        s.toMove = toMove
        return True

def rank_board(board: Board) -> int:
    num_pieces, _ = board.num_pieces()
    ccstate_board, to_move = convert_Board_to_CCState(board)
    return RankingBase.rank(ccstate_board, to_move, num_pieces)

def unrank_board(rank: int, num_spots: int, num_pieces: int) -> Board:
    _, board, to_move = RankingBase.unrank(rank, num_spots, num_pieces)
    return convert_CCState_to_Board(board, to_move)

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

def CCState_to_grid_order(board: list[int]) -> list[list[Tile]]:
    '''
    Converts a 1D list of integers to a grid of tiles, following the format used in Board.
    assumes that the board is a square.
    '''
    side_length = math.sqrt(len(board))
    assert side_length.is_integer(), "Board should be a sqaure"
    side_length = int(side_length)

    width, height = side_length, side_length
    grid_order = [[Tile.EMPTY for _ in range(width)] for _ in range(height)]

    i = 0
    for d in range(width + height - 1):  # sum of indices (x + y)
        logging.debug(f"Processing diagonal {d}")
        # For each diagonal, we need to find the valid (x, y) pairs
        for x in range(d + 1):
            # Calculate y based on the current x and d
            y = d - x
            if x < width and y < height:
                logging.debug(f"Setting tile ({x}, {y})")
                if board[i] == 1:
                    grid_order[y][x] = Tile.PLAYER_X
                elif board[i] == 2:
                    grid_order[y][x] = Tile.PLAYER_O
                i += 1
            else:
                logging.debug(f"\tSkipping tile ({x}, {y}) as it is out of bounds")

    return grid_order


def convert_Board_to_CCState(board: Board) -> tuple[list[int], int]:
    '''
    Converts a Board object to a 1D list of integers, following the format used in CCState.
    '''
    new_board = grid_to_CCState_order(board.board)
    toMove = 0 if board.current_player == Player.PLAYER_X else 1
    return new_board, toMove

def convert_CCState_to_Board(board: list[int], to_move: int) -> Board:
    '''
    converts a CCState (int list of the board and int for player's turn) to my Board format.
    assumes that the board is a square.
    assumes that the home_size follows the board_to_home_size dict.
    '''
    grid = CCState_to_grid_order(board)
    board_size = len(grid)
    new_board = Board(board_size, board_to_home_size[board_size])
    new_board.set_board(grid)
    new_board.current_player = Player.PLAYER_X if to_move == 0 else Player.PLAYER_O
    return new_board

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

def check_board_conversion(board_size, num_pieces):
    '''
    checks that the given board (just the initial ones for now)
    can be converted to CCState format and back
    I've checked that CCState conversion works for a few boards,
    so this is really checking CCState -> grid conversion.
    '''
    print(f"board size {board_size} and {num_pieces} pieces")
    game = Game(board_size=board_size, num_pieces=num_pieces)
    board_start = game.board
    board, toMove = convert_Board_to_CCState(board_start)
    board_end = convert_CCState_to_Board(board, toMove)

    assert board_start.board == board_end.board
    assert board_start.current_player == board_end.current_player
    assert board_start.home_size == board_end.home_size

def check_ranking_conversion(board_size, num_pieces):
    game = Game(board_size=board_size, num_pieces=num_pieces)
    start_board = game.board
    ccstate_board, start_to_move = convert_Board_to_CCState(start_board)

    print(start_board.board_view())

    ranking = rank_ccstate(ccstate_board, start_to_move, num_pieces)
    _, end_board, end_to_move = unrank_ccstate(ranking, len(ccstate_board), num_pieces)

    print(f"Board {board_size}, {num_pieces} has rank {ranking}.")

    print(ccstate_board)
    print(end_board)
    print("---")

    assert ccstate_board == end_board
    assert start_to_move == end_to_move

starting_boards = [(11, 10),
                    (9, 10), (9, 8), (9, 7), (9, 6), (9, 5), (9, 4),
                    (7, 6), (7, 5), (7, 4), (7, 3), (7, 2), (7, 1),
                    (6, 6), (6, 4),
                    (5, 6),
                    (4, 6), (4, 4), (4, 3), (4, 2),
                    (3, 1), (3, 2),
                    (2, 1)]

def check_beginning_ccstates():
    for sb in starting_boards:
        check_CCState(sb[0], sb[1])

def check_board_conversions():
    for sb in starting_boards:
        check_board_conversion(sb[0], sb[1])

def check_ranking_conversions():
    for sb in starting_boards:
        try:
            check_ranking_conversion(sb[0], sb[1])
        except:
            print("ranking/unranking error - likely that the board couldn't be reconstructed accurately.")


if __name__ == "__main__":
    check_ranking_conversions()

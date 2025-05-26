import math
from ChineseCheckers import Game, Board, Move, Point, Player, Tile, board_to_home_size

import logging
logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.CRITICAL)

class CCState:
    '''
    A class representing the state of a Chinese Checkers game.
    This is a simplified version for the purpose of ranking and unranking.
    Also has methods for converting between Board and CCState formats.
    Based on the CCState class in CCheckers.h.
    '''
    # 1D list of integers representing the board state
    board: list[int]
    # List of two lists, each containing indices of pieces for Player X and Player O
    pieces: list[list[int]]
    # 0 for Player X, 1 for Player O
    to_move: int

    def __init__(self, num_spots: int, num_pieces: int, num_players: int):
        '''
        Initializes a blank CCState with the given number of spots, pieces, and players.
        '''
        self.board = [0] * num_spots
        self.pieces = [[0 for _ in range(num_pieces)] for _ in range(num_players)]
        self.to_move = 0
    
    @staticmethod
    def grid_to_CCState_order(board: list[list[Tile]]) -> list[int]:
        '''
        Converts a grid of tiles to a 1D list of integers, following the format used in CCState.
        '''
        width, height = len(board[0]), len(board)
        ccstate_order: list[int] = []

        for d in range(width + height - 1):  # sum of indices (x + y)
            logging.debug(f"Processing diagonal {d}")
            # For each diagonal, we need to find the valid (x, y) pairs
            for x in range(d + 1):
                # Calculate y based on the current x and d
                y = d - x
                if x < width and y < height:
                    logging.debug(f"Appending tile ({x}, {y})")
                    tile = 0
                    if board[y][x] == Tile.PLAYER_X:
                        tile = 1
                    elif board[y][x] == Tile.PLAYER_O:
                        tile = 2
                    ccstate_order.append(tile)
                else:
                    logging.debug(f"\tSkipping tile ({x}, {y}) as it is out of bounds")

        return ccstate_order
    
    @staticmethod
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

    @staticmethod
    def convert_Board_to_CCState(board: Board) -> tuple[list[int], int]:
        '''
        Converts a Board object to a 1D list of integers, following the format used in CCState.
        '''
        new_board = CCState.grid_to_CCState_order(board.board)
        toMove = 0 if board.current_player == Player.PLAYER_X else 1
        return new_board, toMove
    
    @staticmethod
    def convert_CCState_to_Board(board: list[int], to_move: int) -> Board:
        '''
        converts a CCState (int list of the board and int for player's turn) to my Board format.
        assumes that the board is a square.
        assumes that the home_size follows the board_to_home_size dict.
        '''
        grid = CCState.CCState_to_grid_order(board)
        board_size = len(grid)
        new_board = Board(board_size, board_to_home_size[board_size])
        new_board.set_board(grid)
        new_board.current_player = Player.PLAYER_X if to_move == 0 else Player.PLAYER_O
        return new_board
    
    def build_pieces_from_board(self) -> None:
        '''
        Given a 1D board list (CCState format), returns a list of two lists:
        - pieces[0]: indices of player 1's pieces (value 1)
        - pieces[1]: indices of player 2's pieces (value 2)
        The indices are sorted into descending order.
        '''
        self.pieces = [[], []]
        for idx, val in enumerate(self.board):
            if val == 1:
                self.pieces[0].append(idx)
            elif val == 2:
                self.pieces[1].append(idx)
        # Sort the pieces in descending order
        self.pieces[0].sort(reverse=True)
        self.pieces[1].sort(reverse=True)
    
    def initialize_from_board(self, board: Board) -> None:
        '''
        Initializes the CCState from a Board.
        This method populates the pieces attribute based on the board state.
        '''
        self.board, self.to_move = CCState.convert_Board_to_CCState(board)
        self.build_pieces_from_board()

    def get_board(self) -> Board:
        '''
        Converts the CCState back to a Board object.
        '''
        return CCState.convert_CCState_to_Board(self.board, self.to_move)
    
    def print_ascii(self) -> None:
        '''
        Prints the CCState in ASCII format.
        Equivalent to void CCState::PrintASCII() const in CCheckers.cpp.
        '''
        print(f"[{self.to_move + 1}] ", end='')
        for x in range(len(self.board)):
            print(f"{self.board[x]} ", end='')
        print()

class RankingBase:
    '''
    RankingBase is a class that provides methods for ranking and unranking.
    It is focused on the Ranking methods found in the CCheckers class in CCheckers.h.
    '''
    def __init__(self, num_spots: int, num_players: int, num_pieces: int):
        self.num_spots = num_spots
        self.num_players = num_players
        self.num_pieces = num_pieces
        self.binomials = RankingBase.init_binomials(num_spots, num_players, num_pieces)
        self.the_sums = self.init_binomial_sums(num_pieces, num_spots)
    
    @staticmethod
    def bi(n: int, k: int) -> int:
        '''
        Equivalent to int64_t CCheckers::bi(unsigned int n, unsigned int k) const in CCheckers.cpp.
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
        Equivalent to void CCheckers::initBinomial() in CCheckers.cpp.
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
        Equivalent to int64_t CCheckers::binomial(unsigned int n, unsigned int k) const in CCheckers.cpp.
        Returns the precomputed binomial coefficient from the binomials array.
        binomials[n*(1+num_players*num_pieces)+k]
        '''
        idx = n * (1 + self.num_players * self.num_pieces) + k
        return self.binomials[idx]
    
    def init_binomial_sums(self, num_pieces: int, num_spots: int) -> list[int]:
        '''
        Initializes and returns the_sums as a flat list of cumulative binomial coefficients.
        Equivalent to void CCheckers::initBinomialSums() in CCheckers.cpp.
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
        Equivalent to int64_t CCheckers::binomialSum(unsigned int n1, unsigned int n2, unsigned int k) const in CCheckers.cpp.
        '''
        idx1 = k * (self.num_spots + 1) + n1
        idx2 = k * (self.num_spots + 1) + n2
        return self.the_sums[idx1] - self.the_sums[idx2]

    @staticmethod
    def multinomial(n: int, k1: int, k2: int):
        '''
        Equivalent to int64_t CCheckers::multinomial(unsigned int n, unsigned int k1, unsigned int k2) const in CCheckers.cpp.
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
    
    def get_max_rank(self) -> int:
        '''
        Equivalent to int64_t CCheckers::getMaxRank() const in CCheckers.cpp.
        '''
        return 2 * RankingBase.multinomial(self.num_spots, self.num_pieces, self.num_pieces)

    def rank(self, s: CCState) -> int:
        '''
        Equivalent to int64_t CCheckers::rank(const CCState &s) in CCheckers.cpp.
        '''
        r = 0
        l1s, l2s = self.num_pieces, self.num_pieces
        num_spots = self.num_spots
        for i in range(num_spots):
            if l1s + l2s <= 0:
                break
            
            if s.board[i] == 2:
                l2s -= 1
            elif s.board[i] == 1:
                if l2s > 0:
                    r = r + RankingBase.multinomial(num_spots - i - 1, l1s, l2s - 1)
                l1s -= 1
            else:
                if l2s > 0:
                    r = r + RankingBase.multinomial(num_spots - i - 1, l1s, l2s - 1)
                if l1s > 0:
                    r = r + RankingBase.multinomial(num_spots - i - 1, l1s - 1, l2s)
        
        return (r << 1) + s.to_move
    
    def unrank(self, rank: int, s: CCState) -> bool:
        '''
        Equivalent to bool CCheckers::unrank(int64_t theRank, CCState &s) const in CCheckers.cpp.
        '''
        # Clear board to zero
        s.board = [0] * self.num_spots
        s.pieces = [[0 for _ in range(self.num_pieces)] for _ in range(self.num_players)]

        # LSB stores player to move
        s.to_move = rank & 0x1
        rank >>= 1

        l1s, l2s = self.num_pieces, self.num_pieces
        i = 0
        while (l1s + l2s) > 0:
            if l2s > 0:
                value2 = RankingBase.multinomial(self.num_spots - i - 1, l1s, l2s - 1)
            else:
                value2 = 0
            if l1s > 0:
                value1 = RankingBase.multinomial(self.num_spots - i - 1, l1s - 1, l2s)
            else:
                value1 = 0
            if rank < value2:
                if l2s <= 0:
                    return False
                s.board[i] = 2
                s.pieces[1][l2s - 1] = i
                l2s -= 1
            elif rank < (value1 + value2):
                if l1s <= 0:
                    return False
                s.board[i] = 1
                s.pieces[0][l1s - 1] = i
                rank -= value2
                l1s -= 1
            else:
                s.board[i] = 0
                rank -= (value1 + value2)
            i += 1
        return True
    
    def get_max_single_player_rank(self) -> int:
        '''
        Equivalent to int64_t CCheckers::getMaxSinglePlayerRank() const in CCheckers.cpp.
        Returns the number of ways to place NUM_PIECES pieces in NUM_SPOTS spots.
        '''
        return self.binomial(self.num_spots, self.num_pieces)
    
    def get_max_single_player_rank_relative(self) -> int:
        '''
        Equivalent to int64_t CCheckers::getMaxSinglePlayerRankRelative() const in CCheckers.cpp.
        Returns the number of ways to place NUM_PIECES pieces in NUM_SPOTS-NUM_PIECES spots.
        '''
        return self.binomial(self.num_spots - self.num_pieces, self.num_pieces)
    
    def rank_player(self, s: CCState, who: int) -> int:
        '''
        Equivalent to int64_t CCheckers::rankPlayer(const CCState &s, int who) const in CCheckers.cpp.
        '''
        r2 = 0
        last = self.num_spots - 1
        for x in range(self.num_pieces):
            idx = self.num_pieces - 1 - x
            piece_pos = s.pieces[who][idx]
            tmp = self.binomial_sum(last, self.num_spots - piece_pos - 1, idx)
            r2 += tmp
            last = self.num_spots - piece_pos - 1 - 1
        return r2

    def rank_player_relative(self, s: CCState, who: int, relative: int) -> int:
        '''
        Equivalent to int64_t CCheckers::rankPlayerRelative(const CCState &s, int who, int relative) const in CCheckers.cpp.
        This method ranks the pieces of player 'who' relative to player 'relative'.
        '''
        mod = [0] * self.num_pieces
        relPos = 0
        myPos = 0
        offset = 0
        while myPos < self.num_pieces:
            if relPos < self.num_pieces and s.pieces[relative][self.num_pieces - 1 - relPos] < s.pieces[who][self.num_pieces - 1 - myPos]:
                relPos += 1
                offset += 1
            else:
                mod[self.num_pieces - 1 - myPos] = s.pieces[who][self.num_pieces - 1 - myPos] - offset
                myPos += 1
        r2 = 0
        last = self.num_spots - 1 - self.num_pieces
        for x in range(self.num_pieces):
            idx = self.num_pieces - 1 - x
            tmp = self.binomial_sum(last, self.num_spots - mod[idx] - 1 - self.num_pieces, idx)
            r2 += tmp
            last = self.num_spots - mod[idx] - 1 - self.num_pieces - 1
        assert r2 >= 0
        assert r2 < self.get_max_single_player_rank_relative()
        return r2

    def unrank_player(self, the_rank: int, s: CCState, who: int) -> bool:
        '''
        Equivalent to bool CCheckers::unrankPlayer(int64_t theRank, CCState &s, int who) const in CCheckers.cpp.
        Unranks a single player's pieces from the given rank and updates the board and pieces for that player.
        '''
        tag = who + 1
        ls = self.num_pieces
        s.board = [0] * self.num_spots
        # Ensure s.pieces is a list of lists of correct size
        if not hasattr(s, 'pieces') or len(s.pieces) != self.num_players or any(len(p) != self.num_pieces for p in s.pieces):
            s.pieces = [[0 for _ in range(self.num_pieces)] for _ in range(self.num_players)]
        for i in range(self.num_spots):
            if ls == 0:
                break
            value = self.binomial(self.num_spots - i - 1, ls - 1) if ls > 0 else 0
            if the_rank < value:
                s.board[i] = tag
                s.pieces[who][ls - 1] = i
                ls -= 1
            else:
                s.board[i] = 0
                the_rank -= value
        for x in range(1, self.num_pieces):
            assert s.pieces[who][x - 1] > s.pieces[who][x]
        s.to_move = who
        return True
    
    def unrank_player_relative_helper(self, the_rank: int, s: CCState, who: int) -> None:
        '''
        Equivalent to void CCheckers::unrankPlayerRelativeHelper(int64_t theRank, CCState &s, int who) const in CCheckers.cpp.
        Updates s.pieces[who] in-place based on the rank.
        '''
        ls = self.num_pieces
        i = 0
        while ls > 0:
            if ls > 0:
                value = self.binomial(self.num_spots - i - 1 - self.num_pieces, ls - 1)
            else:
                value = 0
            if the_rank < value:
                s.pieces[who][ls - 1] = i
                ls -= 1
            else:
                the_rank -= value
            i += 1
        for x in range(1, self.num_pieces):
            assert s.pieces[who][x - 1] > s.pieces[who][x]

    def unrank_player_relative(self, r: int, s: 'CCState', who: int, relative: int) -> bool:
        '''
        Equivalent to bool CCheckers::unrankPlayerRelative(int64_t r, CCState &s, int who, int relative) const in CCCheckers.cpp.
        Puts pieces in relative location inside s, then converts to absolute location, and fills board.
        '''
        # Step 1: Put pieces in relative location
        self.unrank_player_relative_helper(r, s, who)

        # Step 2: Put into absolute location
        relPos = 0
        myPos = 0
        offset = self.num_pieces
        while myPos < self.num_pieces:
            if (relPos < self.num_pieces) and (s.pieces[relative][relPos] >= s.pieces[who][myPos] + offset):
                relPos += 1
                offset -= 1
            else:
                s.pieces[who][myPos] += offset
                myPos += 1

        # Step 3: Fill values into array
        for x in range(self.num_pieces):
            s.board[s.pieces[who][x]] = 1 + who
        return True

class CCDefaultRank:
    '''
    Equivalent to CCDefaultRank in CCRankings.h.
    '''
    def __init__(self, num_spots: int, num_players: int, num_pieces: int):
        self.rb = RankingBase(num_spots, num_players, num_pieces)
    
    def get_max_rank(self) -> int:
        return self.rb.get_max_rank()
    
    def rank(self, s: CCState) -> int:
        return self.rb.rank(s)
    
    def unrank(self, rank: int, s: CCState) -> bool:
        return self.rb.unrank(rank, s)

class CCLocalRank12:
    '''
    Equivalent to CCLocalRank12 in CCRankings.h.
    '''
    def __init__(self, num_spots: int, num_players: int, num_pieces: int):
        self.rb = RankingBase(num_spots, num_players, num_pieces)
    
    def get_max_rank(self) -> int:
        return self.rb.get_max_rank()
    
    def rank(self, s: CCState) -> int:
        '''
        Equivalent to int64_t CCLocalRank12::rank(const CCState &s) const in CCRankings.cpp.
        '''
        r0 = self.rb.rank_player(s, 0)
        r1 = self.rb.rank_player_relative(s, 1, 0)
        assert r0 < self.rb.get_max_single_player_rank()
        assert r1 >= 0
        assert r1 < self.rb.get_max_single_player_rank_relative()
        return (r0 * self.rb.get_max_single_player_rank_relative() + r1) * 2 + s.to_move
    
    def unrank(self, r: int, s: CCState) -> bool:
        '''
        Equivalent to CCLocalRank12::unrank(int64_t r, CCState &s) const in CCRankings.cpp.
        '''
        toMove = int(r & 1)
        r >>= 1
        r0 = r // self.rb.get_max_single_player_rank_relative()
        r1 = r % self.rb.get_max_single_player_rank_relative()
        self.rb.unrank_player(r0, s, 0)
        self.rb.unrank_player_relative(r1, s, 1, 0)
        s.to_move = toMove
        return True

def rank_board(board: Board) -> int:
    num_pieces, _ = board.num_pieces()
    num_spots = board.num_spots()
    r = CCDefaultRank(num_spots, 2, num_pieces)
    s = CCState(num_spots, num_pieces, 2)
    s.initialize_from_board(board)
    return r.rank(s)

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

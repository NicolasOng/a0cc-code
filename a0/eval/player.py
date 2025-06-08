from a0.game import PlayerClass, play
from cc.core import Game, Player

def player_evaluation(board_size: int, num_pieces: int, player1: PlayerClass, player2: PlayerClass, num_games: int):
    '''
    Evaluate two players by playing a series of games between them.
    Returns the number of wins, losses, and draws for each player.
    TODO: run games in parallel
    '''
    # Initialize counters for wins, losses, and draws (for player 1)
    wins = 0
    losses = 0
    draws = 0
    for game_num in range(num_games):
        # play a game
        print(f"Game {game_num + 1}/{num_games}")
        game_data = play(Game(board_size, num_pieces, True), [player1, player2], turn_limit=100)

        # get the winner of the game
        if game_data['winner'] == Player.PLAYER_X:
            wins += 1
        elif game_data['winner'] == Player.PLAYER_O:
            losses += 1
        else:
            draws += 1
    
    return wins, losses, draws

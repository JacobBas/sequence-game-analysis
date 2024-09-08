"""
This code creates a simulation of playing Sequence so that analysis can be done on the
probibilities of winning under different circustances.

As of right now the code is just implemented to play the game at random, but more complexity
and "intellegence" will be added to better understand the optimal playing strategy.

The next things that should be done is creating offensive and defensive playing strategies
based on a weighting / scoring of each cell given the overall board.
"""

import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

Coordinate = tuple[int, int]
# fmt: off
Card = Literal[
    "A♠️", "2♠️", "3♠️", "4♠️", "5♠️", "6♠️", "7♠️", "8♠️", "9♠️", "10♠️", "J♠️", "Q♠️", "K♠️",
    "A♣️", "2♣️", "3♣️", "4♣️", "5♣️", "6♣️", "7♣️", "8♣️", "9♣️", "10♣️", "J♣️", "Q♣️", "K♣️",
    "A♦️", "2♦️", "3♦️", "4♦️", "5♦️", "6♦️", "7♦️", "8♦️", "9♦️", "10♦️", "J♦️", "Q♦️", "K♦️",
    "A❤️", "2❤️", "3❤️", "4❤️", "5❤️", "6❤️", "7❤️", "8❤️", "9❤️", "10❤️", "J❤️", "Q❤️", "K❤️",
    "***"

]
# fmt: on
Chip = Literal["🔵", "🟢", "🔴", " ", "<🟡>"]
NumberOfPlayers = Literal[2, 3, 4, 6, 9]

# fmt: off
FULL_DECK: list[Card] = [
    "A♠️", "2♠️", "3♠️", "4♠️", "5♠️", "6♠️", "7♠️", "8♠️", "9♠️", "10♠️", "J♠️", "Q♠️", "K♠️",
    "A♣️", "2♣️", "3♣️", "4♣️", "5♣️", "6♣️", "7♣️", "8♣️", "9♣️", "10♣️", "J♣️", "Q♣️", "K♣️",
    "A♦️", "2♦️", "3♦️", "4♦️", "5♦️", "6♦️", "7♦️", "8♦️", "9♦️", "10♦️", "J♦️", "Q♦️", "K♦️",
    "A❤️", "2❤️", "3❤️", "4❤️", "5❤️", "6❤️", "7❤️", "8❤️", "9❤️", "10❤️", "J❤️", "Q❤️", "K❤️",
]
BLANK_SPACE = "***"
WINNING_CHIP = "<🟡>"
# fmt: on
TWO_EYED_JACKS: list[Card] = ["J♦️", "J♣️"]
ONE_EYED_JACKS: list[Card] = ["J❤️", "J♠️"]


@dataclass
class SequenceBoardCell:
    coordinate: Coordinate
    card: Card
    chip: Chip
    playable: bool


@dataclass
class Player:
    hand: list[Card]
    chip: Chip


@dataclass
class CellSpecificStats:
    index: Coordinate
    win_count: int = 0
    as_first_move_win_count: int = 0


def random_boolean() -> bool:
    return random.choice([True, False])


def remove_jacks_from_deck(cards: list[Card]) -> list[Card]:
    return [card for card in cards if card not in TWO_EYED_JACKS + ONE_EYED_JACKS]


def make_draw_deck(number_of_decks: int) -> list[Card]:
    deck = remove_jacks_from_deck(FULL_DECK) * number_of_decks
    random.shuffle(deck)
    return deck


def deal_from_deck(
    number_of_players: int, number_of_cards: int, deck: list[Card]
) -> tuple[list[list[Card]], list[Card]]:
    player_cards: list[list[Card]] = [[] for _ in range(number_of_players)]
    for _ in range(number_of_cards):
        for player in range(number_of_players):
            player_cards[player].append(deck.pop(0))
    return player_cards, deck


def make_standard_sequence_board() -> list[list[SequenceBoardCell]]:
    """
    This function makes a standard and randomized sequence board of cards. The standard
    sequence board uses 2 decks of cards (excluding the jacks) for a full count of 48x2 = 96
    possible spaces.

    The boards dimensions are 10x10 with all four of the corner spots removed.
    """
    # creating the deck of cards required for the sequence board
    sequence_board_deck = remove_jacks_from_deck(FULL_DECK) * 2
    random.shuffle(sequence_board_deck)

    # generating the sequence baord from the random shuffled deck of cards
    board_matrix: list[list[SequenceBoardCell]] = []
    for n in range(10):
        # generating the rows
        row: list[SequenceBoardCell] = []
        for m in range(10):
            # if we are generating a corner piece then we want to make sure that we are
            # using a blank space here
            if (n, m) in [(0, 0), (0, 9), (9, 0), (9, 9)]:
                row.append(
                    SequenceBoardCell(
                        coordinate=(n, m), card=BLANK_SPACE, chip=" ", playable=False
                    )
                )
            else:
                selected_card = sequence_board_deck.pop()
                row.append(
                    SequenceBoardCell(
                        coordinate=(n, m), card=selected_card, chip=" ", playable=True
                    )
                )

        board_matrix.append(row)

    return board_matrix


def format_sequence_board_to_str(board: list[list[SequenceBoardCell]]) -> str:
    """
    Takes a sequence board and formats the board into an easier to understand string
    representation.

    This string can then be printed out to the terminal for ease of understanding.
    """

    # Define the width for each cell (card/chip), let's assume max 5 characters for consistency
    cell_width = 8

    board_str = ""
    for row in board:
        row_str = ""
        for cell in row:
            # If the cell has a chip, display it; otherwise, display the card
            content = cell.chip if cell.chip.strip() else cell.card

            # Pad the content to make sure it's always of the same length
            cell_str = f"[{content.center(cell_width)}]"

            # there are some strange formatting happening because of the emojis when
            # printing this out to the terminal. In order to fight against this, we
            # are doing some additional formatting to ensure that everything is formatted
            # as expected when we print out the values
            if content == BLANK_SPACE:
                cell_str = cell_str[:-2] + "]"

            elif content in content in ["🔵", "🟢", "🔴", " ", "<🟡>"]:
                cell_str = cell_str[:-3] + "]"

            row_str += cell_str
        board_str += row_str + "\n"
    return board_str


def place_chip_randomly(
    current_board: list[list[SequenceBoardCell]],
    draw_pile_cards: list[Card],
    hand_cards: list[Card],
    chip_color: Chip,
) -> tuple[list[list[SequenceBoardCell]], list[Card], list[Card], Coordinate]:
    """
    This function plays a turn on the sequence board as if the player was just randomly
    playing a game of sequence with no real thought into their movements.

    This function takes in the current state of the board and players cards and returns out
    the new state of the board and the remaining cards for the user.

    NOTE: Going to ignore dead cards for now since that makes things alot more tricky
    and doesn't really give any real benefit to the implenentation as of right now. We can
    figure out that problem after.

        # dead_cards: cards that have no option to play on the board such that they belong
        # in the discard pile
    """
    # empty_cells: are cells that currently do not have a chip on them; this class is helpful
    # for if the current player has a two eyed jack that can play wild
    empty_cells: list[SequenceBoardCell] = []

    # friendly_cells: are cells that are currently being played by the team; these really do
    # not mean much as of right now...
    friendly_cells: list[SequenceBoardCell] = []

    # opposing_cells: are cells that currently have an opposing team chip on them; this
    # classification is helpful if the current player has a one eyed jack available to play
    opposing_cells: list[SequenceBoardCell] = []

    # available_to_play_cells: are cells that there is currently available to play card in
    # the players hand
    available_to_play_cells: list[SequenceBoardCell] = []

    # scan through the baord to see what spaces I can play given the current available play cards
    for row in current_board:
        for cell in row:
            # classifying based on the chip
            if cell.chip == " ":
                if cell.card in hand_cards:
                    available_to_play_cells.append(cell)
                else:
                    empty_cells.append(cell)

            elif cell.chip == chip_color:
                friendly_cells.append(cell)

            else:
                opposing_cells.append(cell)

    # given how jacks can be played, we need to check to see if there are any jacks
    # within the hand
    can_play_wild = any(jack in hand_cards for jack in TWO_EYED_JACKS)
    can_remove_opposing_wild = any(jack in hand_cards for jack in ONE_EYED_JACKS)

    # given all of the pre-analysis work we can finally make the random turn happen which
    # is defined in the below if statement
    if can_play_wild and random_boolean() and empty_cells:
        # selecting a random cell
        selected_cell = random.choice(empty_cells)
        # updating the board with the new chip
        current_board[selected_cell.coordinate[0]][
            selected_cell.coordinate[1]
        ].chip = chip_color
        # discarding the card from the hand pile and replacing with draw card
        jack_indecies = [
            i for i, card in enumerate(hand_cards) if card in TWO_EYED_JACKS
        ]
        hand_cards.pop(jack_indecies[0])
        hand_cards.append(draw_pile_cards.pop(0))

    elif can_remove_opposing_wild and random_boolean() and opposing_cells:
        # selecting a random cell
        selected_cell = random.choice(empty_cells)
        # updating the board with the empty chip to remove the play
        current_board[selected_cell.coordinate[0]][
            selected_cell.coordinate[1]
        ].chip = " "
        # discarding the card from the hand pile and replacing with draw card
        jack_indecies = [
            i for i, card in enumerate(hand_cards) if card in ONE_EYED_JACKS
        ]
        hand_cards.pop(jack_indecies[0])
        hand_cards.append(draw_pile_cards.pop(0))

    else:
        # selecting a random cell
        selected_cell = random.choice(available_to_play_cells)
        # updating the board with the new chip
        current_board[selected_cell.coordinate[0]][
            selected_cell.coordinate[1]
        ].chip = chip_color
        # discarding the card from the hand pile and replacing with draw card
        jack_indecies = [
            i for i, card in enumerate(hand_cards) if card == selected_cell.card
        ]
        hand_cards.pop(jack_indecies[0])
        hand_cards.append(draw_pile_cards.pop(0))

    return current_board, draw_pile_cards, hand_cards, selected_cell.coordinate


@lru_cache
def get_vertical_indecies(n: int, m: int) -> list[list[Coordinate]]:
    collected_coords: list[list[Coordinate]] = []
    for col in range(m):
        coord_group: list[Coordinate] = []
        for row in range(n):
            coord_group.append((row, col))
        collected_coords.append(coord_group)
    return collected_coords


@lru_cache
def get_horizontal_indecies(n: int, m: int) -> list[list[Coordinate]]:
    collected_coords: list[list[Coordinate]] = []
    for row in range(n):
        coord_group: list[Coordinate] = []
        for col in range(m):
            coord_group.append((row, col))
        collected_coords.append(coord_group)
    return collected_coords


@lru_cache
def get_diagonal_indices(n: int, m: int) -> list[list[Coordinate]]:
    diagonal_indices = []

    # Get diagonals starting from the first column (including main diagonal)
    for start_row in range(n):
        diag = []
        r, c = start_row, 0
        while r >= 0 and c < m:
            diag.append((r, c))
            r -= 1
            c += 1
        diagonal_indices.append(diag)

    # Get diagonals starting from the second column (to avoid duplicates)
    for start_col in range(1, m):
        diag = []
        r, c = n - 1, start_col
        while r >= 0 and c < m:
            diag.append((r, c))
            r -= 1
            c += 1
        diagonal_indices.append(diag)

    return diagonal_indices


def check_vertical_win_condition(
    board: list[list[SequenceBoardCell]], chip_color: Chip
) -> list[Coordinate]:
    rows = len(board)
    cols = len(board[0])
    indecies = get_vertical_indecies(rows, cols)

    for group in indecies:
        group_str = ""
        for index in group:
            group_str += board[index[0]][index[1]].chip

        start_index = group_str.find(chip_color * 5)
        if start_index != -1:
            return group[start_index : start_index + 5]
    return []


def check_horizontal_win_condition(
    board: list[list[SequenceBoardCell]], chip_color: Chip
) -> list[Coordinate]:
    rows = len(board)
    cols = len(board[0])
    indecies = get_horizontal_indecies(rows, cols)

    for group in indecies:
        group_str = ""
        for index in group:
            group_str += board[index[0]][index[1]].chip

        start_index = group_str.find(chip_color * 5)
        if start_index != -1:
            return group[start_index : start_index + 5]
    return []


def check_diagonal_win_condition(
    board: list[list[SequenceBoardCell]], chip_color: Chip
) -> list[Coordinate]:
    rows = len(board)
    cols = len(board[0])
    diag_indecies = get_diagonal_indices(rows, cols)

    def _reverse_diag_index(index: Coordinate) -> Coordinate:
        return (rows - 1 - index[0], index[1])

    for group in diag_indecies:
        diag_str = ""
        rev_diag_str = ""
        for index in group:
            reversed_index = _reverse_diag_index(index)
            diag_str += board[index[0]][index[1]].chip
            rev_diag_str += board[reversed_index[0]][reversed_index[1]].chip

        # checking the diagonal group
        start_index = diag_str.find(chip_color * 5)
        if start_index != -1:
            return group[start_index : start_index + 5]

        # checking the reversed diagonal group
        start_index = rev_diag_str.find(chip_color * 5)
        if start_index != -1:
            return [_reverse_diag_index(index) for index in group][
                start_index : start_index + 5
            ]
    return []


def update_board_with_winning_sequence(
    board: list[list[SequenceBoardCell]], indecies: list[Coordinate]
) -> list[list[SequenceBoardCell]]:
    for index in indecies:
        board[index[0]][index[1]].chip = WINNING_CHIP  # type: ignore
    return board


if __name__ == "__main__":
    game_count = 0
    player_chips: list[Chip] = ["🔵", "🟢", "🔴"]
    win_counter: dict[Chip, dict[str, int]] = {
        "🔵": {"order": 1, "count": 0},
        "🟢": {"order": 2, "count": 0},
        "🔴": {"order": 3, "count": 0},
    }

    # preparing the statistics to be collected ===============================================
    # becuase we want to collect statistics on the game being run, we are going to collect
    # these statistics at the cell level. To do this, we are going to create a very simple
    # matrix of values that will allow us to very easily collect this information and display
    # the information as a heat graph
    board_shape = make_standard_sequence_board()
    rows = len(board_shape)
    cols = len(board_shape[0])
    collected_statistics: list[list[CellSpecificStats]] = []
    for i in range(rows):
        row: list[CellSpecificStats] = []
        for j in range(cols):
            row.append(CellSpecificStats(index=(i, j)))
        collected_statistics.append(row)

    # running the simulation of the data =====================================================
    for _ in range(100_000):
        # Setting up the game by creating the board, creating the draw deck, and dealing
        # to the players
        sequence_board = make_standard_sequence_board()
        draw_deck = make_draw_deck(number_of_decks=2)

        # initializing the two players. This can be made more generic later on
        play_cards, draw_deck = deal_from_deck(
            number_of_players=len(player_chips), number_of_cards=6, deck=draw_deck
        )
        players = []
        for i, chip in enumerate(player_chips):
            players.append(
                Player(hand=play_cards[i], chip=chip),
            )

        draw = False
        winning_chip = " "
        winning_message = ""
        winning_indecies: list[Coordinate] = []
        round_of_play = 1
        first_move_coordinate: dict[Chip, Coordinate] = {}
        while winning_chip == " " and draw == False:
            for player in players:
                try:
                    sequence_board, draw_deck, player.hand, played_coordinate = (
                        place_chip_randomly(
                            current_board=sequence_board,
                            draw_pile_cards=draw_deck,
                            hand_cards=player.hand,
                            chip_color=player.chip,
                        )
                    )

                    # tracking the first move coordinate to gain an understanding of the best
                    # first moves
                    if round_of_play == 1:
                        first_move_coordinate[player.chip] = played_coordinate

                    # checking the vertial win condition
                    if winning_indecies := check_vertical_win_condition(
                        board=sequence_board, chip_color=player.chip
                    ):
                        winning_chip = player.chip
                        winning_message = "You have won with a vertical sequence!"
                        break

                    # checking the horizontal win condition
                    elif winning_indecies := check_horizontal_win_condition(
                        board=sequence_board, chip_color=player.chip
                    ):
                        winning_chip = player.chip
                        winning_message = "You have won with a horizontal sequence!"
                        break

                    # checking the diagonal win condition
                    elif winning_indecies := check_diagonal_win_condition(
                        board=sequence_board, chip_color=player.chip
                    ):
                        winning_chip = player.chip
                        winning_message = "You have won with a diagonal sequence!"
                        break

                except IndexError:
                    draw = True

            round_of_play += 1

        game_count += 1
        print(f"Game count: {game_count}")

        if not draw:
            # collecting the winning indecies
            for index in winning_indecies:
                collected_statistics[index[0]][index[1]].win_count += 1

            # collecting the winning first move index
            winning_fm_index = first_move_coordinate[winning_chip]
            collected_statistics[winning_fm_index[0]][
                winning_fm_index[1]
            ].as_first_move_win_count += 1

            # collecting a win counter for each team
            win_counter[winning_chip]["count"] += 1

    # plotting the statistics for visual intuition ===========================================

    # extracting the statistics that we want to plot...
    def extract_win_counts(matrix: list[list[CellSpecificStats]]):
        return [
            [cell.win_count if cell.win_count != 0 else np.nan for cell in row]
            for row in matrix
        ]

    def extract_first_move_win_counts(matrix: list[list[CellSpecificStats]]):
        return [
            [
                cell.as_first_move_win_count
                if cell.as_first_move_win_count != 0
                else np.nan
                for cell in row
            ]
            for row in matrix
        ]

    # win count matrix ==============================================
    count_matrix = extract_win_counts(collected_statistics)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        count_matrix,  # type: ignore
        annot=True,
        fmt="1.0f",
        cmap="YlGnBu",
        cbar=True,
        mask=np.isnan(count_matrix),
    )
    output_image_path = "plots/winning_cells_matrix.png"
    plt.savefig(output_image_path)
    plt.title("Win Count Heatmap")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.show()

    # first move win count matrix ===================================
    count_matrix = extract_first_move_win_counts(collected_statistics)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        count_matrix,  # type: ignore
        annot=True,
        fmt="1.0f",
        cmap="YlGnBu",
        cbar=True,
        mask=np.isnan(count_matrix),
    )
    output_image_path = "plots/winning_first_move_matrix.png"
    plt.savefig(output_image_path)
    plt.title("First Move Win Count Heatmap")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.show()

    # play order win distribution ===================================
    sorted_data = dict(sorted(win_counter.items(), key=lambda item: item[1]["order"]))
    labels = ["First Turn", "Second Turn", "Third Turn"]
    counts = [value["count"] for value in sorted_data.values()]
    output_image_path = "plots/distribution_of_order_winnings.png"
    plt.savefig(output_image_path)
    plt.figure(figsize=(8, 6))
    plt.bar(labels, counts)
    plt.xlabel("Players")
    plt.ylabel("Win Count")
    plt.title("Wins from First to Last Play Order")
    plt.show()

    # print sequence when we want to print out the end game data
    # print(
    #     f"End Game {game_count} Results ================================================================="
    # )
    # if draw:
    #     print("Game has ended in a draw...")
    # else:
    #     assert winning_indecies
    #     print(
    #         format_sequence_board_to_str(
    #             update_board_with_winning_sequence(sequence_board, winning_indecies)
    #         )
    #     )
    #     for player in players:
    #         print(f"Player {player.chip} Hand: {player.hand}")
    #     print("")
    #     print(f"Congrats team {winning_chip}! {winning_message}")
    #     print(f"The winning indecies are {winning_indecies}")

    # print(f"This game lasted {round_of_play} rounds of play.")
    # print()

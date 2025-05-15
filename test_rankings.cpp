#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>
#include <map>
#include <cstdint>

using namespace std;

const int NUM_SPOTS = 9;
const int NUM_PIECES = 2;

struct CCState {
    int board[NUM_SPOTS] = {0};
    int pieces[2][NUM_PIECES] = {{0}};
    int toMove = 0;
};

class CCheckers {
public:
    int64_t rank(const CCState &s) const;
    bool unrank(int64_t theRank, CCState &s) const;
    int64_t multinomial(unsigned int n, unsigned int k1, unsigned int k2) const;
};

int64_t CCheckers::multinomial(unsigned int n, unsigned int k1, unsigned int k2) const
{
	//assert(n >= (k1 + k2));
	int64_t num = 1;
	const unsigned int bound = (n - (k1 + k2));
	while(n > bound)
	{
		num *= n--;
	}
	
	static uint64_t table[21] =
	{ 1ll, 1ll, 2ll, 6ll, 24ll, 120ll, 720ll, 5040ll, 40320ll, 362880ll, 3628800ll, 39916800ll, 479001600ll,
		6227020800ll, 87178291200ll, 1307674368000ll, 20922789888000ll, 355687428096000ll,
		6402373705728000ll, 121645100408832000ll, 2432902008176640000ll };

	int64_t den = table[k1]*table[k2];
//	while(k1 > 1)
//	{
//		den *= k1--;
//	}
//	while(k2 > 1)
//	{
//		den *= k2--;
//	}
	return num / den;
}

int64_t CCheckers::rank(const CCState &s) const
{
	int64_t r = 0;
	unsigned int l1s = NUM_PIECES, l2s = NUM_PIECES;
	for (int i = 0; (l1s + l2s) > 0; ++i)
	{
		//assert(board[i] >= 0 && board[i] <= 2);  // check the colors
		// cout << "DEBUG: board[" << i << "] = " << board[i] << ", l1s = " << l1s << ", l2s = " << l2s << ", r = " << r << endl;
		switch (s.board[i])
		{
			case 2:
				l2s -= 1;
				break;
			case 1:
				if (l2s > 0)
				{
					r = r + multinomial(NUM_SPOTS - i - 1, l1s, l2s-1);
				}
				l1s -= 1;
				break;
			default: // case 0:
				if (l2s > 0)
				{
					r = r + multinomial(NUM_SPOTS - i - 1, l1s, l2s-1);
				}
				if (l1s > 0)
				{
					r = r + multinomial(NUM_SPOTS - i - 1, l1s-1, l2s);
				}
				break;
		}
	}
	// LSB stores the player to move
	return (r << 1) + s.toMove;
}

/*
 * Rank algorithm adapted from "GPU Exploration of Two-Player Games with
 * Perfect Hash Functions" by Stefan Edelkamp, Damian Sulewski,
 * Cengizhan Yücel, Third Annual Symposium on Combinatorial Search, 2010.
 */
bool CCheckers::unrank(int64_t theRank, CCState &s) const
{
        // clear to zero, this is necessary as we may break out of the
        // loop before i gets all the way to NUM_SPOTS-1 (i.e., when
        // all the pieces have been placed)
 	memset(s.board, 0, NUM_SPOTS*sizeof(int));	
  
	// LSB stores player to move.
	s.toMove = theRank & 0x1;
	theRank = theRank >> 1;
	
	unsigned int l1s = NUM_PIECES, l2s = NUM_PIECES;
	for (int i=0; (l1s + l2s) > 0; ++i)
	{
		int64_t value1, value2;
		if (l2s > 0)
		{
			value2 = multinomial(NUM_SPOTS - i - 1, l1s, l2s - 1);
		}
		else {
			value2 = 0;
		}
		if(l1s > 0)
		{
			value1 = multinomial(NUM_SPOTS - i - 1, l1s - 1, l2s);
		}
		else {
			value1 = 0;
		}
		// this block of code guarantees that the element at the ith index gets either 2, 1, 0
		if (theRank < value2)
		{
			if(l2s <= 0){ // trying to place too many 2s
				return false;
			}
			s.board[i] = 2;
			s.pieces[1][l2s-1] = i;
			l2s = l2s - 1;
		}
		else if (theRank < (value1 + value2))
		{
			if (l1s <= 0)
			{ // trying to place too many 1s;
				return false;
			}
			s.board[i] = 1;
			s.pieces[0][l1s-1] = i;
			theRank = theRank - value2;
			l1s = l1s - 1;
		}
		else {	
			s.board[i] = 0;
			theRank = theRank - (value1 + value2);
		}
	}
	return true;
}

// Helper to compare two CCState objects
bool equal(const CCState &a, const CCState &b) {
    if (a.toMove != b.toMove) return false;
    for (int i = 0; i < NUM_SPOTS; ++i)
        if (a.board[i] != b.board[i]) return false;
    return true;
}

void printState(const CCState &s) {
    cout << "Board: ";
    for (int i = 0; i < NUM_SPOTS; ++i)
        cout << s.board[i] << " ";
    cout << " | toMove: " << s.toMove << endl;
}

int main() {
    CCheckers checkers;

    /*
    CCState original;
    original.board[0] = 1;
    original.board[1] = 0;
    original.board[2] = 2;
    original.board[3] = 1;
    original.board[4] = 0;
    original.board[5] = 2;
    original.board[6] = 0;
    original.board[7] = 0;
    original.toMove = 1;
    */

    CCState original = {
        // board
        {0, 1, 1, 0, 0, 0, 2, 2, 0},
        // pieces
        {{0}},
        // toMove
        0
    };

    printState(original);

    int64_t r = checkers.rank(original);
    cout << "Rank: " << r << endl;
    r = 966;

    CCState recovered;
    if (checkers.unrank(r, recovered)) {
        printState(recovered);
    } else {
        cout << "Unrank failed!\n";
    }

    return 0;
}

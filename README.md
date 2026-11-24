# AndyBot

AndyBot is a UCI chess engine written in C++. It combines an alpha-beta search with an NNUE evaluation network.

## Download

Windows binaries are in this repository. Pick the one your CPU supports:

| File | Needs |
|---|---|
| `andybot-1.0-avx512.exe` | AVX-512 with VNNI (AMD Zen 4 and later, recent Intel server and some desktop CPUs) |
| `andybot-1.0-avx2.exe` | AVX2 and BMI2 (most CPUs since 2015) |

The network is built into the executable, so there is nothing else to download.

## UCI options

| Option | Default | Range | Meaning |
|---|---|---|---|
| `Hash` | 256 | 1 to 16384 | transposition table size in MB |
| `Threads` | 1 | 1 to 256 | search threads |
| `Ponder` | false | | think on the opponent's time |
| `Move Overhead` | 30 | 0 to 5000 | milliseconds kept back per move for GUI and network delay |
| `Clear Hash` | | | button: empties the transposition table |

AndyBot has no opening book and does not use endgame tablebases.

## Source code

The complete source is in [`src/`](src), and the network it uses is in [`networks/`](networks). The engine is a
single translation unit, `src/main.cpp`, and builds with clang.

## Engine details

- **Evaluation:** an NNUE network with HalfKAv2_hm input features and a 1024-wide accumulator per side, a pairwise
  product (1024 to 512 per side), and eight small output heads chosen by piece count, each with a skip connection.
  Inference is quantised and uses AVX2 or AVX-512.
- **Search:** iterative-deepening principal-variation search with a lockless shared transposition table, null-move
  pruning, late-move reductions, singular extensions, futility and SEE pruning, history-based move ordering,
  quiescence search, pondering and Lazy SMP.
- **Training:** the network was trained with [bullet](https://github.com/jw1912/bullet) on about 120 billion
  positions from Leela Chess Zero and Stockfish training data, filtered to quiet positions.

## Credit

AndyBot could not have been written without the engines and projects below.

### Engine influences

- [Berserk](https://github.com/jhonnold/berserk)
- [Stockfish](https://github.com/official-stockfish/Stockfish)
- [Leela Chess Zero](https://github.com/LeelaChessZero/lc0)

AndyBot's search is based on Berserk's.

### Additional resources

- [bullet](https://github.com/jw1912/bullet)
- [nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch)
- Training data from the Leela Chess Zero and Stockfish projects
- [Cute Chess](https://github.com/cutechess/cutechess)
- [Chess Programming Wiki](https://www.chessprogramming.org)
- [TalkChess](https://talkchess.com)
- [CCRL](https://computerchess.org.uk/ccrl/)

## Licence

GNU General Public License, version 3 or later. See [LICENSE](LICENSE).

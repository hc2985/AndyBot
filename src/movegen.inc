// AndyBot, a UCI chess engine. Copyright (C) 2026 Hyeonung Cho.
//
// AndyBot is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
// later version. It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
// (the LICENSE file) for details. Credits to the engines AndyBot builds on are in README.md.

enum class GenMode : std::uint8_t { All, Tactical, Quiet };

template <typename Out>
void genpseudo_bb(const Board& b, Out& out, GenMode mode);

template <typename Out>
void genpseudo_mailbox(const Board& b, Out& out, bool caponly) {
    int us = b.stm;
    for (int from = 0; from < 64; ++from) {
        char p = b.sq[from];
        if (p == '.' || col(p) != us) continue;
        int f = file_of(from), r = rank_of(from);
        char pt = lower_piece_ascii(p);

        if (pt == 'p') {
            int dir = (us == WHITE) ? 8 : -8;
            int start = (us == WHITE) ? 1 : 6;
            int prom = (us == WHITE) ? 6 : 1;
            int one = from + dir;
            // Capture-only generation is also used by qsearch.  Quiet
            // promotions are tactical and must remain available there.
            if (onb(one) && b.sq[one] == '.'
                && (!caponly || r == prom)) {
                if (r == prom) {
                    for (char pr : {'q','r','b','n'})
                        out.push_back(Move{from,one,pr,false,false,false,false});
                } else {
                    out.push_back(Move{from,one,0,false,false,false,false});
                    if (r == start) {
                        int two = from + 2 * dir;
                        if (onb(two) && b.sq[two] == '.') out.push_back(Move{from,two,0,false,false,false,true});
                    }
                }
            }
            auto addcap = [&](int to) {
                if (!onb(to)) return;
                if (b.sq[to] != '.' && col(b.sq[to]) == (us ^ 1)) {
                    if (r == prom) for (char pr : {'q','r','b','n'}) out.push_back(Move{from,to,pr,true,false,false,false});
                    else out.push_back(Move{from,to,0,true,false,false,false});
                }
            };
            if (us == WHITE) { if (f > 0) addcap(from + 7); if (f < 7) addcap(from + 9); }
            else { if (f > 0) addcap(from - 9); if (f < 7) addcap(from - 7); }
            if (b.ep >= 0) {
                if (us == WHITE) { if (f > 0 && from + 7 == b.ep) out.push_back(Move{from,b.ep,0,true,true,false,false}); if (f < 7 && from + 9 == b.ep) out.push_back(Move{from,b.ep,0,true,true,false,false}); }
                else { if (f > 0 && from - 9 == b.ep) out.push_back(Move{from,b.ep,0,true,true,false,false}); if (f < 7 && from - 7 == b.ep) out.push_back(Move{from,b.ep,0,true,true,false,false}); }
            }
            continue;
        }

        if (pt == 'n') {
            static const int df[8] = {1,2,2,1,-1,-2,-2,-1};
            static const int dr[8] = {2,1,-1,-2,-2,-1,1,2};
            for (int i = 0; i < 8; ++i) {
                int nf = f + df[i], nr = r + dr[i];
                if (nf < 0 || nf > 7 || nr < 0 || nr > 7) continue;
                int to = nr * 8 + nf;
                if (b.sq[to] == '.') { if (!caponly) out.push_back(Move{from,to,0,false,false,false,false}); }
                else if (col(b.sq[to]) == (us ^ 1)) out.push_back(Move{from,to,0,true,false,false,false});
            }
            continue;
        }

        if (pt == 'b' || pt == 'r' || pt == 'q') {
            std::vector<std::pair<int,int>> d;
            if (pt == 'b' || pt == 'q') { d.push_back({1,1}); d.push_back({1,-1}); d.push_back({-1,1}); d.push_back({-1,-1}); }
            if (pt == 'r' || pt == 'q') { d.push_back({1,0}); d.push_back({-1,0}); d.push_back({0,1}); d.push_back({0,-1}); }
            for (auto [df, dr] : d) {
                int nf = f + df, nr = r + dr;
                while (nf >= 0 && nf <= 7 && nr >= 0 && nr <= 7) {
                    int to = nr * 8 + nf;
                    if (b.sq[to] == '.') { if (!caponly) out.push_back(Move{from,to,0,false,false,false,false}); }
                    else { if (col(b.sq[to]) == (us ^ 1)) out.push_back(Move{from,to,0,true,false,false,false}); break; }
                    nf += df; nr += dr;
                }
            }
            continue;
        }

        if (pt == 'k') {
            for (int df = -1; df <= 1; ++df) for (int dr = -1; dr <= 1; ++dr) {
                if (!df && !dr) continue;
                int nf = f + df, nr = r + dr;
                if (nf < 0 || nf > 7 || nr < 0 || nr > 7) continue;
                int to = nr * 8 + nf;
                if (b.sq[to] == '.') { if (!caponly) out.push_back(Move{from,to,0,false,false,false,false}); }
                else if (col(b.sq[to]) == (us ^ 1)) out.push_back(Move{from,to,0,true,false,false,false});
            }
            if (!caponly) {
                if (us == WHITE && from == 4) {
                    if ((b.castling & CWK) && b.sq[5] == '.' && b.sq[6] == '.' && !attacked(b,4,BLACK) && !attacked(b,5,BLACK) && !attacked(b,6,BLACK)) out.push_back(Move{4,6,0,false,false,true,false});
                    if ((b.castling & CWQ) && b.sq[3] == '.' && b.sq[2] == '.' && b.sq[1] == '.' && !attacked(b,4,BLACK) && !attacked(b,3,BLACK) && !attacked(b,2,BLACK)) out.push_back(Move{4,2,0,false,false,true,false});
                }
                if (us == BLACK && from == 60) {
                    if ((b.castling & CBK) && b.sq[61] == '.' && b.sq[62] == '.' && !attacked(b,60,WHITE) && !attacked(b,61,WHITE) && !attacked(b,62,WHITE)) out.push_back(Move{60,62,0,false,false,true,false});
                    if ((b.castling & CBQ) && b.sq[59] == '.' && b.sq[58] == '.' && b.sq[57] == '.' && !attacked(b,60,WHITE) && !attacked(b,59,WHITE) && !attacked(b,58,WHITE)) out.push_back(Move{60,58,0,false,false,true,false});
                }
            }
            continue;
        }
    }
}

// Top-level genpseudo seen by search and other callers: routes to bitboard.
template <typename Out>
void genpseudo(const Board& b, Out& out, bool caponly) {
    genpseudo_bb(b, out, caponly ? GenMode::Tactical : GenMode::All);
}

template <typename Out>
void genpseudo_tactical(const Board& b, Out& out) {
    genpseudo_bb(b, out, GenMode::Tactical);
}

template <typename Out>
void genpseudo_quiet(const Board& b, Out& out) {
    genpseudo_bb(b, out, GenMode::Quiet);
}

std::vector<Move> genlegal(Board& b, bool caponly) {
    std::vector<Move> p; p.reserve(128); genpseudo<std::vector<Move>>(b, p, caponly);
    int us = b.stm;
    std::size_t wr = 0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        const Move m = p[i];
        Undo u; makemove(b, m, u);
        bool ok = !incheck(b, us);
        unmakemove(b, u);
        if (ok) p[wr++] = m;
    }
    p.resize(wr);
    return p;
}

// =============================================================================
// BITBOARD-BASED MOVE GENERATOR (parallel implementation, validated by perft)
// =============================================================================

// True iff `sq` is attacked by any piece of color `by_color`.
__forceinline
bool is_square_attacked_bb(const Board& b, int sq, int by_color) {
    // pawn attacks: a `by_color` pawn at X attacks sq <=> PAWN_ATTACKS[by_color][X] contains sq
    // Equivalent reverse lookup: PAWN_ATTACKS[opposite][sq] gives squares from which a `by_color` pawn could attack sq.
    if (bb::PAWN_ATTACKS[by_color ^ 1][sq] & b.pieces[by_color][0]) return true;
    if (bb::KNIGHT_ATTACKS[sq] & b.pieces[by_color][1]) return true;
    if (bb::KING_ATTACKS[sq]   & b.pieces[by_color][5]) return true;
    const bb::Bitboard bq = b.pieces[by_color][2] | b.pieces[by_color][4];
    if (bb::bishop_attacks(sq, b.occ) & bq) return true;
    const bb::Bitboard rq = b.pieces[by_color][3] | b.pieces[by_color][4];
    if (bb::rook_attacks(sq, b.occ) & rq) return true;
    return false;
}

// BETWEEN: squares strictly between two aligned squares (else 0).
// LINE: the whole rank/file/diagonal through both (else 0).
struct LineTables {
    bb::Bitboard between[64][64]{};
    bb::Bitboard line[64][64]{};
    LineTables() {
        static constexpr int dirs[8][2] = {{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}};
        for (int a = 0; a < 64; ++a) {
            for (const auto& d : dirs) {
                bb::Bitboard full = bb::Bitboard(1) << a;
                for (int sign = -1; sign <= 1; sign += 2) {
                    for (int s = 1; s < 8; ++s) {
                        const int f = (a & 7) + sign * d[0] * s, r = (a >> 3) + sign * d[1] * s;
                        if (f < 0 || f > 7 || r < 0 || r > 7) break;
                        full |= bb::Bitboard(1) << (r * 8 + f);
                    }
                }
                bb::Bitboard path = 0;
                for (int s = 1; s < 8; ++s) {
                    const int f = (a & 7) + d[0] * s, r = (a >> 3) + d[1] * s;
                    if (f < 0 || f > 7 || r < 0 || r > 7) break;
                    const int b = r * 8 + f;
                    between[a][b] = path;
                    line[a][b] = full;
                    path |= bb::Bitboard(1) << b;
                }
            }
        }
    }
};
const LineTables LINE_TABLES;

// Is `sq` attacked by `by_color`, with `occ` as the blocker set?
inline bool square_attacked_occ(const Board& b, int sq, int by_color, bb::Bitboard occ) {
    if (bb::PAWN_ATTACKS[by_color ^ 1][sq] & b.pieces[by_color][0]) return true;
    if (bb::KNIGHT_ATTACKS[sq] & b.pieces[by_color][1]) return true;
    if (bb::KING_ATTACKS[sq] & b.pieces[by_color][5]) return true;
    if (bb::bishop_attacks(sq, occ) & (b.pieces[by_color][2] | b.pieces[by_color][4])) return true;
    if (bb::rook_attacks(sq, occ) & (b.pieces[by_color][3] | b.pieces[by_color][4])) return true;
    return false;
}

// Pieces of the side to move that are pinned to its own king.
inline bb::Bitboard pinned_pieces(const Board& b) {
    const int us = b.stm, them = us ^ 1;
    const int ksq = b.king[us];
    bb::Bitboard snipers =
        (bb::rook_attacks(ksq, 0) & (b.pieces[them][3] | b.pieces[them][4]))
      | (bb::bishop_attacks(ksq, 0) & (b.pieces[them][2] | b.pieces[them][4]));
    bb::Bitboard pinned = 0;
    while (snipers) {
        const int s = bb::pop_lsb(snipers);
        const bb::Bitboard blockers = LINE_TABLES.between[ksq][s] & b.occ;
        if (blockers && !(blockers & (blockers - 1)) && (blockers & b.color_occ[us])) pinned |= blockers;
    }
    return pinned;
}

// Legality of a pseudo-legal move when the side to move is NOT in check.
// Not for en passant or castling (callers use make + incheck for those).
inline bool legal_when_not_in_check(const Board& b, const Move& m, bb::Bitboard pinned) {
    const int us = b.stm;
    const int ksq = b.king[us];
    if (m.from == ksq)
        return !square_attacked_occ(b, m.to, us ^ 1, b.occ & ~(bb::Bitboard(1) << m.from));
    if (!((pinned >> m.from) & 1)) return true;
    return (LINE_TABLES.line[ksq][m.from] >> m.to) & 1;
}

// Pieces of the opponent giving check to the side to move.
inline bb::Bitboard checkers_of(const Board& b) {
    const int us = b.stm, them = us ^ 1;
    const int k = b.king[us];
    return (bb::PAWN_ATTACKS[us][k] & b.pieces[them][0])
         | (bb::KNIGHT_ATTACKS[k] & b.pieces[them][1])
         | (bb::bishop_attacks(k, b.occ) & (b.pieces[them][2] | b.pieces[them][4]))
         | (bb::rook_attacks(k, b.occ) & (b.pieces[them][3] | b.pieces[them][4]));
}

// Legality of a pseudo-legal move when the side to move IS in check (not en passant / castling). Same verdict as
// make + incheck: king moves need a safe square; otherwise single check only, unpinned, and capture or block.
inline bool legal_in_check(const Board& b, const Move& m, bb::Bitboard pinned, bb::Bitboard checkers) {
    const int us = b.stm;
    const int k = b.king[us];
    if (m.from == k)
        return !square_attacked_occ(b, m.to, us ^ 1, b.occ & ~(bb::Bitboard(1) << m.from));
    if (checkers & (checkers - 1)) return false;
    if ((pinned >> m.from) & 1) return false;
    return ((LINE_TABLES.between[k][bb::lsb(checkers)] | checkers) >> m.to) & 1;
}

// Would this quiet, non-castling, non-promotion move put the opponent in check? Exactly
// is_square_attacked_bb(child, king[them], us) evaluated on the child position, without making the move.
inline bool quiet_gives_check(const Board& b, const Move& m, int moved_kind) {
    const int us = b.stm, them = us ^ 1;
    const int ek = b.king[them];
    const bb::Bitboard fb = bb::sq_bb(m.from), tb = bb::sq_bb(m.to);
    const bb::Bitboard occ = (b.occ & ~fb) | tb;
    bb::Bitboard mine[6];
    for (int k = 0; k < 6; ++k) mine[k] = b.pieces[us][k];
    mine[moved_kind] = (mine[moved_kind] & ~fb) | tb;
    if (bb::PAWN_ATTACKS[them][ek] & mine[0]) return true;
    if (bb::KNIGHT_ATTACKS[ek] & mine[1]) return true;
    if (bb::KING_ATTACKS[ek] & mine[5]) return true;
    if (bb::bishop_attacks(ek, occ) & (mine[2] | mine[4])) return true;
    if (bb::rook_attacks(ek, occ) & (mine[3] | mine[4])) return true;
    return false;
}

// True only when no quiet move of the side to move (castling included; promotions and en passant are not quiet)
// can give check. Conservative: it may answer false although no quiet checks exist. The opponent must not be in
// check, so a slider can never stand between its own target square and the enemy king.
inline bool no_quiet_check_possible(const Board& b) {
    const int us = b.stm, them = us ^ 1;
    const int ek = b.king[them];
    const bb::Bitboard occ = b.occ, empty = ~occ;
    const bb::Bitboard rq = b.pieces[us][3] | b.pieces[us][4];
    const bb::Bitboard bq = b.pieces[us][2] | b.pieces[us][4];
    // Discovered checks: one of our pieces stands alone between one of our sliders and their king.
    bb::Bitboard snipers = (bb::rook_attacks(ek, 0) & rq) | (bb::bishop_attacks(ek, 0) & bq);
    while (snipers) {
        const bb::Bitboard blockers = LINE_TABLES.between[ek][bb::pop_lsb(snipers)] & occ;
        if (!(blockers & (blockers - 1)) && (blockers & b.color_occ[us])) return false;
    }
    const bb::Bitboard line = bb::rook_attacks(ek, occ);
    // Castling: the rook lands on the d- or f-file square (line computed without our king and rooks: a superset).
    if (b.castling & (us == WHITE ? (CWK | CWQ) : (CBK | CBQ))) {
        const bb::Bitboard dests = us == WHITE ? (bb::sq_bb(5) | bb::sq_bb(3)) : (bb::sq_bb(61) | bb::sq_bb(59));
        if (bb::rook_attacks(ek, occ & ~b.pieces[us][5] & ~b.pieces[us][3]) & dests) return false;
    }
    // Direct checks: an empty square from which that piece type attacks their king, reachable by one of ours.
    const bb::Bitboard lsq = line & empty;
    const bb::Bitboard dsq = bb::bishop_attacks(ek, occ) & empty;
    const bb::Bitboard nsq = bb::KNIGHT_ATTACKS[ek] & empty;
    bb::Bitboard p;
    for (p = b.pieces[us][1]; p;) if (bb::KNIGHT_ATTACKS[bb::pop_lsb(p)] & nsq) return false;
    for (p = b.pieces[us][2]; p;) if (bb::bishop_attacks(bb::pop_lsb(p), occ) & dsq) return false;
    for (p = b.pieces[us][3]; p;) if (bb::rook_attacks(bb::pop_lsb(p), occ) & lsq) return false;
    for (p = b.pieces[us][4]; p;) {
        const int s = bb::pop_lsb(p);
        if ((bb::bishop_attacks(s, occ) | bb::rook_attacks(s, occ)) & (dsq | lsq)) return false;
    }
    const bb::Bitboard psq = bb::PAWN_ATTACKS[them][ek] & empty;  // squares from which our pawn attacks their king
    if (psq) {
        const bb::Bitboard pawns = b.pieces[us][0];
        if (us == WHITE) {
            const bb::Bitboard one = (pawns << 8) & empty;
            if ((one | (((one & bb::RANK_3_BB) << 8) & empty)) & psq) return false;
        } else {
            const bb::Bitboard one = (pawns >> 8) & empty;
            if ((one | (((one & bb::RANK_6_BB) >> 8) & empty)) & psq) return false;
        }
    }
    return true;
}
// Bitboard pseudo-legal move generator. Mirrors genpseudo() exactly in semantics
// — including the same Move struct flags — so callers and downstream make/unmake
// see identical move objects. Templated on output container so the same body
// serves both heap (std::vector<Move>) and stack (MoveList) callers.
template <typename Out>
void genpseudo_bb(const Board& b, Out& out, GenMode mode) {
    const int us = b.stm;
    const int them = us ^ 1;
    const bb::Bitboard us_occ = b.color_occ[us];
    const bb::Bitboard them_occ = b.color_occ[them];
    const bb::Bitboard all_occ = b.occ;
    const bb::Bitboard not_us = ~us_occ;
    const bool want_tactical = mode != GenMode::Quiet;
    const bool want_quiet = mode != GenMode::Tactical;

    auto emit_simple = [&](int from, bb::Bitboard targets) {
        if (!want_quiet) targets &= them_occ;
        else if (!want_tactical) targets &= ~them_occ;
        while (targets) {
            const int to = bb::pop_lsb(targets);
            Move m; m.from = from; m.to = to;
            m.cap = ((them_occ >> to) & 1ULL) != 0;
            out.push_back(m);
        }
    };

    // ===== Knights =====
    {
        bb::Bitboard kn = b.pieces[us][1];
        while (kn) {
            const int from = bb::pop_lsb(kn);
            emit_simple(from, bb::KNIGHT_ATTACKS[from] & not_us);
        }
    }
    // ===== Bishops =====
    {
        bb::Bitboard bp = b.pieces[us][2];
        while (bp) {
            const int from = bb::pop_lsb(bp);
            emit_simple(from, bb::bishop_attacks(from, all_occ) & not_us);
        }
    }
    // ===== Rooks =====
    {
        bb::Bitboard rk = b.pieces[us][3];
        while (rk) {
            const int from = bb::pop_lsb(rk);
            emit_simple(from, bb::rook_attacks(from, all_occ) & not_us);
        }
    }
    // ===== Queens =====
    {
        bb::Bitboard qn = b.pieces[us][4];
        while (qn) {
            const int from = bb::pop_lsb(qn);
            emit_simple(from, bb::queen_attacks(from, all_occ) & not_us);
        }
    }
    // ===== King (non-castling) =====
    {
        bb::Bitboard kg = b.pieces[us][5];
        while (kg) {
            const int from = bb::pop_lsb(kg);
            emit_simple(from, bb::KING_ATTACKS[from] & not_us);
        }
    }
    // ===== Castling =====
    if (want_quiet) {
        if (us == WHITE) {
            if ((b.castling & CWK)
                && !(all_occ & (bb::sq_bb(5) | bb::sq_bb(6)))
                && !is_square_attacked_bb(b, 4, BLACK)
                && !is_square_attacked_bb(b, 5, BLACK)
                && !is_square_attacked_bb(b, 6, BLACK)) {
                Move m; m.from = 4; m.to = 6; m.castle = true; out.push_back(m);
            }
            if ((b.castling & CWQ)
                && !(all_occ & (bb::sq_bb(1) | bb::sq_bb(2) | bb::sq_bb(3)))
                && !is_square_attacked_bb(b, 4, BLACK)
                && !is_square_attacked_bb(b, 3, BLACK)
                && !is_square_attacked_bb(b, 2, BLACK)) {
                Move m; m.from = 4; m.to = 2; m.castle = true; out.push_back(m);
            }
        } else {
            if ((b.castling & CBK)
                && !(all_occ & (bb::sq_bb(61) | bb::sq_bb(62)))
                && !is_square_attacked_bb(b, 60, WHITE)
                && !is_square_attacked_bb(b, 61, WHITE)
                && !is_square_attacked_bb(b, 62, WHITE)) {
                Move m; m.from = 60; m.to = 62; m.castle = true; out.push_back(m);
            }
            if ((b.castling & CBQ)
                && !(all_occ & (bb::sq_bb(57) | bb::sq_bb(58) | bb::sq_bb(59)))
                && !is_square_attacked_bb(b, 60, WHITE)
                && !is_square_attacked_bb(b, 59, WHITE)
                && !is_square_attacked_bb(b, 58, WHITE)) {
                Move m; m.from = 60; m.to = 58; m.castle = true; out.push_back(m);
            }
        }
    }

    // ===== Pawns =====
    const bb::Bitboard pawns = b.pieces[us][0];
    if (us == WHITE) {
        const bb::Bitboard empty = ~all_occ;
        const bb::Bitboard prom = bb::RANK_8_BB;
        const bb::Bitboard nonprom = ~prom;
        // Single pushes.  Capture-only generation still emits quiet
        // promotions because qsearch must see them.
        const bb::Bitboard pushes = (pawns << 8) & empty;
        bb::Bitboard pp = pushes & prom;
        if (want_tactical) {
            while (pp) {
                const int to = bb::pop_lsb(pp);
                for (char promo : {'Q','R','B','N'}) {
                    Move m; m.from = to - 8; m.to = to; m.promo = promo; out.push_back(m);
                }
            }
        }
        if (want_quiet) {
            bb::Bitboard pnp = pushes & nonprom;
            while (pnp) {
                const int to = bb::pop_lsb(pnp);
                Move m; m.from = to - 8; m.to = to; out.push_back(m);
            }
            // Double push
            bb::Bitboard d = ((pushes & bb::RANK_3_BB) << 8) & empty;
            while (d) {
                const int to = bb::pop_lsb(d);
                Move m; m.from = to - 16; m.to = to; m.dbl = true; out.push_back(m);
            }
        }
        // Captures NW (+7) and NE (+9)
        if (want_tactical) {
        bb::Bitboard cnw = ((pawns & bb::NOT_FILE_A) << 7) & them_occ;
        bb::Bitboard cne = ((pawns & bb::NOT_FILE_H) << 9) & them_occ;
        bb::Bitboard cnw_p = cnw & prom, cnw_np = cnw & nonprom;
        bb::Bitboard cne_p = cne & prom, cne_np = cne & nonprom;
        while (cnw_np) { const int to = bb::pop_lsb(cnw_np); Move m; m.from = to - 7; m.to = to; m.cap = true; out.push_back(m); }
        while (cnw_p)  { const int to = bb::pop_lsb(cnw_p);  for (char p : {'Q','R','B','N'}) { Move m; m.from = to - 7; m.to = to; m.cap = true; m.promo = p; out.push_back(m); } }
        while (cne_np) { const int to = bb::pop_lsb(cne_np); Move m; m.from = to - 9; m.to = to; m.cap = true; out.push_back(m); }
        while (cne_p)  { const int to = bb::pop_lsb(cne_p);  for (char p : {'Q','R','B','N'}) { Move m; m.from = to - 9; m.to = to; m.cap = true; m.promo = p; out.push_back(m); } }
        // En passant
        if (b.ep >= 0) {
            const bb::Bitboard ept = bb::sq_bb(b.ep);
            if (((pawns & bb::NOT_FILE_A) << 7) & ept) { Move m; m.from = b.ep - 7; m.to = b.ep; m.ep = true; m.cap = true; out.push_back(m); }
            if (((pawns & bb::NOT_FILE_H) << 9) & ept) { Move m; m.from = b.ep - 9; m.to = b.ep; m.ep = true; m.cap = true; out.push_back(m); }
        }
        }
    } else {
        const bb::Bitboard empty = ~all_occ;
        const bb::Bitboard prom = bb::RANK_1_BB;
        const bb::Bitboard nonprom = ~prom;
        const bb::Bitboard pushes = (pawns >> 8) & empty;
        bb::Bitboard pp = pushes & prom;
        if (want_tactical) {
            while (pp) {
                const int to = bb::pop_lsb(pp);
                for (char promo : {'q','r','b','n'}) {
                    Move m; m.from = to + 8; m.to = to; m.promo = promo; out.push_back(m);
                }
            }
        }
        if (want_quiet) {
            bb::Bitboard pnp = pushes & nonprom;
            while (pnp) {
                const int to = bb::pop_lsb(pnp);
                Move m; m.from = to + 8; m.to = to; out.push_back(m);
            }
            bb::Bitboard d = ((pushes & bb::RANK_6_BB) >> 8) & empty;
            while (d) {
                const int to = bb::pop_lsb(d);
                Move m; m.from = to + 16; m.to = to; m.dbl = true; out.push_back(m);
            }
        }
        // Black pawn captures: SE = -7 (file -1, rank -1)... wait, careful with the directions.
        if (want_tactical) {
        // Black pawn at sq attacks sq-9 (down-left, file-1) and sq-7 (down-right, file+1).
        // From perspective of WHERE: a black pawn capture lands at `to`; from-square is to+9 (if cap from upper-right perspective: pawn moved from rank+1, file+1) or to+7 (rank+1, file-1).
        // Actually let me redo: if a black pawn moves from sq to sq-7 (which is rank-1, file+1), then the from-relative-to-to is "to + 7" mapped from "from - 7". Hmm.
        // Cleaner: a black pawn at FROM attacking TO has TO = FROM - 9 (down-left) or TO = FROM - 7 (down-right).
        // So for captures landing on TO: FROM = TO + 9 (if down-left) or FROM = TO + 7 (if down-right).
        // For "down-right" (FILE+1, the pawn was on FILE-1 of TO): the source file must be < H (i.e., the pawn had file != H so shifting to capture down-right is valid). Generation: ((pawns & NOT_FILE_H) >> 7) lands at TO with shifts (-7 in flat sq). Pawns at FILE!=H shift right by 7 puts them at FILE+1, RANK-1. OK.
        // For "down-left" (FILE-1, pawn was on FILE+1 of TO): ((pawns & NOT_FILE_A) >> 9) — pawns at FILE!=A shift right by 9 puts them at FILE-1, RANK-1.
        bb::Bitboard csr = ((pawns & bb::NOT_FILE_H) >> 7) & them_occ;  // down-right capture targets
        bb::Bitboard csl = ((pawns & bb::NOT_FILE_A) >> 9) & them_occ;  // down-left capture targets
        bb::Bitboard csr_p = csr & prom, csr_np = csr & nonprom;
        bb::Bitboard csl_p = csl & prom, csl_np = csl & nonprom;
        while (csr_np) { const int to = bb::pop_lsb(csr_np); Move m; m.from = to + 7; m.to = to; m.cap = true; out.push_back(m); }
        while (csr_p)  { const int to = bb::pop_lsb(csr_p);  for (char p : {'q','r','b','n'}) { Move m; m.from = to + 7; m.to = to; m.cap = true; m.promo = p; out.push_back(m); } }
        while (csl_np) { const int to = bb::pop_lsb(csl_np); Move m; m.from = to + 9; m.to = to; m.cap = true; out.push_back(m); }
        while (csl_p)  { const int to = bb::pop_lsb(csl_p);  for (char p : {'q','r','b','n'}) { Move m; m.from = to + 9; m.to = to; m.cap = true; m.promo = p; out.push_back(m); } }
        // En passant
        if (b.ep >= 0) {
            const bb::Bitboard ept = bb::sq_bb(b.ep);
            if (((pawns & bb::NOT_FILE_H) >> 7) & ept) { Move m; m.from = b.ep + 7; m.to = b.ep; m.ep = true; m.cap = true; out.push_back(m); }
            if (((pawns & bb::NOT_FILE_A) >> 9) & ept) { Move m; m.from = b.ep + 9; m.to = b.ep; m.ep = true; m.cap = true; out.push_back(m); }
        }
        }
    }

    // Suppress unused-var warning for `them` in some build configs.
    (void)them;
}

// Reconstruct a hash-verified TT move without generating every pseudo-legal
// move. The validation deliberately mirrors genpseudo_bb semantics, including
// quiet promotions in capture-only qsearch.
bool decode_tt_move(const Board& b, int key, bool caponly, Move& out) {
    if (key < 0 || key > ((4 << 12) | (63 << 6) | 63)) return false;
    const int from = key & 63;
    const int to = (key >> 6) & 63;
    const int pc = (key >> 12) & 7;
    if (from == to || pc > 4) return false;

    const char piece = b.sq[from];
    const char target = b.sq[to];
    const int us = b.stm;
    if (piece == '.' || col(piece) != us || (target != '.' && col(target) == us)) return false;

    char promo = 0;
    switch (pc) {
        case 1: promo = 'q'; break;
        case 2: promo = 'r'; break;
        case 3: promo = 'b'; break;
        case 4: promo = 'n'; break;
        default: break;
    }

    const char kind = lower_piece_ascii(piece);
    const int delta = to - from;
    const int file_delta = std::abs(file_of(to) - file_of(from));
    bool capture = target != '.';
    bool ep = false;
    bool castle = false;
    bool dbl = false;
    bool valid = false;

    if (kind == 'p') {
        const int dir = us == WHITE ? 8 : -8;
        const int start_rank = us == WHITE ? 1 : 6;
        const int promotion_rank = us == WHITE ? 7 : 0;
        if (delta == dir && target == '.') {
            valid = true;
        } else if (delta == 2 * dir && rank_of(from) == start_rank && target == '.'
                   && b.sq[from + dir] == '.') {
            valid = true;
            dbl = true;
        } else if ((delta == dir - 1 || delta == dir + 1) && file_delta == 1) {
            if (capture && col(target) == (us ^ 1)) {
                valid = true;
            } else if (to == b.ep && target == '.') {
                const int captured_sq = us == WHITE ? to - 8 : to + 8;
                const char expected = us == WHITE ? 'p' : 'P';
                if (onb(captured_sq) && b.sq[captured_sq] == expected) {
                    valid = true;
                    capture = true;
                    ep = true;
                }
            }
        }
        const bool reaches_last_rank = rank_of(to) == promotion_rank;
        if (reaches_last_rank != (promo != 0)) valid = false;
    } else {
        if (promo) return false;
        const bb::Bitboard target_mask = bb::sq_bb(to);
        if (kind == 'n') valid = (bb::KNIGHT_ATTACKS[from] & target_mask) != 0;
        else if (kind == 'b') valid = (bb::bishop_attacks(from, b.occ) & target_mask) != 0;
        else if (kind == 'r') valid = (bb::rook_attacks(from, b.occ) & target_mask) != 0;
        else if (kind == 'q') valid = ((bb::bishop_attacks(from, b.occ)
                                     | bb::rook_attacks(from, b.occ)) & target_mask) != 0;
        else if (kind == 'k') {
            valid = (bb::KING_ATTACKS[from] & target_mask) != 0;
            if (!valid && target == '.') {
                const int them = us ^ 1;
                if (us == WHITE && from == 4 && to == 6 && (b.castling & CWK)
                    && b.sq[5] == '.' && b.sq[6] == '.'
                    && !is_square_attacked_bb(b, 4, them)
                    && !is_square_attacked_bb(b, 5, them)
                    && !is_square_attacked_bb(b, 6, them)) valid = castle = true;
                else if (us == WHITE && from == 4 && to == 2 && (b.castling & CWQ)
                    && b.sq[3] == '.' && b.sq[2] == '.' && b.sq[1] == '.'
                    && !is_square_attacked_bb(b, 4, them)
                    && !is_square_attacked_bb(b, 3, them)
                    && !is_square_attacked_bb(b, 2, them)) valid = castle = true;
                else if (us == BLACK && from == 60 && to == 62 && (b.castling & CBK)
                    && b.sq[61] == '.' && b.sq[62] == '.'
                    && !is_square_attacked_bb(b, 60, them)
                    && !is_square_attacked_bb(b, 61, them)
                    && !is_square_attacked_bb(b, 62, them)) valid = castle = true;
                else if (us == BLACK && from == 60 && to == 58 && (b.castling & CBQ)
                    && b.sq[59] == '.' && b.sq[58] == '.' && b.sq[57] == '.'
                    && !is_square_attacked_bb(b, 60, them)
                    && !is_square_attacked_bb(b, 59, them)
                    && !is_square_attacked_bb(b, 58, them)) valid = castle = true;
            }
        }
    }

    if (!valid) return false;
    if (caponly && !(capture || ep || promo)) return false;
    out = Move{from, to, promo, capture, ep, castle, dbl};
    return true;
}

// Perft helpers — recursive node count. Uses pseudo-legal generator + post-make
// in-check filter (matches the standard "perft" definition: count of leaf
// positions, but legal-move filtered).
std::uint64_t perft_mailbox(Board& b, int depth) {
    if (depth == 0) return 1;
    MoveList moves;
    genpseudo_mailbox(b, moves, false);
    const int us = b.stm;
    std::uint64_t total = 0;
    for (const Move& m : moves) {
        Undo u; makemove(b, m, u);
        if (!incheck(b, us)) total += perft_mailbox(b, depth - 1);
        unmakemove(b, u);
    }
    return total;
}

std::uint64_t perft_bb(Board& b, int depth) {
    if (depth == 0) return 1;
    MoveList moves;
    genpseudo_bb(b, moves, GenMode::All);
    const int us = b.stm;
    std::uint64_t total = 0;
    for (const Move& m : moves) {
        Undo u; makemove(b, m, u);
        if (!incheck(b, us)) total += perft_bb(b, depth - 1);
        unmakemove(b, u);
    }
    return total;
}

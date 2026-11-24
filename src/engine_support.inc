// AndyBot, a UCI chess engine. Copyright (C) 2026 Hyeonung Cho.
//
// AndyBot is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
// later version. It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
// (the LICENSE file) for details. Credits to the engines AndyBot builds on are in README.md.

int mat_eval(const Board& b) {
    int s = 0; for (char p : b.sq) { if (p == '.') continue; int v = pval(p); s += isw(p) ? v : -v; }
    return (b.stm == WHITE) ? s : -s;
}

PositionState parse_position_cmd(const std::vector<std::string>& tok) {
    PositionState pos;
    if (tok.size() < 2) return pos;
    std::size_t i = 1;
    if (tok[i] == "startpos") { pos.fen = "startpos"; ++i; }
    else if (tok[i] == "fen") {
        ++i; std::vector<std::string> parts;
        while (i < tok.size() && tok[i] != "moves") parts.push_back(tok[i++]);
        std::ostringstream oss; for (std::size_t k = 0; k < parts.size(); ++k) { if (k) oss << ' '; oss << parts[k]; }
        pos.fen = trim(oss.str()); if (pos.fen.empty()) pos.fen = "startpos";
    }
    if (i < tok.size() && tok[i] == "moves") { ++i; for (; i < tok.size(); ++i) pos.moves.push_back(tok[i]); }
    return pos;
}

std::optional<int> parse_go_depth(const std::vector<std::string>& tok) {
    for (std::size_t i = 0; i + 1 < tok.size(); ++i) if (tok[i] == "depth") return std::max(1, std::atoi(tok[i + 1].c_str()));
    return std::nullopt;
}

std::optional<int> parse_go_value(const std::vector<std::string>& tok, std::string_view key) {
    for (std::size_t i = 0; i + 1 < tok.size(); ++i) {
        if (tok[i] == key) return std::max(0, std::atoi(tok[i + 1].c_str()));
    }
    return std::nullopt;
}

"""재현 3: Fig. 6 과 식 (C7)  W_h <= 0.8532 τ^3 N  (주기 hexagonal, S2 tiling, 순서 H1=b, H2=r, H3=g)"""
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False
from tile_trotter import hexagonal_periodic, adjacency, W_h_C6, W_eq15, schatten1

# 격자가 진짜 hexagonal 인지 (이분 그래프, 3-정규, 가장 짧은 고리 = 6) 확인
N, bonds = hexagonal_periodic(6)
G = nx.Graph([e for c in "brg" for e in bonds[c]])
print("L=6 격자 확인: 이분 그래프", nx.is_bipartite(G), "| 3-정규", set(dict(G.degree).values()) == {3},
      "| girth", min(len(c) for c in nx.minimum_cycle_basis(G)))

Ls = list(range(4, 25, 2))
rows = []
for L in Ls:
    N, bonds = hexagonal_periodic(L)
    Rs = [adjacency(N, bonds[c]) for c in "brg"]
    w = W_h_C6(Rs)
    w15 = W_eq15(Rs, schatten1)          # 식 (15) 를 쪼개지 않은 (더 촘촘한) 값, 참고용
    rows.append((L, N, w, w / N, w15))
    print(f"L={L:2d}  N={N:5d}  W_h/τ³={w:9.3f}  W_h/(τ³N)={w / N:.4f}   [식15 그대로: {w15:9.3f}]")

ratio = max(r[3] for r in rows)
print(f"\n최대 W_h/(τ³N) = {ratio:.4f}   (논문 식 C7: 0.8532)")

plt.figure(figsize=(5.2, 3.8))
plt.plot([r[1] for r in rows], [r[2] for r in rows], "ko", label="이 재현 (식 C6)")
plt.plot([0, 1200], [0, 0.8532 * 1200], "r--", lw=1, label="논문 식 (C7): 0.8532 N")
plt.xlabel("Number of lattice sites, N"); plt.ylabel(r"$W_h/\tau^3$")
plt.title("Fig. 6 재현: 주기 hexagonal, S2 tiling")
plt.legend(); plt.tight_layout()
plt.savefig("fig6_reproduction.png", dpi=150)
np.savetxt("fig6_data.csv", np.array(rows), delimiter=",", header="L,N,W_h_C6,W_h_over_N,W_h_eq15", comments="")

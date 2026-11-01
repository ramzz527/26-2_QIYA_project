"""재현 2: 주기 hexagonal 격자의 S2 tiling (App. D, Fig. 7-8) + Trotter step 당 게이트 수 (식 27-28, E12-E13)"""
import numpy as np
from tile_trotter import hexagonal_periodic, tiles_from_section, tile_circuit, count

print(" L    N  | 결합(b,r,g) | S2 tile 수(b,r,g) = N/4 ? | 모든 결합 1회 | 3-정규")
for L in [4, 6, 8, 10, 12, 16, 24]:
    N, bonds = hexagonal_periodic(L)
    tiles = {c: tiles_from_section(bonds[c]) for c in "brg"}
    kinds = {k for c in "brg" for k, _ in tiles[c]}
    allb = [tuple(sorted(e)) for c in "brg" for e in bonds[c]]
    deg = np.bincount(np.array(allb).ravel(), minlength=N)
    nt = tuple(len(tiles[c]) for c in "brg")
    print(f"{L:2d} {N:5d}  | {tuple(len(bonds[c]) for c in 'brg')} | {nt} {kinds} {all(x == N // 4 for x in nt)}"
          f" | {len(set(allb)) == len(allb) == 3 * N // 2} | {np.all(deg == 3)}")

# ---- Trotter step 당 비-Clifford 비용 (r 이 클 때: 바깥 t/2 끼리 합쳐짐) ----
# 식 (25)-(26): e^{-iHC t/2} e^{-iHb t/2} e^{-iHr t/2} e^{-iHg t} e^{-iHr t/2} e^{-iHb t/2} e^{-iHC t/2}
# -> step 당: b 2번, r 2번, g 1번, H_C 1번.  스핀 2개.
S2 = count(tile_circuit("S2", 1.0, 1.0))
print("\nTrotter step 당 비-Clifford 비용 (N 의 배수, 스핀 2개)")
for L in [4, 8, 16]:
    N, bonds = hexagonal_periodic(L)
    n_tiles = {c: len(tiles_from_section(bonds[c])) for c in "brg"}
    apps = {"b": 2, "r": 2, "g": 1}
    hop_tiles = 2 * sum(apps[c] * n_tiles[c] for c in "brg")          # x2 스핀
    hop_R, hop_T = hop_tiles * S2["rotations"], hop_tiles * S2["T"]
    HI_R = N                                                           # Z↑Z↓ 항 N 개
    HV_R = 4 * (3 * N // 2)                                            # 결합 3N/2 개 x 스핀쌍 4
    print(f"  L={L:2d}: Hubbard  N_R = {(hop_R + HI_R) / N:.0f}N, N_T = {hop_T / N:.0f}N   (논문 E12-E13: 6N, 10N)")
    print(f"        Extended N_R = {(hop_R + HI_R + HV_R) / N:.0f}N, N_T = {hop_T / N:.0f}N   (논문 식 27-28: 12N, 10N)")

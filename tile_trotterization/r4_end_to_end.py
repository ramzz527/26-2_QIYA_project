"""재현 4: 게이트 수준 tile Trotter step (식 20) 의 실제 오차 vs 논문 bound (식 21: W_tile <= W_SO2 + W_h)

계 A: 벤젠 고리 (hexagon 하나, 6 site, 스핀 포함 12 큐비트) Hubbard / extended Hubbard
      섹션 3개 b, r, g  각 S2 tile 1개 (스핀당)   — 논문 hexagonal 구조의 최소 버전
      JW 순서: (0↑..5↑, 0↓..5↓).  S2 tile 의 지역 순서 (center, leaf, leaf) 는 fswap 으로 맞춤
계 B: 2D SSH 4x4 열린 경계 (spinless 16 큐비트)
      섹션 t1: C4 4개,  섹션 t2: C4 1개 + S1 4개 (tile 섞기)
      행 순서 JW 에서 정사각형은 연속하지 않으므로 fswap 네트워크 필요
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, expm_multiply
from scipy.linalg import expm
from functools import reduce
from tile_trotter import (FermionCircuit, tiles_from_section, apply_gates, count, ssh2d_square, adjacency)

# ---------------- 희소 페르미온 연산자 ----------------
def sp_annihilators(n):
    I = sp.identity(2, format="csr"); Z = sp.diags([1.0, -1.0]).tocsr()
    Q = sp.csr_matrix(np.array([[0, 1], [0, 0]], dtype=float))
    return [reduce(lambda x, y: sp.kron(x, y, format="csr"), [Z] * j + [Q] + [I] * (n - j - 1)) for j in range(n)]

def opnorm(M):
    M = sp.csr_matrix(M)
    return abs(eigsh((M + M.getH()) / 2, k=1, which="LM", return_eigenvectors=False)[0])

def comm(A, B):
    return A @ B - B @ A


# =============================== 계 A: 벤젠 Hubbard ===============================
Ns, tau = 6, 1.0
n = 2 * Ns
a = sp_annihilators(n)
num = [a[j].T @ a[j] for j in range(n)]
I_n = sp.identity(2 ** n, format="csr")
up, dn = (lambda i: i), (lambda i: i + Ns)
ring = [(i, (i + 1) % Ns) for i in range(Ns)]
sections = {"b": [(0, 1), (1, 2)], "r": [(2, 3), (3, 4)], "g": [(4, 5), (5, 0)]}

def hop(bonds):
    H = sp.csr_matrix((2 ** n, 2 ** n))
    for i, j in bonds:
        for s in (up, dn):
            H = H - tau * (a[s(i)].T @ a[s(j)] + a[s(j)].T @ a[s(i)])
    return H

def coulomb(U, V):
    H = sp.csr_matrix((2 ** n, 2 ** n))
    for i in range(Ns):                          # 식 (6): U Σ (n↑-1/2)(n↓-1/2)
        H = H + U * (num[up(i)] - I_n / 2) @ (num[dn(i)] - I_n / 2)
    for i, j in ring:                            # 식 (10): V Σ_<ij> Σ_σσ' (n-1/2)(n-1/2)
        for s1 in (up, dn):
            for s2 in (up, dn):
                H = H + V * (num[s1(i)] - I_n / 2) @ (num[s2(j)] - I_n / 2)
    return H

def build_step(U, V, t):
    """식 (20):  e^{-iHC t/2} e^{-iHb t/2} e^{-iHr t/2} e^{-iHg t} e^{-iHr t/2} e^{-iHb t/2} e^{-iHC t/2}"""
    fc = FermionCircuit(n)
    def HC(dt):   # e^{-i HC dt},  HC = (U/4)ΣZ↑Z↓ + (V/4)Σ ZZ   ->  e^{iθZZ}, θ = -(U/4)dt
        for i in range(Ns):
            fc.zz(up(i), dn(i), -U / 4 * dt)
        if V != 0:
            for i, j in ring:
                for s1 in (up, dn):
                    for s2 in (up, dn):
                        fc.zz(s1(i), s2(j), -V / 4 * dt)
    def section(c, dt):
        for kind, modes in tiles_from_section(sections[c]):
            for s in (up, dn):
                fc.tile(kind, [s(m) for m in modes], tau, dt)
    HC(t / 2); section("b", t / 2); section("r", t / 2); section("g", t)
    section("r", t / 2); section("b", t / 2); HC(t / 2)
    fc.restore()
    return fc

print("=== 계 A: 벤젠 고리, 스핀 포함 12 큐비트 ===")
Hb, Hr, Hg = (hop(sections[c]) for c in "brg")
Hh = Hb + Hr + Hg
for U, V in [(4.0, 0.0), (4.0, 2.0)]:
    HC = coulomb(U, V)
    H = (Hh + HC).toarray()
    # --- 오차 계수 (모두 정확한 다체 연산자 노름) ---
    C = comm(HC, Hh)
    W_SO2 = opnorm(comm(C, Hh)) / 12 + opnorm(comm(C, HC)) / 24                      # 식 (23)
    W_h = 0.0                                                                            # 식 (15), 섹션 b, r, g
    for k, Hk in enumerate([Hb, Hr]):
        rest = [Hr + Hg, Hg][k]
        Ck = comm(Hk, rest)
        W_h += opnorm(comm(Ck, rest)) / 12 + opnorm(comm(Ck, Hk)) / 24
    W_direct = 0.0                                                                       # 식 (15), 항 4개 그대로
    for k, (Hk, rest) in enumerate([(HC, Hh), (Hb, Hr + Hg), (Hr, Hg)]):
        Ck = comm(Hk, rest)
        W_direct += opnorm(comm(Ck, rest)) / 12 + opnorm(comm(Ck, Hk)) / 24
    model = "Hubbard" if V == 0 else "Extended"
    print(f"\n[{model} U={U}, V={V}]  W_SO2={W_SO2:.3f}  W_h={W_h:.3f}  W_SO2+W_h={W_SO2 + W_h:.3f}  "
          f"(식 15를 H_C,b,r,g 4항에 직접 적용: {W_direct:.3f}, 정의상 같음)")
    # 회로 == 섹션 곱 (무작위 상태 3개로 확인)
    def sec_product(t, v):
        for Hx, dt in [(HC, t/2), (Hb, t/2), (Hr, t/2), (Hg, t), (Hr, t/2), (Hb, t/2), (HC, t/2)]:
            v = expm_multiply(-1j * Hx * dt, v)
        return v
    rng = np.random.default_rng(3)
    fc = build_step(U, V, 0.3)
    ok = True
    for _ in range(3):
        v = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
        ok &= np.allclose(apply_gates(v, fc.gates, n), sec_product(0.3, v))
    print("    게이트 수준 회로 == 식 (20) 섹션 곱:", ok)
    # 오차 노름: (N↑, N↓) 섹터별 정확 계산 후 최댓값
    idx = np.arange(2 ** n)
    nup = np.array([bin(i >> Ns).count("1") for i in idx]); ndn = np.array([bin(i & (2**Ns - 1)).count("1") for i in idx])
    sectors = [idx[(nup == p) & (ndn == q)] for p in range(Ns + 1) for q in range(Ns + 1)]
    mats = {k: Mx.tocsr() for k, Mx in {"H": Hh + HC, "C": HC, "b": Hb, "r": Hr, "g": Hg}.items()}
    def err_norm(t):
        e = 0.0
        for s in sectors:
            M = {k: Mx[s][:, s].toarray() for k, Mx in mats.items()}
            E = lambda k, dt: expm(-1j * M[k] * dt)
            Sx = E("C", t/2) @ E("b", t/2) @ E("r", t/2) @ E("g", t) @ E("r", t/2) @ E("b", t/2) @ E("C", t/2)
            e = max(e, np.linalg.norm(E("H", t) - Sx, 2))
        return e
    print("    t      ||U-S||          W_tile t^3 bound   비율   기울기")
    prev = None
    for t in [0.4, 0.2, 0.1, 0.05, 0.025]:
        err = err_norm(t)
        bound = (W_SO2 + W_h) * t ** 3
        slope = "" if prev is None else f"{np.log(prev[1] / err) / np.log(prev[0] / t):.2f}"
        print(f"  {t:5.3f}   {err:.3e}        {bound:.3e}       {err / bound:.3f}   {slope}")
        prev = (t, err)
    c = count(fc.gates)
    print(f"  step 1회 비용: 임의 회전 {c['rotations']}, T {c['T']}, fswap {fc.n_fswap} (JW 순서 맞추기 + 복귀)")


# =============================== 계 B: 2D SSH 4x4 ===============================
print("\n=== 계 B: 2D SSH 4x4 열린 경계 (spinless 16 큐비트), t1=0.5, t2=1.0 ===")
L, t1, t2 = 4, 0.5, 1.0
N, sb = ssh2d_square(L, periodic=False)
tilesA, tilesB = tiles_from_section(sb["t1"]), tiles_from_section(sb["t2"])
print("섹션 t1 tile:", [k for k, _ in tilesA], "| 섹션 t2 tile:", [k for k, _ in tilesB])
hA, hB = -adjacency(N, sb["t1"], t1), -adjacency(N, sb["t2"], t2)

b = sp_annihilators(N)
def quad(h):
    return sum(h[i, j] * b[i].T @ b[j] for i in range(N) for j in range(N) if h[i, j] != 0)
HA, HB = quad(hA), quad(hB)

def ssh_step(t):   # 강한 섹션(t2)을 바깥에:  e^{-iB t/2} e^{-iA t} e^{-iB t/2}
    fc = FermionCircuit(N)
    for tiles, tt, dt in [(tilesB, t2, t / 2), (tilesA, t1, t), (tilesB, t2, t / 2)]:
        for kind, modes in tiles:
            fc.tile(kind, modes, tt, dt)
    fc.restore()
    return fc

rng = np.random.default_rng(7)
psi = rng.normal(size=2 ** N) + 1j * rng.normal(size=2 ** N); psi /= np.linalg.norm(psi)
t = 0.3
fc = ssh_step(t)
out_circ = apply_gates(psi, fc.gates, N)
out_sec = expm_multiply(-1j * HB * t / 2, expm_multiply(-1j * HA * t, expm_multiply(-1j * HB * t / 2, psi)))
out_exact = expm_multiply(-1j * (HA + HB) * t, psi)
print(f"회로 == 섹션 곱 e^(-iBt/2)e^(-iAt)e^(-iBt/2):  {np.allclose(out_circ, out_sec)}  "
      f"(fswap {fc.n_fswap}개로 행 순서 JW 에서 정사각형 tile 처리)")

# 단일입자 공식으로 정확한 Trotter 오차 노름과 W
def mb_norm(M):
    ev = np.linalg.eigvalsh((M + M.conj().T) / 2); return max(ev[ev > 0].sum(), -ev[ev < 0].sum())
Cs = hB @ hA - hA @ hB
W = mb_norm(Cs @ hA - hA @ Cs) / 12 + mb_norm(Cs @ hB - hB @ Cs) / 24
u_exact = expm(-1j * (hA + hB) * t)
u_trot = expm(-1j * hB * t / 2) @ expm(-1j * hA * t) @ expm(-1j * hB * t / 2)
th = np.angle(np.linalg.eigvals(u_exact.conj().T @ u_trot))
phi = max(th[th > 0].sum(), -th[th < 0].sum())                 # 다체 위상의 최대 크기
err_exact = abs(1 - np.exp(1j * phi))
print(f"t={t}:  상태 오차 ||(U-S)ψ|| = {np.linalg.norm(out_circ - out_exact):.3e}  <=  "
      f"정확한 ||U-S|| = {err_exact:.3e}  <=  W t^3 = {W * t ** 3:.3e}  (W={W:.4f})")

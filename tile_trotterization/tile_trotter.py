"""
tile_trotter.py  —  Bay-Smidt et al., PRX Quantum 6, 030348 (2025) "tile Trotterization" 재현용 라이브러리

구성
  1. 게이트 / 상태벡터 시뮬레이터 (Clifford + T + Rz)
  2. F 게이트(T 2개), fswap, XX/YY 회전
  3. tile 회로: S1, S2, C4, S4  (논문 App. A, 식 A20, A29, A38, A47)
  4. FermionCircuit: JW 순서를 추적하며 fswap 으로 tile 을 인접하게 만들어 적용
  5. 격자: 주기 hexagonal (App. D, Fig. 7) + S2 tiling,  2D SSH (정사각, C4 tiling)
  6. 오차 계수: W_h (식 C6), 일반 식 (15) 의 정확 계산

규약
  - |1> = 점유,  a_j = Z_0 ... Z_{j-1} Q_j,  Q = |0><1|,  n = (I - Z)/2
  - 큐비트 0 이 kron 의 가장 왼쪽
  - Rz(θ) = diag(e^{-iθ/2}, e^{iθ/2})
"""
from __future__ import annotations
import numpy as np
from functools import reduce
from dataclasses import dataclass, field

# ----------------------------------------------------------------------------------------------
# 1. 게이트와 시뮬레이터
# ----------------------------------------------------------------------------------------------
_s = 1 / np.sqrt(2)
G1 = {
    "H": np.array([[1, 1], [1, -1]], complex) * _s,
    "S": np.diag([1, 1j]).astype(complex),
    "Sd": np.diag([1, -1j]).astype(complex),
    "T": np.diag([1, np.exp(1j * np.pi / 4)]),
    "Td": np.diag([1, np.exp(-1j * np.pi / 4)]),
    "X": np.array([[0, 1], [1, 0]], complex),
    "Z": np.diag([1, -1]).astype(complex),
}
G2 = {
    "CNOT": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], complex),  # (control, target)
    "CZ": np.diag([1, 1, 1, -1]).astype(complex),
    "SWAP": np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], complex),
}
CLIFFORD = {"H", "S", "Sd", "X", "Z", "CNOT", "CZ", "SWAP"}
DAGGER = {"H": "H", "S": "Sd", "Sd": "S", "T": "Td", "Td": "T", "X": "X", "Z": "Z",
          "CNOT": "CNOT", "CZ": "CZ", "SWAP": "SWAP", "Rz": "Rz"}


@dataclass(frozen=True)
class Gate:
    name: str
    qubits: tuple
    theta: float = 0.0          # Rz 각도 (Rz 만 사용)

    def matrix(self):
        if self.name == "Rz":
            return np.diag([np.exp(-1j * self.theta / 2), np.exp(1j * self.theta / 2)])
        return G1[self.name] if len(self.qubits) == 1 else G2[self.name]

    def dagger(self):
        return Gate(DAGGER[self.name], self.qubits, -self.theta)

    def shifted(self, k):
        return Gate(self.name, tuple(q + k for q in self.qubits), self.theta)


def dagger(gates):
    return [g.dagger() for g in reversed(gates)]


def apply_gates(state, gates, n):
    """state: (2^n,) 또는 (2^n, batch).  gates 를 시간 순서대로 적용."""
    batch = state.ndim == 2
    psi = state.reshape([2] * n + ([state.shape[1]] if batch else []))
    for g in gates:
        U = g.matrix()
        if len(g.qubits) == 1:
            q = g.qubits[0]
            psi = np.moveaxis(np.tensordot(U, psi, axes=([1], [q])), 0, q)
        else:
            q0, q1 = g.qubits
            U4 = U.reshape(2, 2, 2, 2)
            psi = np.tensordot(U4, psi, axes=([2, 3], [q0, q1]))
            psi = np.moveaxis(psi, [0, 1], [q0, q1])
    return psi.reshape(state.shape)


def unitary(gates, n):
    return apply_gates(np.eye(2 ** n, dtype=complex), gates, n)


def count(gates):
    """논문 Table III 방식의 비용: 임의 회전(Rz), T(T, T†), CNOT, fswap 은 별도 집계"""
    c = {"rotations": 0, "T": 0, "CNOT": 0, "clifford_other": 0}
    for g in gates:
        if g.name == "Rz":
            c["rotations"] += 1
        elif g.name in ("T", "Td"):
            c["T"] += 1
        elif g.name == "CNOT":
            c["CNOT"] += 1
        else:
            c["clifford_other"] += 1
    return c


# ----------------------------------------------------------------------------------------------
# 2. 기본 블록
# ----------------------------------------------------------------------------------------------
def F_gate(q0, q1, sum_to="right"):
    """논문 F_ij (식 A7-A9, Fig. 5b). 인접 모드 (q0, q1) 를 합/차 모드로 회전. T 게이트 2개.
       sum_to='right': (a0+a1)/√2 가 q1 에,  'left': q0 에 놓인다.
       분해:  F = (Z⊗I) CNOT(1→0) [ (I⊗A†) CZ (I⊗A) ] CNOT(1→0) CZ̄ ,  A = H T H S,  CZ̄ = (X⊗I)CZ(X⊗I)"""
    if sum_to == "left":   # SWAP 으로 감싼 버전 (SWAP 은 Clifford)
        return [Gate("SWAP", (q0, q1))] + F_gate(q0, q1, "right") + [Gate("SWAP", (q0, q1))]
    A = [Gate("S", (q1,)), Gate("H", (q1,)), Gate("T", (q1,)), Gate("H", (q1,))]   # 시간 순서: S, H, T, H
    gates = [Gate("X", (q0,)), Gate("CZ", (q0, q1)), Gate("X", (q0,)),             # CZ̄
             Gate("CNOT", (q1, q0))]
    gates += A + [Gate("CZ", (q0, q1))] + dagger(A)
    gates += [Gate("CNOT", (q1, q0)), Gate("Z", (q0,))]
    return gates


def fswap(q0, q1):
    """fermionic SWAP = SWAP · CZ (식 A10). Clifford"""
    return [Gate("CZ", (q0, q1)), Gate("SWAP", (q0, q1))]


def exp_XX(q0, q1, theta):
    """e^{iθ X⊗X}: 임의 회전 1개"""
    return ([Gate("H", (q0,)), Gate("H", (q1,)), Gate("CNOT", (q0, q1)), Gate("Rz", (q1,), -2 * theta),
             Gate("CNOT", (q0, q1)), Gate("H", (q0,)), Gate("H", (q1,))])


def exp_YY(q0, q1, theta):
    """e^{iθ Y⊗Y} = (S⊗S) e^{iθXX} (S†⊗S†): 임의 회전 1개"""
    return ([Gate("Sd", (q0,)), Gate("Sd", (q1,))] + exp_XX(q0, q1, theta) +
            [Gate("S", (q0,)), Gate("S", (q1,))])


def bond(q0, q1, g, t):
    """H = -g (a0† a1 + h.c.) (인접 모드) 의 e^{-iHt} = e^{i g t/2 XX} e^{i g t/2 YY}  (식 A20)"""
    return exp_XX(q0, q1, g * t / 2) + exp_YY(q0, q1, g * t / 2)


def exp_ZZ(q0, q1, theta):
    """e^{iθ Z⊗Z}: 임의 회전 1개 (대각 항이라 JW 인접성 불필요)"""
    return [Gate("CNOT", (q0, q1)), Gate("Rz", (q1,), -2 * theta), Gate("CNOT", (q0, q1))]


# ----------------------------------------------------------------------------------------------
# 3. Tile 회로 (지역 큐비트 0..k-1, 지역 JW 순서 = 아래 'order' 설명)
# ----------------------------------------------------------------------------------------------
#  S1 : (a, b)
#  S2 : (center, leaf, leaf)                         R = K_{1,2}
#  C4 : (g1, g1', g2, g2')  두 대각선 쌍              R = K_{2,2}
#  S4 : (center, leaf, leaf, leaf, leaf)             R = K_{1,4}
TILE_SIZE = {"S1": 2, "S2": 3, "C4": 4, "S4": 5}


def tile_adjacency(kind):
    k = TILE_SIZE[kind]
    R = np.zeros((k, k))
    if kind == "C4":
        R[np.ix_([0, 1], [2, 3])] = 1
        R[np.ix_([2, 3], [0, 1])] = 1
    else:
        R[0, 1:] = R[1:, 0] = 1
    return R


def tile_circuit(kind, tau, t):
    """e^{-i H_tile t},  H_tile = -τ Σ R_ij a_i† a_j  (논문 식 A1-A2)
       공통 구조: V 로 '합 모드 두 개' 를 인접 큐비트에 모은 뒤 세기 λτ 인 결합 하나(S1)로 진화, V† 로 복귀"""
    if kind == "S1":
        return bond(0, 1, tau, t)
    if kind == "S2":          # 잎 (1,2) 를 합 -> q1,  결합 (0,1) 세기 √2 τ
        V = F_gate(1, 2, "left")
        return V + bond(0, 1, np.sqrt(2) * tau, t) + dagger(V)
    if kind == "C4":          # 쌍 (0,1) 합 -> q1, 쌍 (2,3) 합 -> q2, 결합 (1,2) 세기 2τ   (식 A38)
        V = F_gate(0, 1, "right") + F_gate(2, 3, "left")
        return V + bond(1, 2, 2 * tau, t) + dagger(V)
    if kind == "S4":          # 잎 4개를 F 3번으로 합 -> q2, fswap 으로 q1 로, 결합 (0,1) 세기 2τ  (식 A47)
        V = F_gate(1, 2, "right") + F_gate(3, 4, "left") + F_gate(2, 3, "left") + fswap(1, 2)
        return V + bond(0, 1, 2 * tau, t) + dagger(V)
    raise ValueError(kind)


# ----------------------------------------------------------------------------------------------
# 4. 페르미온 연산자 (검증용, 작은 n)
# ----------------------------------------------------------------------------------------------
def annihilators(n):
    I, Z = np.eye(2), np.diag([1.0, -1.0])
    Q = np.array([[0, 1], [0, 0]], dtype=float)
    return [reduce(np.kron, [Z] * j + [Q] + [I] * (n - j - 1)) for j in range(n)]


def quadratic(h, a):
    """Σ h_ij a_i† a_j  (dense)"""
    n = len(a)
    return sum(h[i, j] * a[i].T @ a[j] for i in range(n) for j in range(n) if h[i, j] != 0)


# ----------------------------------------------------------------------------------------------
# 5. JW 순서를 추적하는 페르미온 회로
# ----------------------------------------------------------------------------------------------
@dataclass
class FermionCircuit:
    n: int
    gates: list = field(default_factory=list)

    def __post_init__(self):
        self.order = list(range(self.n))           # 위치 -> 모드
        self.n_fswap = 0

    def pos(self, m):
        return self.order.index(m)

    def _fswap(self, p):
        self.gates += fswap(p, p + 1)
        self.order[p], self.order[p + 1] = self.order[p + 1], self.order[p]
        self.n_fswap += 1

    def make_contiguous(self, modes):
        """modes 를 이 순서대로 연속 위치에 놓는다 (인접 fswap 만 사용). 시작 위치 반환"""
        start = min(self.pos(m) for m in modes)
        for k, m in enumerate(modes):
            while self.pos(m) > start + k:
                self._fswap(self.pos(m) - 1)
        return start

    def tile(self, kind, modes, tau, t):
        start = self.make_contiguous(modes)
        self.gates += [g.shifted(start) for g in tile_circuit(kind, tau, t)]

    def zz(self, m1, m2, theta):
        self.gates += exp_ZZ(self.pos(m1), self.pos(m2), theta)

    def restore(self):
        """JW 순서를 0..n-1 로 되돌림 (버블 정렬)"""
        for i in range(self.n):
            for p in range(self.n - 1 - i):
                if self.order[p] > self.order[p + 1]:
                    self._fswap(p)


# ----------------------------------------------------------------------------------------------
# 6. 격자와 tiling
# ----------------------------------------------------------------------------------------------
def hexagonal_periodic(L):
    """논문 App. D / Fig. 7(b) 의 주기 hexagonal 격자 (Lx = Ly = L, N = 2L^2) 와 3색 S2 tiling.
       사이트 = (사슬 c, 위치 p), c = 0..L-1 (아래->위), p = 0..2L-1 (지그재그 사슬 따라)
       - 사슬 안 결합 (p, p+1):  c 짝수 -> [g, r, r, b][p%4],  c 홀수 -> [r, g, b, r][p%4]
       - 사슬 사이 결합 (c, p 홀수) - (c+1, p-1):  c 짝수 -> g,  c 홀수 -> b
       반환: N, {색: [결합 (i,j)]}"""
    assert L % 2 == 0 and L >= 2
    site = lambda c, p: (c % L) * 2 * L + (p % (2 * L))
    pat_even, pat_odd = "grrb", "rgbr"
    bonds = {"b": [], "r": [], "g": []}
    for c in range(L):
        for p in range(2 * L):
            col = (pat_even if c % 2 == 0 else pat_odd)[p % 4]
            bonds[col].append((site(c, p), site(c, p + 1)))
            if p % 2 == 1:
                bonds["g" if c % 2 == 0 else "b"].append((site(c, p), site(c + 1, p - 1)))
    return 2 * L * L, bonds


def ssh2d_square(L, periodic=False):
    """2D SSH (L x L, site = x + L*y). 반환: N, {'t1': 결합들, 't2': 결합들}"""
    bonds = {"t1": [], "t2": []}
    for y in range(L):
        for x in range(L):
            for dx, dy in [(1, 0), (0, 1)]:
                if not periodic and (x + dx >= L or y + dy >= L):
                    continue
                coord = x if dx else y
                i, j = x + L * y, (x + dx) % L + L * ((y + dy) % L)
                bonds["t1" if coord % 2 == 0 else "t2"].append((i, j))
    return L * L, bonds


def tiles_from_section(bonds):
    """섹션의 결합들을 tile 로 분해. 연결 성분이 S1 / S2(경로 3) / C4(사각형) / S4(별) 이어야 함.
       반환: [(kind, 지역 JW 순서의 모드 리스트)]"""
    from collections import defaultdict
    adj = defaultdict(set)
    for i, j in bonds:
        assert j not in adj[i], "중복 결합"
        adj[i].add(j); adj[j].add(i)
    seen, tiles = set(), []
    for v in list(adj):
        if v in seen:
            continue
        comp, stack = set(), [v]
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u); stack += list(adj[u])
        seen |= comp
        nb = sum(len(adj[u]) for u in comp) // 2
        deg = sorted(len(adj[u]) for u in comp)
        if len(comp) == 2:
            tiles.append(("S1", sorted(comp)))
        elif len(comp) == 3 and deg == [1, 1, 2]:
            c = next(u for u in comp if len(adj[u]) == 2)
            tiles.append(("S2", [c] + sorted(adj[c])))
        elif len(comp) == 4 and nb == 4 and deg == [2, 2, 2, 2]:
            a = min(comp); opp = next(u for u in comp if u != a and u not in adj[a])
            rest = sorted(comp - {a, opp})
            tiles.append(("C4", [a, opp] + rest))           # 대각선 쌍끼리
        elif len(comp) == 5 and deg == [1, 1, 1, 1, 4]:
            c = next(u for u in comp if len(adj[u]) == 4)
            tiles.append(("S4", [c] + sorted(adj[c])))
        else:
            raise ValueError(f"tile 로 분해 불가한 성분: {sorted(comp)}")
    return tiles


def adjacency(N, bonds, weight=1.0):
    R = np.zeros((N, N))
    for i, j in bonds:
        R[i, j] = R[j, i] = weight
    return R


# ----------------------------------------------------------------------------------------------
# 7. Trotter 오차 계수
# ----------------------------------------------------------------------------------------------
def comm(A, B):
    return A @ B - B @ A


def schatten1(M):
    return np.abs(np.linalg.eigvalsh((M + M.conj().T) / 2)).sum()


def W_h_C6(Rs, tau=1.0):
    """논문 식 (C6): 섹션 3개 (순서 H1, H2, H3), 두 스핀 포함, Schatten-1 노름"""
    R1, R2, R3 = Rs
    n = schatten1
    t12 = (n(comm(comm(R1, R2), R2)) + n(comm(comm(R1, R2), R3)) + n(comm(comm(R1, R3), R2))
           + n(comm(comm(R1, R3), R3)) + n(comm(comm(R2, R3), R3)))
    t24 = n(comm(comm(R1, R2), R1)) + n(comm(comm(R1, R3), R1)) + n(comm(comm(R2, R3), R2))
    return tau ** 3 * (t12 / 12 + t24 / 24)


def W_eq15(Hs, norm):
    """논문 식 (15) 그대로 (항을 쪼개지 않은 버전). Hs = [H1, ..., Hm] (적용 순서), norm = 노름 함수"""
    m = len(Hs)
    w = 0.0
    for b in range(m - 1):
        rest = sum(Hs[b + 1:])
        C = comm(Hs[b], rest)
        w += norm(comm(C, rest)) / 12 + norm(comm(C, Hs[b])) / 24
    return w

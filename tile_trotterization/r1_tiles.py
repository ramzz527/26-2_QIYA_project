"""재현 1: tile 회로 (논문 App. A) 검증 + Table I / III 의 비-Clifford 비용"""
import numpy as np
from scipy.linalg import expm
from tile_trotter import (F_gate, unitary, tile_circuit, tile_adjacency, annihilators, quadratic,
                          count, TILE_SIZE)

s = 1 / np.sqrt(2)
F_target = np.array([[1, 0, 0, 0], [0, s, s, 0], [0, s, -s, 0], [0, 0, 0, -1]])
Fg = F_gate(0, 1)
print("F 게이트 (식 A9 행렬, 합 모드 -> 오른쪽 큐비트) Clifford+T 분해 일치:",
      np.allclose(unitary(Fg, 2), F_target), "| 비용:", count(Fg))

paper = {"S1": (2, 0), "S2": (2, 4), "C4": (2, 8), "S4": (2, 12)}   # Table I: (임의 회전, T)
paper_fswap = {"S1": 0, "S2": 0, "C4": 0, "S4": 2}
print("\n tile | 회로 = e^{-iHt} | 단일입자 고유값            | 회전 T  (논문)   | CNOT")
rng = np.random.default_rng(1)
for kind in ["S1", "S2", "C4", "S4"]:
    k = TILE_SIZE[kind]
    R = tile_adjacency(kind)
    a = annihilators(k)
    ok = True
    for _ in range(3):
        tau, t = rng.uniform(0.2, 1.5, 2)
        ok &= np.allclose(unitary(tile_circuit(kind, tau, t), k), expm(-1j * quadratic(-tau * R, a) * t))
    c = count(tile_circuit(kind, 1.0, 1.0))
    ev = np.round(np.linalg.eigvalsh(R), 3)
    print(f"  {kind} | {str(ok):5}          | {str(ev):26}| {c['rotations']}   {c['T']:2}  {paper[kind]} | {c['CNOT']}")
print("\n(fswap: S4 회로는 V 에 1개, V† 에 1개 = 2개, 논문 Table III 과 동일)")

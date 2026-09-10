#!/usr/bin/env python3
import json
import math
import re
import socket
import sys
from fractions import Fraction
from typing import List, Tuple, Optional


# ------------------------------------------------------------
# Small, dependency-free LLL for tiny dimensions.
# Enough for this challenge (5 samples => 5x5 lattice).
# ------------------------------------------------------------

def lll_reduction(B: List[List[int]], delta: Fraction = Fraction(3, 4)) -> List[List[int]]:
    B = [list(map(int, row)) for row in B]
    n = len(B)
    m = len(B[0])

    def gram_schmidt(basis: List[List[int]]):
        bstar = [[Fraction(0) for _ in range(m)] for __ in range(n)]
        mu = [[Fraction(0) for _ in range(n)] for __ in range(n)]
        bnorm = [Fraction(0) for _ in range(n)]
        for i in range(n):
            v = [Fraction(x) for x in basis[i]]
            for j in range(i):
                if bnorm[j] == 0:
                    mu[i][j] = Fraction(0)
                else:
                    mu[i][j] = sum(Fraction(basis[i][k]) * bstar[j][k] for k in range(m)) / bnorm[j]
                for k in range(m):
                    v[k] -= mu[i][j] * bstar[j][k]
            bstar[i] = v
            bnorm[i] = sum(x * x for x in v)
        return mu, bstar, bnorm

    mu, bstar, bnorm = gram_schmidt(B)
    k = 1
    while k < n:
        for j in range(k - 1, -1, -1):
            q = round(mu[k][j])
            if q:
                B[k] = [B[k][i] - q * B[j][i] for i in range(m)]
                mu, bstar, bnorm = gram_schmidt(B)
        if bnorm[k] >= (delta - mu[k][k - 1] * mu[k][k - 1]) * bnorm[k - 1]:
            k += 1
        else:
            B[k], B[k - 1] = B[k - 1], B[k]
            mu, bstar, bnorm = gram_schmidt(B)
            k = max(k - 1, 1)
    return B


# ------------------------------------------------------------
# Step 1: Recover the hidden 384-bit seed from leaked values
# x_i = a_i * seed + b_i, with a_i ~ 384 bits and b_i ~ 200 bits.
# ------------------------------------------------------------

def recover_seed(xs: List[int], noise_bits: int = 200) -> int:
    if len(xs) < 2:
        raise ValueError("need at least 2 samples")

    x0 = xs[0]
    M = 1 << noise_bits
    t = len(xs) - 1

    lattice = []
    lattice.append([M] + xs[1:])
    for i in range(t):
        row = [0] * (t + 1)
        row[i + 1] = -x0
        lattice.append(row)

    red = lll_reduction(lattice)

    candidates = []
    for row in red:
        first = int(row[0])
        if first == 0 or first % M != 0:
            continue
        q0 = abs(first) // M
        if q0 == 0:
            continue
        seed = x0 // q0
        if seed <= 0:
            continue
        # Validate: all residues must be small.
        residues = [x % seed for x in xs]
        if all(r.bit_length() <= noise_bits + 2 for r in residues):
            score = max(r.bit_length() for r in residues)
            candidates.append((score, seed, q0, residues))

    if not candidates:
        raise ValueError("failed to recover seed")

    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]


# ------------------------------------------------------------
# Step 2: Factor the original modulus once the seed is known.
# If p = a*s + b and q = c*s + d, then:
#   n = ac*s^2 + (ad+bc)*s + bd
# with b,d about 192 bits.
# We brute-force tiny carries and use a perfect-square discriminant.
# ------------------------------------------------------------

def factor_with_seed(n: int, s: int) -> Tuple[int, int]:
    C = n % s
    V = (n // s) % s
    A0 = (n // s) // s

    for t in (0, 1):
        D = C + t * s  # bd
        for u in range(5):
            A = A0 - u  # ac
            if A <= 0:
                continue
            B = V + u * s - t  # ad + bc
            Delta = B * B - 4 * A * D
            if Delta < 0:
                continue
            sq = math.isqrt(Delta)
            if sq * sq != Delta:
                continue
            if ((B + sq) & 1) or ((B - sq) & 1):
                continue

            X = (B + sq) // 2
            Y = (B - sq) // 2

            # Try both assignments: (ad, bc) = (X, Y) or (Y, X)
            for ad, bc in ((X, Y), (Y, X)):
                a = math.gcd(A, ad)
                if a <= 0 or ad % a != 0:
                    continue
                c = A // a
                d = ad // a
                if d <= 0 or D % d != 0:
                    continue
                b = D // d
                p = a * s + b
                q = c * s + d
                if p * q == n:
                    return int(p), int(q)

    raise ValueError("failed to factor n with recovered seed")


# ------------------------------------------------------------
# Networking / parsing
# ------------------------------------------------------------

def recv_interactive_transcript(host: str, port: int, rounds: int = 5, init_timeout: float = 180.0, round_timeout: float = 45.0) -> str:
    with socket.create_connection((host, port), timeout=15.0) as sock:
        chunks = []
        buf = ""
        sock.settimeout(init_timeout)

        # Initial generation is very slow on this challenge.
        while True:
            try:
                data = sock.recv(4096)
            except socket.timeout:
                break
            except (ConnectionResetError, BrokenPipeError, OSError):
                break
            if not data:
                break
            chunks.append(data)
            buf = b"".join(chunks).decode(errors="replace")
            objs = extract_json_objects(buf)
            if len(objs) >= 1:
                break

        for _ in range(rounds):
            sock.sendall(b"0\n")
            sock.settimeout(round_timeout)
            while True:
                try:
                    data = sock.recv(4096)
                except socket.timeout:
                    break
                except (ConnectionResetError, BrokenPipeError, OSError):
                    break
                if not data:
                    break
                chunks.append(data)
                buf = b"".join(chunks).decode(errors="replace")
                objs = extract_json_objects(buf)
                if len(objs) >= 1 + (_ + 1):
                    break

    return b"".join(chunks).decode(errors="replace")


def extract_json_objects(blob: str) -> List[dict]:
    objs = []
    for m in re.finditer(r"\{[^{}]*\}", blob, flags=re.S):
        try:
            objs.append(json.loads(m.group(0)))
        except Exception:
            pass
    return objs


def long_to_bytes(n: int) -> bytes:
    if n == 0:
        return b"\x00"
    return n.to_bytes((n.bit_length() + 7) // 8, "big")


def solve_from_values(c: int, e: int, n: int, leaked_seeds: List[int]) -> bytes:
    seed = recover_seed(leaked_seeds)
    p, q = factor_with_seed(n, seed)
    phi = (p - 1) * (q - 1)
    d = pow(e, -1, phi)
    m = pow(c, d, n)
    return long_to_bytes(m)


def solve_remote(host: str, port: int) -> bytes:
    transcript = recv_interactive_transcript(host, port, rounds=5, init_timeout=180.0, round_timeout=45.0)
    print("[+] Raw transcript:")
    print(transcript)

    objs = extract_json_objects(transcript)
    if len(objs) < 6:
        if not transcript.strip():
            raise ValueError("instance returned no data")
        raise ValueError(f"expected at least 6 JSON objects, got {len(objs)}")

    first = objs[0]
    c = int(first["c"])
    e = int(first["e"])
    n = int(first["n"])
    leaked = [int(o["seed"]) for o in objs[1:6]]

    print(f"[+] Got c/e/n and {len(leaked)} leaked seed values")
    pt = solve_from_values(c, e, n, leaked)
    return pt


# ------------------------------------------------------------
# Optional self-test on smaller parameters.
# ------------------------------------------------------------

def _rand_odd(bits: int) -> int:
    import random
    return (1 << (bits - 1)) | random.getrandbits(bits - 1) | 1


def selftest() -> None:
    import random
    from sympy import isprime, randprime

    def genprime_from_seed(seed: int, half_bits: int) -> Tuple[int, int, int]:
        while True:
            a = randprime(1 << (half_bits - 1), 1 << half_bits)
            b = randprime(1 << (half_bits - 1), 1 << half_bits)
            p = a * seed + b
            if isprime(p):
                return int(p), int(a), int(b)

    import random
    seed = ((1 << 59) | random.getrandbits(59)) & ~1  # force even seed, mirroring the only workable server instances
    p, a, b = genprime_from_seed(seed, 30)
    q, c, d = genprime_from_seed(seed, 30)
    n = p * q
    e = 65537
    msg = b"test_flag"
    m = int.from_bytes(msg, "big")
    ct = pow(m, e, n)

    leaks = []
    for _ in range(5):
        A = randprime(1 << 59, 1 << 60)
        B = randprime(1 << 29, 1 << 30)
        leaks.append(int(A * seed + B))

    rec_seed = recover_seed(leaks, noise_bits=30)
    rp, rq = factor_with_seed(n, rec_seed)
    phi = (rp - 1) * (rq - 1)
    dec = pow(ct, pow(e, -1, phi), n)
    got = long_to_bytes(dec)

    assert rec_seed == seed
    assert rp * rq == n
    assert got == msg
    print("[+] selftest passed")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        selftest()
        sys.exit(0)

    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} HOST PORT")
        print(f"or:    {sys.argv[0]} --selftest")
        sys.exit(1)

    host = sys.argv[1]
    port = int(sys.argv[2])
    pt = solve_remote(host, port)
    print("[+] plaintext bytes:", pt)
    try:
        print("[+] plaintext text:", pt.decode())
    except Exception:
        pass

# Define imports
import numpy as np
import random
from math import gcd

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT
from qiskit.primitives import Sampler, BackendSampler

from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


def query_qc(circuit):
    service = QiskitRuntimeService()
    backend = service.least_busy(simulator=False, operational=True)
    sampler = BackendSampler(backend)
    print("sent to quantum computer")
    return sampler.run(circuit)


modular_exponentiation_bits = 4


def initialize_circuit(circuit, n):
    circuit.h(range(n))
    circuit.x(n)


def modular_exponentation(circuit, n, a):
    for x in range(n):
        circuit.append(
            modular_exponentiation_helper(a, 2**x),
            [x] + list(range(n, n + modular_exponentiation_bits)),
        )


modular_expansion_primes = [2, 7, 8, 11, 13]


# # We are running modular exponentation for 4 qubits on 15, creating a prebuilt gate
# Explanation: https://quantumcomputing.stackexchange.com/questions/29647/trying-to-construct-modular-exponentiation-gate-in-qiskit
def modular_exponentiation_helper(a, x):
    circuit = QuantumCircuit(4)
    # if 3 or 5, then we would have a gcd of 15, and have our solution. Therefore our possible inputs to this function are
    if a not in modular_expansion_primes:
        raise ValueError(
            "Bad a value, must be 2, 7, 8, 11, or 13"
        )  #  -1 # We got here when we should not have

    for iteration in range(x):
        if a in [2, 13]:
            circuit.swap(0, 1)
            circuit.swap(1, 2)
            circuit.swap(2, 3)
        if a in [7, 8]:
            circuit.swap(2, 3)
            circuit.swap(1, 2)
            circuit.swap(0, 1)
        if a == 11:
            circuit.swap(1, 3)
            circuit.swap(0, 2)
        if a in [7, 11, 13]:
            # circuit.x(range(4))
            for q in range(4):
                circuit.x(q)

    circuit = circuit.to_gate()
    circuit.name = "%i^%i mod 15" % (a, x)  # Hard coded 15
    control_circuit = circuit.control()
    return control_circuit


def inverse_qft(circuit, measurement_qubits):
    circuit.append(
        QFT(len(measurement_qubits), do_swaps=True).inverse(), measurement_qubits
    )


def quantum_find_order(a, n):
    # Build the quantum circuit
    qc = QuantumCircuit(
        n + modular_exponentiation_bits, n
    )  # We want n target and n measurement qubits, plus n classical bits

    initialize_circuit(qc, n)

    modular_exponentation(qc, n, a)

    inverse_qft(qc, range(n))

    qc.measure(
        list(range(n)), list(range(n))
    )  # Measure a quantum bit (qubit) in the Z basis into a classical bit (cbit).

    # Run on a quantum computer
    job = query_qc(qc)
    result = job.result()
    print("finished")

    # Need to add a noisy model, or modify the quantum computer code so it will properly run

    # This return should return the number of find order, not the quantum computer. Will need to do some processing
    # return qc
    distribution = result.quasi_dists[0]
    r_values = {}
    max_n = 2**n
    for num in distribution.keys():
        denom = gcd(num, max_n)
        r = max_n // denom
        if r not in r_values:
            r_values[r] = 0

        r_values[r] += 1

    found_r = max(list(r_values.items()), key=lambda x: x[1])[0]
    # print(f"r = {found_r}")
    return found_r


# A is a, n is number of qubits for target
def find_order(a, n):
    # Build the quantum circuit
    qc = QuantumCircuit(
        n + modular_exponentiation_bits, n
    )  # We want n target and n measurement qubits, plus n classical bits

    initialize_circuit(qc, n)

    modular_exponentation(qc, n, a)

    inverse_qft(qc, range(n))

    qc.measure(
        list(range(n)), list(range(n))
    )  # Measure a quantum bit (qubit) in the Z basis into a classical bit (cbit).

    # Get the results with a simulator

    # Sampler (simulator) option
    sampler = Sampler()
    job = sampler.run(qc)
    result = job.result()

    # This return should return the number of find order, not the quantum computer. Will need to do some processing
    # return qc
    distribution = result.quasi_dists[0]
    r_values = {}
    max_n = 2**n
    for num in distribution.keys():
        denom = gcd(num, max_n)
        r = max_n // denom
        if r not in r_values:
            r_values[r] = 0

        r_values[r] += 1

    found_r = max(list(r_values.items()), key=lambda x: x[1])[0]
    # print(f"r = {found_r}")
    return found_r


shors_runs = 10


def quantum_shors(n):
    for i in range(shors_runs):
        a = random.choice(modular_expansion_primes)
        d = gcd(a, n)
        if d > 1:
            return d

        r = quantum_find_order(
            a, n
        )  # You may hard code find_order to have an a of 7, because find_order currently fails for certain values of a

        x = (a**r) % n
        d = gcd(int(x), n)
        if d > 1:
            return d

    raise RuntimeError("failed to find factor")


def shors(n):
    for i in range(shors_runs):
        a = random.choice(modular_expansion_primes)
        d = gcd(a, n)
        if d > 1:
            return d

        r = find_order(
            a, n
        )  # You may hard code find_order to have an a of 7, because find_order currently fails for certain values of a

        x = (a**r) % n
        d = gcd(int(x), n)
        if d > 1:
            return d

    raise RuntimeError("failed to find factor")


# Helper function to help check if n is a power of a prime
# Return [] if not power of a prime, and [p] * k if power of a prime p
primes = [
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
]  # There are more prime numbers but this is good enough for us for now


def get_small_factors(n: int, factors: list):
    if n <= 0:
        raise RuntimeError("Cannot check if pk on number <= 0")

    if n == 1:
        return n

    for num in primes:
        while n % num == 0:
            n //= num
            factors.append(num)

    return n


# Return a list of the prime factors of n
def integer_factorization(n):
    factors = []
    while n > 1:
        n = get_small_factors(n, factors)
        if n == 1:
            return factors

        d = shors(n)
        while n % d == 0:
            n //= d
            factors.append(d)

    return factors


# Return a list of the prime factors of n
def shor_only_integer_factorization(n):
    factors = []
    while n > 1:
        d = shors(n)
        while n % d == 0:
            n //= d
            factors.append(d)

    return factors


def quantum_shor_only_integer_factorization(n):
    factors = []
    while n > 1:
        d = quantum_shors(n)
        while n % d == 0:
            n //= d
            factors.append(d)

    return factors


lst = []
get_small_factors(1, lst)
print(lst)
lst.clear()

get_small_factors(3, lst)
print(lst)
lst.clear()

get_small_factors(13, lst)
print(lst)
lst.clear()

get_small_factors(24, lst)
print(lst)
lst.clear()

get_small_factors(48841, lst)  # 13^2 * 17^2
print(lst)
lst.clear()

get_small_factors(371293, lst)  # 13^5
print(lst)
lst.clear()


# This is just a test for find_order. We want to solve the factoring problem like in the next cell
r = find_order(
    7, 8
)  # reminder a must be 2, 7, 8, 11, 13. In this case, only 7, 11, 13 are interesting
print(f"find_order -> {r}")

factors = shor_only_integer_factorization(8)
print(factors)

# factors = shor_only_integer_factorization(16)
# print(factors)

factors = integer_factorization(15)
print(factors)

factors = integer_factorization(2048)
print(factors)

factors = quantum_shor_only_integer_factorization(14)
print(factors)

# factors = integer_factorization(31)
# print(factors)

# factors = integer_factorization(31**3)
# print(factors)

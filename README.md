# Quantum Boltzmann Machines

This repository contains source code for two variants of Bound-Based Quantum Restricted Boltzmann Machines (QRBMs), as proposed in [Amin et al. (2016)](https://arxiv.org/abs/1601.02036).

## Variants Included

- **General QRBM**: Learns a distribution over the data but does **not** support inference.
- **Discriminative QRBM**: A practical implementation suitable for **classification tasks**.

## Background

This project was developed in the summer of 2024 during an internship at [Infleqtion](https://infleqtion.com/). It was used to classify **pairs of cancer types** using binary DNA features representing various genetic mutations.

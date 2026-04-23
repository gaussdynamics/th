# Unit System

All quantities in the rail physics package use the following SI-derived units. Parameters and I/O must be in these units.

| Quantity   | Symbol | Unit   | Notes                          |
|-----------|--------|--------|--------------------------------|
| distance  | x      | m      | along-track position           |
| speed     | v      | m/s    | longitudinal velocity          |
| time      | t      | s      | seconds                        |
| mass      | m      | kg     | vehicle mass                   |
| force     | F      | N      | Newton                         |
| power     | P      | W      | Watt                           |
| grade     | —      | decimal| slope (1% = 0.01)              |
| curvature | κ      | 1/m    | inverse radius                 |

- **Grade**: decimal slope; positive = uphill. `F_grade = m * g * grade(x)` with g in m/s².
- **Curvature**: 1/m; `F_curve` uses a dimensionless coefficient times `m * g * |curvature(x)|`.
- **Resistance**: Davis coefficients A (N), B (N/(m/s)), C (N/(m/s)²).

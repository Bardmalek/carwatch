# Behaviour classifier — synthetic evaluation

Trained and tested on randomised SYNTHETIC trajectories only (3 s windows). This shows the learned model matches the rules' intent; it is not a real-world accuracy figure. Real numbers need labelled real footage.

| Test set | Flag | ML precision | ML recall | Rules precision | Rules recall |
|---|---|---|---|---|---|
| held-out (same distribution) | WEAVING | 1.000 | 0.998 | 0.866 | 0.993 |
| held-out (same distribution) | HARSH ACCEL/BRAKE | 0.936 | 0.934 | 0.548 | 0.934 |
| shifted (2x noise, bigger weaves) | WEAVING | 0.893 | 1.000 | 0.811 | 0.996 |
| shifted (2x noise, bigger weaves) | HARSH ACCEL/BRAKE | 0.877 | 0.873 | 0.415 | 0.956 |

# Real-track evaluation — leave-one-sequence-out on VisDrone ground truth

Windows are 3 s of real moving vehicle tracks (stabilized). Clean windows = normal traffic; weaving / hard braking is INJECTED with known labels. Recall is on injected events; false-alarm rate is on clean real windows (some of which may contain real events, so it is an upper bound on the true false-alarm rate). Each fold trains on 5 sequences and tests on the 6th.

Clean windows: 1009; injected weave: 1009; injected hard brake/accel: 534.

| Method | Weave recall | Weave false alarms (clean) | Harsh recall | Harsh false alarms (clean) |
|---|---|---|---|---|
| Rules | 0.980 | 0.114 | 0.978 | 0.542 |
| Synthetic-only ML | 0.996 | 0.102 | 0.723 | 0.351 |
| Real-aug ML | 0.975 | 0.044 | 0.848 | 0.154 |
| Real+synthetic ML | 0.978 | 0.047 | 0.826 | 0.116 |

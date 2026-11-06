# Stage3 CPU continuation integration — partial, 2026-11-06

The staged integration now exercises the actual ConditionalWan wrapper around a tiny, randomly initialized30-block native Wan. Twenty-nine CPU checks pass, including three-step AdamW parameters/moments: rho0 flow/C1/A2 are bitwise equal to the core Base path; gamma0 with corruption is equal to C1 and differs from clean flow-only. Equal EMA/student features are compared at the **same layer9**, not between student9 and teacher21.

A2 uses the same noisedHR/native timestep/sigma as its teacher. The teacher alone has clean LR and runs before the student graph. Temporary capture hooks are removed before checkpoint replay, while the student LR graph remains alive through backward. Teacher features are detached owned copies. The student projector is outside the EMA. The strict normalized sigma>.3 window uses tensor precision at the boundary.

Source, shadow, loaded-state and gradient finite checks aggregate device-local verdicts before one normal-path host read per device; failure then locates the offending parameter. All exact names/shapes/dtypes/devices/frozen checks and validate-before-mutation rules remain. Late-parameter corruption and128-tensor single-verdict fixtures pass. This avoids thousands of prospective per-step GPU synchronizations without claiming a measured speedup.

All future comparison arms explicitly use fresh AdamW with foreach=False. CPU trajectory equality does not prove CUDA equality to historical default foreach selection. A future loader requires the matched new Base8192 checkpoint/hash/seed and explicit allowed root, restores task parameters only, and resets optimizer/scheduler/RNG state.

No official pretrained model/GPU training occurred in these CPU fixtures. The tiny synthetic optimizer updates are not formal RE. There is no full continuation queue, checkpoint writer or feature-collapse logger yet. **G3 remains unpassed:** real GPU equivalence, actual EMA/A2 memory/step cost and the<=1.5x limit still require a prospective benchmark after the current Base queue releases the GPU. This source is published from staging only; actual Base runtime41407b0 remains untouched.

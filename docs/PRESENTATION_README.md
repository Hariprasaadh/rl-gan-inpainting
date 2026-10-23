# Presentation Guide: Slide-by-Slide Explanation

This guide explains every slide of `RL_GAN_Inpainting_Presentation.pptx` in plain language. It assumes you know only the basics of reinforcement learning (RL). Read Part 1 first (the vocabulary), then use Part 2 to understand and rehearse each slide. Each slide section has four parts:

* **On the slide**: what the audience sees.
* **Explained simply**: what it really means.
* **What to say**: a short script you can speak.
* **If they ask**: likely questions with answers.

The same talking points are also stored as speaker notes inside the PowerPoint file.

---

## Part 1. Vocabulary you need

### 1.1 The image problem

| Term | Meaning |
|---|---|
| **Inpainting** | Filling in a missing or damaged part of an image so that it looks natural. |
| **Mask** | A black-and-white picture marking the damaged area. In this project **1 = hole**, 0 = known pixel. |
| **GAN** | Generative Adversarial Network: a generator makes images and a discriminator judges whether they look real. DeepFill-v2 is a GAN-trained inpainting network. |
| **Backbone** | The big pretrained network that does the heavy lifting. Here it is **DeepFill-v2, pretrained on Places2** (millions of scene photos). |
| **Frozen** | Its weights are never changed during our training. We reuse its knowledge as it is. |
| **Adapter** | A *small* extra network placed after the backbone. It makes a small correction to the backbone's output. We have four, each with its own speciality. |
| **Coarse pass / refine pass** | DeepFill-v2 works in two stages. Stage 1 (coarse) makes a rough fill. Stage 2 (refine) uses *contextual attention* (copying patches from the known region) to make a sharper fill. |
| **Composite** | The result pasted back: `completed = masked + refined × mask`. Pixels outside the hole stay untouched; only the hole is replaced. |

### 1.2 Quality numbers

| Metric | Plain meaning | Better is |
|---|---|---|
| **PSNR** (dB) | How close the pixels are to the original photo. Typical good values are 30 to 40 dB. | higher |
| **SSIM** (0 to 1) | How similar the structure (edges, contrast) is to the original. | higher |
| **LPIPS** | Distance between deep features. It approximates how different the images *look to a human*. | lower |
| **L1** | Average absolute pixel difference. | lower |

Remember: "+1 dB PSNR" is a small but real improvement; "+5 dB" is a big one.

### 1.3 Reinforcement learning basics

RL is learning by trial and reward. Think of training a dog: the dog tries something, you give a treat (reward) or not, and over time the dog learns which behaviour pays off.

| Term | Meaning | In this project |
|---|---|---|
| **Agent** | The decision maker that learns. | The PPO policy |
| **Environment** | The world the agent acts in. | The inpainting system with images and masks (`InpaintingBanditEnv`) |
| **State (s)** | What the agent observes before deciding. | A 261-number vector describing the damaged image |
| **Action (a)** | What the agent chooses. | One of 4 adapters: 0 Global, 1 Local, 2 Boundary, 3 Texture |
| **Reward (r)** | A number saying how good the action was. | Quality gain of the inpainted result minus a cost |
| **Policy π(a given s)** | The agent's rule: the probability of each action in each state. | A small neural network ending in a softmax over 4 actions |
| **Episode** | One run from start to end. | **One step only**: see one image, pick one adapter, get one reward |
| **Return** | Total (discounted) reward over an episode. | Equals the single reward |
| **Discount factor γ** | How much future rewards count (0 means "only the immediate reward"). | **γ = 0** |
| **MDP** | Markov Decision Process: states, actions, rewards, and transitions across many steps. | Not needed here |
| **Contextual bandit** | A one-step RL problem: the agent sees a *context* (state), picks an action, gets a reward, and that is the end. | **This is our formulation** |
| **Value function V(s)** | The reward the agent *expects* from state s. | The critic network |
| **Advantage A** | "How much better was this action than expected?" `A = r − V(s)` when γ = 0. | Used to update the policy |
| **Exploration vs exploitation** | Trying new actions versus repeating the known best one. | Controlled by the entropy bonus |
| **Entropy** | A measure of how spread out the action probabilities are. High entropy = still exploring. | Bonus coefficient 0.05 |
| **PPO** | Proximal Policy Optimization: a stable, popular policy-gradient algorithm that limits how far each update can move the policy. | Stable-Baselines3 PPO |
| **Actor-critic** | Two networks: the actor chooses actions, the critic estimates how good the situation is. | PPO's `MlpPolicy` |
| **Rollout** | A batch of experience collected before one learning update. | 128 episodes |

---

## Part 2. Slide-by-slide

The deck has 15 slides. At about 1.5 to 2 minutes per slide it fits a 20 to 25 minute talk.

---

### Slide 1. Title: RL-GAN Adaptive Image Inpainting

**On the slide**
Project title, subtitle "A PPO agent that chooses the right repair strategy for every damaged image", course code BCSE432E, the two authors, the faculty name, a damaged photo next to the repaired photo, and a card reading "PPO POLICY DECISION: Global Completion".

**Explained simply**
Normal inpainting uses one fixed network for every kind of damage. Our project adds a *decision maker* trained with RL. It looks at the damaged picture and chooses which of four repair specialists should fix it. The card on the right shows exactly that: for the meadow image the agent chose "Global Completion".

**What to say**
"Our project is called RL-GAN adaptive inpainting. We take a strong pretrained inpainting GAN, add four small specialist repair modules, and train a reinforcement learning agent using PPO to pick the best specialist for each damaged image."

**If they ask**
* *Why "RL-GAN"?* The inpainting model is a GAN (DeepFill-v2), and RL is used to control it.
* *What is the RL part exactly?* The choice among four strategies; everything else is supervised or pretrained.

---

### Slide 2. Motivation: One hole, many repair needs

**On the slide**
Three examples (thin scratches, large missing regions, textured surfaces) and four number cards: **4** strategy adapters, **261-D** state per image, **1** decision per image, **235 ms** per request on an RTX 3050.

**Explained simply**
Damage is not all alike:
* A thin scratch needs a repair that blends *smoothly at the border* (seam continuity).
* A big missing region needs the *overall structure* to be rebuilt.
* A textured area (grass, fabric) needs *fine detail*.

One network with one behaviour is a compromise. If we have several specialists and a way to choose between them, each image gets the repair that suits it. Choosing among options based on what you observe is precisely a decision problem, so RL fits.

The four numbers are the project at a glance: 4 specialists, an agent that sees 261 numbers, makes 1 decision, and the full request takes about 235 milliseconds in the web studio.

**What to say**
"Different damage needs different repair. Instead of one compromise model, we keep four specialists and teach an agent to pick one. The agent sees a 261-number description of the image, makes one decision, and the whole request takes about a quarter of a second."

**If they ask**
* *Why not just train one big model?* Our backbone is already strong and large. Training small adapters plus a controller is cheaper and keeps the pretrained knowledge intact.
* *What is 235 ms?* Measured warm on an RTX 3050 laptop GPU in the web studio. It includes the coarse pass, state, policy, baseline pass, adapter pass and PSNR/SSIM calculation.

---

### Slide 3. Architecture: The inference pipeline

**On the slide**
A flow diagram. Top row: Damaged image → Coarse pass → State builder → PPO policy. Bottom row: Refine pass → four adapter chips → Completed image. Colours: blue = frozen, green = trainable, orange = RL controller. A dashed arrow from PPO "selects 1 of 4".

**Explained simply**
Follow one image through the system:

1. **Input**: the damaged image plus its mask (256×256).
2. **Coarse pass** (frozen DeepFill-v2 stage 1): makes a rough fill. We paste this rough fill into the hole to get the *coarse composite*.
3. **State builder**: turns the coarse pass into 261 numbers (next slide).
4. **PPO policy**: reads the 261 numbers and outputs an action 0, 1, 2 or 3.
5. **Refine pass** (frozen DeepFill-v2 stage 2): produces a sharper fill using contextual attention.
6. **Selected adapter**: only the chosen adapter edits the refined output.
7. **Completed image**: `masked + refined × mask`; only the hole changes.

Key point: the backbone is **frozen**; only the adapters (green) and the policy (orange) are trained. All values are in the range [-1, 1].

**What to say**
"The image first goes through the frozen coarse stage. From that we build a 261-dimensional state and give it to the PPO agent. Meanwhile the refine stage produces a sharper result, and only the adapter chosen by the agent edits it. The final image keeps all known pixels and replaces only the hole."

**If they ask**
* *Why run a coarse pass before deciding?* The rough fill already tells us a lot about the hole (what the network expects there, how strong the edges are). That information makes a better state than the raw damaged image.
* *Does the agent see the final image?* No. It decides *before* the adapter runs; that is why the reward is computed afterwards.

---

### Slide 4. State: A 261-D view of the damage

**On the slide**
A coloured bar split into 256 | 4 | 1 and three cards: Latent (256), Mask stats (4), Boundary value (1).

**Explained simply**
In RL the **state** is everything the agent is allowed to know. Ours has three parts:

* **Latent, 256 numbers.** Features from the coarse stage's bottleneck layer (`conv10_atrous`, 96 channels). We average each channel over the image (*global average pooling*) and pass it through a small 2-layer network to get 256 numbers. Think of it as a compact "summary of what the network sees".
* **Mask statistics, 4 numbers.**
  1. *missing ratio*: what fraction of the image is missing
  2. *bounding-box ratio*: how much area the hole's bounding box covers
  3. *number of regions ÷ 10*: how many separate pieces the hole has
  4. *centroid distance*: how far the hole's centre is from the image centre
* **Boundary value, 1 number.** The average edge strength (Sobel gradient) on a 1-pixel ring just *outside* the hole in the coarse composite. It says how much structure crosses the border.

256 + 4 + 1 = 261.

**What to say**
"The agent never sees raw pixels. It sees 261 numbers: 256 learned features from the coarse network, four simple statistics about the hole's size and shape, and one number telling how strong the edges are at the hole's border."

**If they ask**
* *Why these features?* The latent captures content, the mask stats capture geometry, and the boundary value captures how hard the seam will be. Together they cover what the four adapters specialise in.
* *Is the 256-d head trained?* It sits in the encoder and is trained together with the adapters in pretraining.

---

### Slide 5. Strategies: Four adapters, four specialities

**On the slide**
Four coloured cards (Global, Local, Boundary, Texture) each with Mechanism, Training loss and cost, and a bar: `out = tanh( x + 0.5·tanh(Δ) )` FiLM-conditioned by a 64-D strategy embedding.

**Explained simply**
Each adapter is a small network that takes the backbone's refined RGB result `x` and adds a bounded correction `Δ`. The formula `tanh(x + 0.5·tanh(Δ))` means: "start from x, add at most a small correction, keep values in range". The adapter learns *which* correction. **FiLM** is a trick that lets a 64-number "strategy embedding" scale and shift the adapter's internal features, so the same architecture can behave differently per strategy.

| Adapter | What it does | Trained with | Cost |
|---|---|---|---|
| **Global** (0) | Plain pass; balanced reconstruction | L1 on hole + L1 on valid area + VGG perceptual loss | 0.00 |
| **Local** (1) | Scales the input by `1 + 0.5·mask` to focus on the hole | 6×L1 on hole + SSIM-style term | 0.10 |
| **Boundary** (2) | Blends along a ring (dilate minus erode) around the hole edge | Sobel gradient L1, weighted 2.0 on the ring | 0.15 |
| **Texture** (3) | Adds a Laplacian high-pass term inside the hole plus a small high-frequency branch | FFT magnitude loss + VGG perceptual loss | 0.20 |

**Cost** is a small penalty used in the reward. Cheap actions are slightly favoured; more specialised actions must earn their cost by improving quality.

The different **loss functions** are what make the adapters different: each is pulled toward its own goal (hole fidelity, seam continuity, fine detail, balanced result).

**What to say**
"Each adapter is a tiny network with its own training loss. Global is balanced, Local concentrates on the hole, Boundary on the seam, Texture on fine detail. They share one architecture, conditioned by a strategy embedding, and each carries a small cost that the reward accounts for."

**If they ask**
* *What is VGG perceptual loss?* A loss comparing deep VGG-16 features instead of pixels, so results look right to humans.
* *What does "residual" mean?* The adapter outputs a correction added to the input instead of generating the image from scratch.
* *Why a cost?* It makes the agent weigh benefit against extra processing, a standard RL design choice.

---

### Slide 6. RL Formulation: A one-step contextual bandit

**On the slide**
Agent ↔ Environment loop (action a goes right, state s and reward r come back), chips "γ = 0", "1 step", "Discrete(4)", and three cards: State, Action, Reward.

**Explained simply**
This slide is the RL problem definition:

* **State**: the 261-D vector (slide 4).
* **Action**: `Discrete(4)`, meaning exactly one of four adapters.
* **Reward**: how much the chosen adapter improved the result over the coarse composite (slide 8).
* **Episode**: **one step**. The environment (a Gymnasium environment, `InpaintingBanditEnv`) does this:
  * `reset()`: pick an image and random stroke mask, run the coarse pass, build the state.
  * `step(action)`: apply the chosen adapter, measure PSNR, SSIM, LPIPS and the discriminator score, compute the reward, and move on to the next sample.
* **γ = 0**: only the immediate reward matters.

**What to say**
"We model the problem as a contextual bandit. The environment shows an image, the agent picks one of four adapters, the environment returns a reward, and the episode ends. Because there is no future, the discount factor is zero."

**If they ask**
* *What is the "context"?* The 261-D state.
* *What library?* Gymnasium for the environment and Stable-Baselines3 for PPO.

---

### Slide 7. Why a bandit, not a full MDP

**On the slide**
Left: a multi-step MDP drawn as `s0 → s1 → s2` with a STOP box, labelled γ > 0, delayed reward, credit assignment. Right (highlighted): a bandit drawn as `s → a → r` with "one step, then the episode ends".

**Explained simply**
A *full MDP* would let the agent apply several edits one after another and press STOP when satisfied (the repo contains this as `env_multistep.py`). That is harder: rewards arrive late and the agent must work out which earlier step deserves the credit (the *credit assignment problem*).

In our case the right adapter depends on the damage itself, and we can measure the result immediately after one refine pass. So there is nothing to plan over time. A bandit:
* needs just one informed decision per image,
* gets its reward instantly,
* has γ = 0, so no credit assignment,
* trains with low variance, so PPO learns quickly.

**What to say**
"We could have used a multi-step formulation with a stop action, but the best repair is determined by the damage itself and we can measure the outcome right away. That makes a one-step bandit the simplest and most stable formulation."

**If they ask**
* *Is a bandit "real" RL?* Yes. It is the one-step special case of RL, with exploration and policy optimisation, minus long-term planning.
* *Did you try multi-step?* The repository includes a multi-step environment from an earlier experiment line, with action 4 as STOP.

---

### Slide 8. Reward design: What the agent is paid for

**On the slide**
The formula `R = α·ΔPSNR + β·ΔSSIM − γ·ΔLPIPS + δ·ΔD − λ·cost(a)` and five cards giving each term's weight and normalisation.

**Explained simply**
The reward tells the agent what "good" means. It has five parts, all measured **relative to the coarse composite** (the rough fill before any adapter):

| Term | Weight | What it rewards | Normalisation |
|---|---|---|---|
| ΔPSNR | α = 1.0 | pixel accuracy improved | divide by 10 dB, clip to ±2 |
| ΔSSIM | β = 1.0 | structure improved | divide by 0.1, clip to ±2 |
| ΔLPIPS | γ = 0.5 | looks closer to the original (**subtracted**, since lower is better) | divide by 0.1, clip to ±2 |
| ΔD | δ = 0.3 | the discriminator finds it more realistic | change in mean patch confidence |
| cost | λ = 0.1 | **penalty** for the chosen action | 0 / 0.10 / 0.15 / 0.20 |

The total reward is clipped to ±10. Dividing each delta by a typical size puts all terms on a similar scale, so no single metric dominates. The weights come from `configs/rl_agent_bandit.yaml`: `[1.0, 1.0, 0.5, 0.3, 0.1]`.

Why a mix of metrics? PSNR is pixel-wise, SSIM is structural, LPIPS is perceptual, and the discriminator measures realism. Combining them rewards results that are accurate *and* natural.

**What to say**
"The reward is the measured improvement over the coarse fill, using PSNR, SSIM, LPIPS and a discriminator realism score, minus a small cost for the chosen action. Each term is normalised and clipped so that no metric overwhelms the others."

**If they ask**
* *Why clip?* To stop rare extreme values from destabilising training.
* *Is the reward computed with the ground truth?* Yes, during training the environment has the original image to measure against. At inference the agent does not need it; it only reads the state.

---

### Slide 9. PPO: Actor-critic with a clipped update

**On the slide**
A diagram: State (261-D) feeds an Actor (MLP 64×64) producing π(a given s), a softmax over 4 adapters (bars are a schematic), and a Critic (MLP 64×64) producing V(s). Three formula cards: Advantage, Clipped objective, Total loss.

**Explained simply**
PPO is the learning algorithm. It uses **two small neural networks** on the same 261-D state:

* **Actor**: outputs the probability of choosing each of the four adapters.
* **Critic**: outputs **V(s)**, the reward it *expects* for this state.

**1. Advantage: `A = r − V(s)`**
If the actual reward was higher than the critic expected, A is positive, so make that action *more likely*. If lower, A is negative, so make it *less likely*. Because γ = 0, there is no future term; the advantage is just reward minus expectation.

**2. Clipped objective: `min( ρ·A, clip(ρ, 1−ε, 1+ε)·A )`**
ρ = `π_new / π_old` is how much the policy changed for that action. PPO caps this ratio inside `[1−ε, 1+ε]` with **ε = 0.2**. That means "improve the policy, but never move too far in one update", which is the reason PPO is stable.

**3. Total loss: `−L_clip + 0.5·L_V − 0.05·H(π)`**
* `L_clip`: the clipped objective above (maximised, hence the minus).
* `L_V`: the critic's prediction error (weight 0.5, the `vf_coef`).
* `H(π)`: **entropy** of the policy. The bonus (weight 0.05, the `ent_coef`) rewards keeping probabilities spread out, so the agent keeps **exploring** all four adapters instead of locking onto one too early.

Settings not in our config (network sizes 64×64, clip range 0.2) are the Stable-Baselines3 defaults.

**What to say**
"PPO has an actor that outputs the probabilities of the four adapters and a critic that predicts the expected reward. We update the actor using the advantage, which is reward minus the critic's expectation, with a clipping rule that prevents oversized updates. An entropy bonus keeps the agent exploring."

**If they ask**
* *Why PPO and not DQN or REINFORCE?* PPO is stable, simple to tune and supported in Stable-Baselines3; the clipped update guards against collapse with small data.
* *What does the entropy bonus do?* It prevents the policy from becoming deterministic too early; a higher value means more exploration. We raised it from 0.01 to 0.05 to encourage trying all adapters.
* *Why do you still need a critic if γ = 0?* It provides a baseline V(s) that reduces the variance of the advantage.

---

### Slide 10. Training: Two stages, backbone never updated

**On the slide**
Stage 1 card "Adapter pretraining", a "verify gate" box, Stage 2 card "PPO bandit training", and below them a 4-box loop "ONE PPO ITERATION": Rollout, Advantage, Update, Log.

**Explained simply**
Training happens in **two separate stages**, and the backbone never changes.

**Stage 1: Adapter pretraining** (`pretrain-adapters`)
* Data: COCO val2017 images at 256×256 with random stroke masks created on the fly.
* Each adapter gets the *same* batch but is trained with *its own loss* (slide 5).
* Optimiser: Adam, learning rate 1e-4, batch size 4, 5 epochs.
* Result: `checkpoints/adapters/adapters_final.pt`.

**Verify gate** (`verify-strategies`)
A sanity check that the adapters really behave differently: no single strategy may win more than 80% of the test images. If one adapter dominated, the RL agent would have nothing to choose.

**Stage 2: PPO bandit training** (`train-rl`)
* Backbone and adapters are **frozen**; only the policy learns.
* Stable-Baselines3 PPO, γ = 0, learning rate 3e-4, entropy coefficient 0.05, seed 42.
* Result: `checkpoints/rl_agent_bandit/ppo_bandit_final.zip`.

**One PPO iteration**
1. **Rollout**: run 128 one-step episodes with the current policy (image → action → reward).
2. **Advantage**: compute `A = r − V(s)` for each.
3. **Update**: 4 passes (epochs) over the data in mini-batches of 32 to improve the policy.
4. **Log**: a callback writes the action distribution and every reward component to TensorBoard.

Repeat until the timestep budget (set in the config, overridable with `--timesteps`) is used.

**What to say**
"We train in two stages. First the adapters learn their specialities on COCO images with stroke masks. We check they behave differently. Then we freeze everything and train the PPO agent: it collects 128 one-step episodes, computes advantages, and updates its policy in minibatches."

**If they ask**
* *Why two stages instead of joint training?* It keeps the reward stationary (the adapters do not change while the agent learns) and makes each stage simple and reproducible.
* *Is it reproducible?* Yes, all scripts seed Python, NumPy and PyTorch with 42.

---

### Slide 11. Results: Adapters lift PSNR by up to 5.9 dB

**On the slide**
A grouped bar chart of PSNR for stages A, B, C, D across three scenes (Alpine lake, Lush meadow, Sunset peaks), with callouts **+5.9 dB**, **+0.8 dB**, **0.9946**. A caption says A to C use one feed-forward pass and D adds 400 AdamW steps on the test image.

**Explained simply**
This is an **ablation study**: we add one component at a time to see what each contributes.

| Stage | What is used | Uses ground truth at test time? |
|---|---|---|
| **A** | Backbone only, no adapters, no RL | No |
| **B** | Fixed Global adapter always | No |
| **C** | PPO picks the adapter, one forward pass | No |
| **D** | C plus 400 AdamW optimisation steps on that test image | **Yes** |

Numbers (PSNR in dB):

| Scene | PPO chose | A | B | C | D |
|---|---|---|---|---|---|
| Alpine lake | Boundary | 34.45 | 35.21 | 32.25 | 37.23 |
| Lush meadow | Global | 33.72 | 34.54 | 34.54 | 39.65 |
| Sunset peaks | Boundary | 40.58 | 40.00 | 36.80 | 43.19 |

How to read it:
* **A to B**: the Global adapter adds about **+0.8 dB** on the lake and the meadow.
* **Meadow, stage C**: the agent chose Global, so C equals B (**+0.82 dB** over the backbone) with no ground truth.
* **Stage D** reaches 37 to 43 dB: up to **+5.9 dB** (meadow) over the backbone, with SSIM up to **0.9946** (sunset). Stage D optimises directly against the target image, so it shows how closely the adapter structure can represent a solution; it is the *capacity* of the adapters.
* All three scenes use hand-designed masks (`build_controlled_mask`) on sample images 1, 2 and 5.

**What to say**
"We evaluate with a four-stage ablation. Adding the Global adapter lifts PSNR by about 0.8 dB. On the meadow the agent selected Global on its own and gained 0.82 dB without any ground truth. When we let the adapters fit a specific image in stage D, PSNR reaches 37 to 43 dB, up to 5.9 dB above the backbone, which shows how much representational headroom the adapter design has."

**If they ask**
* *Why does D use ground truth?* It is a deliberate upper-reach experiment (test-time optimisation); it is reported separately from A to C, which are pure feed-forward.
* *Where are LPIPS and SSIM?* They are in `results/multi_image_evaluation_report.md` (SSIM e.g. 0.9703 to 0.9946); LPIPS is part of the reward and the evaluators.
* *Is there a severity analysis?* `severity_eval.py` groups results by missing area: 10 to 20%, 20 to 40%, 40 to 60%, 60%+; the fixed evaluation set has 100 masks, 25 per bucket.

---

### Slide 12. Why adaptive: Different images favour different adapters

**On the slide**
A grouped bar chart of PSNR for the four adapters on five verification images, a card "Best per image: Global 2 images, Local 3 images", and a dark card "The right strategy depends on the image, so a learned selector has real work to do."

**Explained simply**
This is the evidence that *choosing* is worthwhile. We ran all four adapters on the same five images (`verify-strategies`):

| Image | Global | Local | Boundary | Texture | Best |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 30.28 | 30.18 | 28.97 | 29.82 | Global |
| 1 | 26.72 | 27.04 | 26.09 | 25.98 | Local |
| 2 | 25.94 | 26.23 | 24.16 | 24.90 | Local |
| 3 | 23.12 | 22.89 | 21.95 | 22.68 | Global |
| 4 | 30.35 | 30.46 | 28.04 | 28.42 | Local |

The winner changes from image to image (Global twice, Local three times); no strategy wins more than 80% of the images. If one adapter were best everywhere, a controller would be pointless. Because the best choice varies, a state-conditioned policy has something to learn.

**What to say**
"We compared all four adapters on the same images. The best one changes from image to image, so a single fixed strategy cannot be best everywhere. That is the motivation for a learned selector."

**If they ask**
* *Why are Boundary and Texture lower here?* They are specialised and their advantage depends on the hole's edge or texture content; this chart is a small five-image check, and the verification gate only needs the adapters to behave distinctly.
* *How many images?* Five, from `data/raw/sample_images`.

---

### Slide 13. Visual results: From damaged input to completed image

**On the slide**
A 3×5 image grid. Rows: Alpine lake (PPO: Boundary), Lush meadow (PPO: Global), Sunset peaks (PPO: Boundary). Columns: Ground truth, Masked, A Backbone, C PPO-selected, D PPO + refine, with PSNR tags.

**Explained simply**
The same stages A, C and D as slide 11, shown as pictures. Left to right: the original, the damaged input (grey stroke is the hole), then the three outputs. Guide the audience's eye to the stroke area: the backbone and PPO results already fill it plausibly, and stage D visibly matches colour and texture more closely, which matches its higher PSNR.

The panels are cropped from `results/demo_image_*.png`, produced by `scripts/run_multi_image_test.py`.

**What to say**
"Here are the same experiments visually. The grey stroke in the masked column is the hole. Each stage fills it, and the stage D result is nearly indistinguishable from the ground truth, in line with its PSNR."

**If they ask**
* *How were the masks made?* Hand-designed in `build_controlled_mask`, to keep scenes controlled.
* *Which action did PPO pick?* The row labels show it: Boundary, Global, Boundary.

---

### Slide 14. Live demo: RL-GAN Studio

**On the slide**
A screenshot of the web app and four feature cards: Paint the hole, See the decision, Compare instantly, Two modes.

**Explained simply**
`python app.py` serves a browser app at `http://localhost:7860`.
* **Paint the hole**: draw a mask with a brush, press Random Mask, or upload your own photo (resized to 256×256).
* **See the decision**: a card names the strategy PPO chose and shows the PSNR gain over the baseline.
* **Compare instantly**: a before/after split slider and a 5-stage pipeline view (ground truth, masked, coarse, baseline, RL-GAN).
* **Two modes**:
  * *Instant Forward Pass* (about 235 ms warm on an RTX 3050).
  * *Ultra-Precision*: after PPO decides, 60 AdamW steps tune the chosen adapter on the uploaded image, then the original weights are restored.

Behind the scenes, `POST /api/inpaint` receives the image and mask, runs coarse pass, state, PPO, baseline and adapter, and returns the images and metrics as JSON.

**What to say**
"We wrapped the whole pipeline in a web studio. You paint a mask, press run, and you see the repaired image, the strategy the agent chose, and the quality numbers. Instant mode takes about a quarter of a second."

**If they ask**
* *What if there is no trained PPO file?* The server falls back to action 1 (Local).
* *Do the weights change after Ultra-Precision?* No, they are restored after each request.

**Demo tip**
Start the server before the talk (`python app.py`) and run one request so the GPU is warm; the first request includes CUDA start-up time.

---

### Slide 15. Key takeaways

**On the slide**
Title "Learn to choose the repair", four cards: Strong prior kept, Four specialists, Cost-aware PPO, Interactive studio, and "Thank you · Questions?".

**Explained simply**
Four sentences summarising the project:
1. **Strong prior kept**: the Places2 DeepFill-v2 backbone stays frozen.
2. **Four specialists**: small residual adapters, each with its own loss.
3. **Cost-aware PPO**: a one-step bandit over a 261-D state with a composite reward including an action cost.
4. **Interactive studio**: mask, decide and compare in about 235 ms.

**What to say**
"To summarise: we keep a strong pretrained inpainting network, add four specialised adapters, and use PPO as a one-step contextual bandit to choose between them using a quality-based, cost-aware reward. We packaged it as an interactive web studio. Thank you, we are happy to take questions."

---

## Part 3. The whole story in 60 seconds

> Inpainting fills missing image regions. A single network treats every hole the same, but different damage needs different repair. We took a pretrained DeepFill-v2 GAN and froze it, added four small adapters that specialise in global structure, local fidelity, boundary continuity and texture, and trained a PPO agent to choose one adapter per image. The agent sees a 261-number state built from the coarse fill and the mask, and is rewarded by the measured gain in PSNR, SSIM, LPIPS and discriminator score minus a small cost. Because the quality is known immediately, we use a one-step contextual bandit with γ = 0. In the ablation, adapters add about 0.8 dB over the backbone, and up to 5.9 dB when fitted to an image. A web studio lets you try it live.

## Part 4. Extra questions you might get

| Question | Short answer |
|---|---|
| What is the difference between supervised learning and what you did in Stage 2? | Stage 1 is supervised (loss against the real image). Stage 2 is RL: the agent is never told the correct action; it only receives a reward after acting. |
| Why is it called a "bandit"? | From slot machines ("one-armed bandits"): pick an arm (action), get a payout (reward), no long-term state change. "Contextual" means the choice depends on an observation. |
| What does the discount factor γ = 0 mean? | Only the immediate reward matters, which is all that exists in a one-step episode. |
| What is exploration here? | Sometimes trying an adapter that currently looks worse to check whether it is actually better for some images. The entropy bonus encourages it. |
| What is the policy's output at inference? | `PPO.predict(state, deterministic=True)` returns the single most likely action. |
| Where is the code? | Model: `src/models/rl_inpainting_model.py`; state: `src/rl/state.py`; reward: `src/rl/reward.py`; environment: `src/rl/env_bandit.py`; PPO training: `src/training/train_rl_bandit.py`; web app: `app.py`. |
| How do I reproduce results? | `python main.py train-rl --config configs/rl_agent_bandit.yaml`, then `python scripts/run_multi_image_test.py` (details in `docs/GETTING_STARTED.md`). |

## Part 5. Related documents

* `docs/ARCHITECTURE.md`: modules and tensor shapes.
* `docs/TRAINING_AND_RL.md`: the training procedure and PPO settings.
* `docs/EVALUATION.md`: metrics, ablation and severity buckets.
* `docs/WEB_STUDIO.md`: the web app and its API.
* `docs/RL_GAN_Inpainting_Report.pdf`: the full project report.

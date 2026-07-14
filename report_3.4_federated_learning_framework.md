# 3.4 Federated Learning Framework

## 3.4.1 Overview and Motivation

Traditionally, training a machine-learning model requires collecting all of the
training data into one place — a single server or data centre. For network
intrusion detection this is a serious problem. The raw material we learn from is
**NetFlow data**: a record of who talked to whom on the network, when, for how
long, and how many bytes and packets were exchanged. This data is extremely
sensitive. It reveals the internal structure of an organisation's network and the
behaviour of its users, and in many cases it cannot legally or contractually be
shared outside the organisation that collected it. As a result, an organisation
that owns only its own traffic can only ever train a model on the kinds of attacks
*it* has already seen.

**Federated Learning (FL)** is a technique that solves this. Instead of moving the
data to the model, FL moves the *model* to the data. Each participant trains a copy
of the model locally, on its own private data, and only ever shares the *learned
parameters* (the numbers inside the model), never the data itself. A central
coordinator combines these parameters into a single, stronger shared model. In this
way, several organisations can collaboratively build one detector that has, in
effect, "seen" all of their attacks — without any of them ever exposing a single
raw network flow.

In this project we treat each of our four NetFlow datasets — **NF-BoT-IoT**,
**NF-UNSW-NB15**, **NF-ToN-IoT**, and **NF-CICIDS2018** — as if it belonged to a
different organisation. Each dataset becomes one **client**. This is a realistic and
deliberately difficult setup, because the four datasets were captured on different
networks with different equipment and different attack types, so their data
distributions genuinely differ. This lets us test whether federated learning can
build a single anomaly detector that works well across all four environments while
keeping each environment's data private.

---

## 3.4.2 The Flower Framework

We implement the federated system using **Flower** (short for *Federated Learning
Framework*, `flwr`), a mature open-source framework designed specifically for FL.
Flower was chosen because it handles the difficult "plumbing" of federated
learning — coordinating clients, passing model parameters back and forth, retrying
on failures, and combining results — so that we can focus on the model itself.

A key practical advantage is that Flower ships as a **ready-made application
template**. A federated project in Flower is just a small Python package with two
required components:

- a **ServerApp** — the coordinator's logic (how to combine updates, how many rounds
  to run), and
- a **ClientApp** — the logic each participant runs (load my data, train the model,
  report the result).

The behaviour of both is controlled by a single plain-text configuration file
(`pyproject.toml`), so the *entire* experiment — which aggregation method, how many
rounds, whether privacy is switched on — is described in one place and launched with
one command (`flwr run`). This makes the experiments reproducible and easy to vary,
which matters because a large part of this project is comparing many federated
configurations against each other.

In this project the ServerApp and ClientApp live in `federated/fedgnn/`
(`server_app.py`, `client_app.py`), and all of the model, data-loading and
parameter-handling logic sits in a shared helper module (`task.py`).

---

## 3.4.3 Client–Server Architecture

Flower uses a **client–server** (also called *star*) architecture. There is one
central **server** and several **clients**, arranged like the hub and spokes of a
wheel.

- **The server (the hub)** owns the single, authoritative *global model*. It never
  sees any data. Its only job is to coordinate: send the current global model out,
  wait for the clients' trained updates to come back, and combine them into an
  improved global model.
- **The clients (the spokes)** each own one private dataset. A client receives the
  global model, trains it on its own local data for a short time, and sends back
  *only the updated parameters*. Clients never talk to each other, and they never
  send data anywhere.

In our setup there is **one server and four clients**, one client per NetFlow
dataset. Because the server only ever handles model parameters and never raw flows,
each dataset's privacy is preserved by construction.

---

## 3.4.4 What Happens in Each Round

Federated training proceeds in repeated **communication rounds**. A round is one
complete loop in which the shared model is sent out, improved locally by every
client, protected, and combined back into a single, better shared model. The
accompanying diagram (Figure 3.X) traces a full round from top to bottom as **six
numbered steps**. The steps fall into three natural groups: the model is
**distributed** (Step 1), it is **improved locally** (Step 2), and then the clients'
work is **protected and combined** on the way back to the server (Steps 3–6). We walk
through each step below; the mechanisms introduced in Steps 3–6 are the subject of
the later sub-sections, so here we only describe *where in the round each one acts*.

**Step 1 — The server hands out the model *(Broadcast, server → clients).*** The round
begins at the central server, which holds the single, authoritative shared model. It
sends a copy of the current model's parameters to every client. Each client is one
organisation with its own private data; in our study the four clients are the four
NetFlow datasets — **NF-BoT-IoT**, **NF-UNSW-NB15**, **NF-ToN-IoT** and
**NF-CICIDS2018**.

**Step 2 — Each client trains the model on its own private data *(Local training).***
Every client loads the received model and trains it on *its own* data for a small
number of **local epochs** (in our default setup, one pass over the client's benign
training traffic). The four clients do this **in parallel**, and each improves the
model in the direction that best fits its own network — so the four resulting models
differ, because the underlying attacks and traffic differ. Critically, **the raw data
never leaves the client's own computer**; only the improved model parameters will be
sent onward.

**Step 3 — Each client keeps its own scaling private *(Personalisation — "Federated
Batch Normalisation", FedBN).*** Before reporting back, each client *holds back* the
small part of the model that is tuned to the scale and statistics of its own data,
and shares only the rest. This is what lets one shared model still cope with four
very different networks. This private-versus-shared split — and the alternatives to
it — is detailed in **Section 3.4.7**. (The diagram illustrates the FedBN option; NA
and FedRep are variations on *what* is held back.)

**Step 4 — Each client blurs its update *(Differential Privacy, DP).*** Next, each
client optionally protects its update against being reverse-engineered: it trims the
update down to a maximum size and adds a small amount of random noise, so that no
single record in its data could later be traced back from the shared model. The
mechanism, and the privacy-versus-accuracy trade-off it entails, is covered in
**Section 3.4.9**.

**Step 5 — The server learns the total, but not any single update *(Secure
Aggregation, SecAgg+).*** Before sending, each client adds a **secret random number**
(a *mask*) to its update. These masks are arranged in pairs across the clients so that
they **cancel out when all the updates are added together**. The server therefore
receives the correct combined total, yet any one client's update, seen on its own,
looks like meaningless random values — so the server can never read an individual
client's contribution. This protocol is explained in **Section 3.4.8**.

**Step 6 — The server combines everyone's work *(Aggregation — "Federated Averaging",
FedAvg).*** Finally the server adds up the four clients' (masked) updates and
**averages** them into one new, improved shared model — so the model learns from
everyone's data without ever seeing it. Thanks to Step 5, the secret masks cancel out
exactly here, so the server only ever obtains the combined total, never a single
client's update. How the averaging is done — and the FedProx alternative — is covered
in **Section 3.4.6**.

**Then repeat.** The new shared model becomes the starting point for the next round,
and Steps 1–6 run again. Training always runs the **full fixed budget of rounds**
(`num-server-rounds`); there is no early stopping. After the final round, the shared
model has been shaped by all four datasets at once, yet no dataset ever left its
owner's machine.

> **A note on model selection.** Unlike our centralised training, the federated
> pipeline uses **no early stopping and no "best round" selection** — it simply runs
> the full round budget and saves the final model. This is a deliberate
> simplification: any form of "best round" selection would require a validation
> signal that peeks across clients, which complicates the privacy story. The trained
> global model is then evaluated separately, offline, on each dataset's held-out test
> data (`evaluate_federated.py`).
>
> **Note on the diagram.** Figure 3.X shows the *full* pipeline with every protection
> switched on (FedBN + DP + SecAgg+ + FedAvg). Steps 3, 4 and 5 are **optional
> layers**: each can be turned on or off independently, which is exactly what the
> experiments in this study vary. A plain FedAvg run, for instance, would skip
> Steps 3–5 and go straight from local training (Step 2) to averaging (Step 6).

---

## 3.4.5 Simulating Multiple Clients

In a real deployment the four clients would be four separate machines in four
different organisations, communicating over a network. For research we do not need
four physical machines — we need to *reproduce the same computation* cheaply and
repeatably. Flower provides a **Simulation Engine** for exactly this.

The simulation engine creates the four clients as independent, isolated processes on
a single machine (using **Ray**, a library for running many Python workers in
parallel). Each simulated client is given only its own dataset, its own copy of the
model, and its own optimiser, so it behaves precisely as a real remote client
would — it simply happens to run on the same computer. The server and the message
passing between them work identically whether the clients are simulated or real;
only the transport is local.

Concretely, the simulation is configured with **four "supernodes"** (Flower's term
for the four client slots). On the GPU machine, the four clients **share a single
GPU**, each being allocated one quarter of it, so all four can train at the same time
without needing four separate GPUs. This makes it feasible to run the large number
of federated configurations this study compares on modest hardware, while the code
remains identical to what a real multi-organisation deployment would run.

---

## 3.4.6 Aggregation — Part A: How the Shared Weights Are Learned

Aggregation is the heart of federated learning: it is how the four clients' separate
local updates are fused into one shared model. We study two aggregation strategies.
Both produce the *shared* part of the model in the same basic way — by averaging —
but they differ in how each client trains locally.

### FedAvg (Federated Averaging)

**FedAvg** is the foundational FL algorithm and our baseline. The idea is simple:
after every client has trained locally, the server computes a **weighted average** of
their parameters, value by value, to form the new global model.

The natural weighting in classic FedAvg is *by amount of data* — a client that
trained on more examples has a bigger say. In our setting that would be a problem:
our four datasets differ enormously in size, so the largest dataset would dominate the
shared model and the smaller networks would be under-served. We therefore use **equal
weighting** (`client-weight = "equal"`): each of the four datasets has the *same
pull* on the shared model, regardless of its size. This ensures the global detector
is balanced across all four network environments rather than tuned to the biggest one.

### FedProx (Federated Proximal)

A well-known weakness of FedAvg appears when clients' data are very different from one
another — exactly our situation. Because each client trains toward its own data, the
four local models can drift far apart within a round, a phenomenon called **client
drift**. Averaging models that have drifted in conflicting directions can slow or
destabilise learning.

**FedProx** addresses this with a small, purely *client-side* change. When a client
trains, it adds a **proximal term** to its loss function:

$$\frac{\mu}{2}\,\lVert w - w_{\text{global}} \rVert^2$$

In plain language, this is a gentle penalty that grows the further a client's
parameters `w` stray from the global parameters `w_global` it started the round with.
It acts like an elastic tether, keeping each client's local training "anchored" near
the shared model so the four updates remain compatible and average together cleanly.
The strength of the tether is set by a single knob, **`proximal-mu`** (μ, default
`0.01`); μ = 0 recovers plain FedAvg.

Importantly, FedProx changes only what happens *inside* a client's local training.
The server-side aggregation is byte-for-byte identical to FedAvg — a weighted
average. This is a useful property, because it means FedProx remains fully compatible
with the secure-aggregation and privacy mechanisms described later (the server does
not need to know or care that clients used a proximal term).

---

## 3.4.7 Aggregation — Part B: What Stays Private Per Client (Personalisation)

Averaging everything into one global model assumes that a single set of parameters
can serve all four networks equally well. But because our datasets are so
heterogeneous, the *best* model for BoT-IoT is not necessarily the best model for
CICIDS2018. **Personalisation** is the idea that each client can keep a small part of
its model *private and local* — never shared, never averaged — so that the shared
model captures what is common across networks while each client's private part adapts
to its own environment.

We compare three options, all implemented cleanly as a **"key filter"**: every client
still exchanges the same list of parameters (which is required for secure
aggregation to work), but certain parameters are simply marked as *private* and
excluded from what is averaged. The private parts are never transmitted; they are
kept on the client between rounds and stored alongside the final model so they can be
paired with the shared model at evaluation time.

### NA — No personalisation

The baseline: **nothing is private**. All parameters are shared and averaged, giving
one fully-global model that every client uses as-is. This is the purest form of
federated learning and the fairest test of whether a single detector can serve all
four networks.

### FedBN — Private Batch-Normalisation layers

Deep-learning models contain **Batch-Normalisation (BN)** layers, whose job is to
rescale the internal signals so that their typical magnitude ("mean and variance") is
well-behaved. The catch is that the *right* rescaling depends on the statistics of the
data — and our four networks have very different traffic statistics.

**FedBN** keeps each client's **BN layers — both their learned parameters and the
running mean/variance statistics they track — private and local**, while sharing and
averaging everything else. Intuitively, the four clients agree on a common
"detector", but each keeps its own private "lens" that adjusts the model to the scale
and distribution of its own traffic. This is a lightweight and very effective way to
cope with heterogeneous data: only a small number of parameters stay local, so the
shared model still benefits from all four datasets. (One technical detail: the integer
counters BN uses internally are never exchanged, because they do not survive the
secure-aggregation encoding.)

### FedRep — Private representation head (shared body, private head)

**FedRep** (*Federated Representation Learning*) splits the model into two conceptual
pieces:

- a **body / encoder** — the shared feature extractor that learns a general-purpose
  representation of network traffic (the *what does normal traffic look like* part),
  and
- a **head** — the part that turns that representation into the final
  reconstruction/anomaly score (in our model, the decoder and the global-edge
  embedding).

Under FedRep, **only the encoder (body) is shared and averaged; the head stays
private on each client**. The reasoning is that a good *representation* of traffic is
general and worth pooling across networks, whereas the final decision layer is best
tailored to each network's specifics.

FedRep also changes *how* a client trains within a round. It **alternates** two
phases: first it trains only the private head while the shared body is frozen (letting
the head adapt to the freshly-received global representation), then it trains only
the body while the head is frozen (improving the shared representation). Each phase
runs for the configured number of local epochs. Only the body's parameters are then
sent back for averaging.

In summary, the three personalisation modes span a spectrum from fully shared (NA),
through sharing everything except normalisation statistics (FedBN), to sharing only
the representation backbone (FedRep). Comparing them tells us how much of the model
*should* be common across networks and how much is better left local.

---

## 3.4.8 Secure Aggregation

Federated learning already keeps raw data on the client. But sharing model
*parameters* is not automatically safe. The updates a client sends can, in principle,
leak information about the data that produced them.

### Threat model

The threat that secure aggregation defends against is an **honest-but-curious
server** (also called *semi-honest*). This is a server that follows the protocol
correctly — it does not sabotage training — but is *curious*: it may inspect every
individual update it receives and try to infer something private about a client's data
from it. Under plain FedAvg the server sees each of the four clients' updates in the
clear, so this is a real exposure.

### How it works

We defend against this using **SecAgg+ (Secure Aggregation)**, a cryptographic
protocol provided by Flower and enabled by a single switch (`secagg = true`). The core
idea is elegant: the server does not actually need to see any *individual* update — it
only needs their *sum* (to compute the average). SecAgg+ arranges for each pair of
clients to agree on a shared random **mask**, added by one and subtracted by the
other. Every client adds masks to its own update before sending it, so each update
looks like meaningless noise on its own. But because the masks are constructed to
**cancel out when all the updates are added together**, the masks vanish in the sum
and the server recovers exactly the correct aggregate — *without ever seeing a single
client's true update.*

To make the scheme robust to clients that go offline mid-round, SecAgg+ uses **Shamir
secret sharing**: each client's secret is split into several **shares** (we use 4
shares) such that any sufficient subset (a **reconstruction threshold** of 3) can
recover it, allowing the surviving clients to cancel a dropped client's mask. Values
are clipped to a fixed range and quantised into integers so the masking arithmetic is
exact.

The result is that the honest-but-curious server sees only the masked *sum* of the
four clients' updates, never any individual one. We verified in this project that
turning SecAgg+ on leaves accuracy essentially unchanged (the small difference is
just quantisation noise), so the privacy protection is effectively free in terms of
model quality.

---

## 3.4.9 Differential Privacy

SecAgg+ hides each *individual* update, but the *aggregate*, and ultimately the
*final released model*, could still reveal something. If an attacker with access to
the finished model can determine whether a particular network flow was in the training
set — a **membership-inference attack** — privacy is still breached. **Differential
Privacy (DP)** is a rigorous, mathematical guarantee against exactly this.

### Threat

DP protects against an adversary who studies the *outputs* of training — the
aggregated updates or the final model — and tries to reverse-engineer facts about
*individual records* in the training data. Whereas secure aggregation trusts the
result but hides the inputs, DP makes the result *itself* provably insensitive to any
single record.

### How it works

DP achieves this by deliberately adding a controlled amount of random **noise**, in
two steps applied to each client's update:

1. **Clipping.** Each client first limits ("clips") the size of its update so that its
   overall magnitude cannot exceed a fixed bound **C** (the *clipping norm*). This caps
   how much any single client — and hence any single record — can influence the model.
   We calibrated **C = 0.03** for our model by measuring the natural size of real
   updates and choosing a bound that constrains them without destroying the signal.

2. **Noise.** Random Gaussian noise is then added. The amount of noise is set by the
   **noise multiplier σ (sigma)**: the standard deviation of the noise is `σ × C`. A
   larger σ means more noise and stronger privacy. σ is the single quantity we vary in
   our privacy-utility study.

We implement DP in **two modes**, which differ in *whom the client has to trust*:

- **Central DP.** Each client clips its own update, and the **server** adds the noise
  to the aggregate. This gives a client-level guarantee but requires trusting the
  server to add the noise honestly. We use the *client-side clipping* variant
  specifically so that it composes cleanly with SecAgg+.
- **Local DP.** Each client clips **and** adds noise to its *own* update before it ever
  leaves the machine. This requires **no trust in the server at all** — the update is
  already private the moment it is sent — but because every client noises independently,
  the total noise is larger and the utility cost is higher.

The strength of the guarantee is summarised by a single number, the **privacy budget
ε (epsilon)**: smaller ε means stronger privacy. Rather than being set directly, ε is
computed *after* training by a **privacy accountant** from the noise level σ, the
failure probability **δ (delta, fixed at 1e-5)**, and the number of rounds. This lets
us report exactly how much privacy each configuration bought.

### The trade-off

Differential privacy is fundamentally a **trade-off between privacy and utility**.
Noise is what provides the privacy guarantee, but the same noise degrades the
model's accuracy. Turning σ up tightens the privacy budget ε but blurs the model and
lowers detection performance; turning σ down sharpens accuracy but weakens the
guarantee. A central experiment in this project is the **privacy–utility curve** — a
sweep over σ that maps out exactly how much detection accuracy we give up for each
level of privacy — so that a practitioner can choose an operating point that meets
their privacy requirements at an acceptable accuracy cost. (We also observed that a
*small* amount of noise can occasionally act as a mild regulariser and even help on
some datasets, but beyond a point the accuracy loss is steep.)

---

## 3.4.10 Advantages

The federated design gives this project several concrete benefits:

- **Data privacy by construction.** Raw NetFlow data — highly sensitive and often
  legally un-shareable — never leaves the organisation that owns it. Only model
  parameters are exchanged.
- **Collaboration without disclosure.** Multiple organisations can jointly build a
  detector that has effectively learned from all of their attacks, without any of them
  revealing their traffic to the others or to a central party.
- **Broader coverage.** Because the shared model is shaped by four different networks
  and attack types at once, it can generalise better than a model trained on any single
  environment in isolation.
- **Layered, tunable privacy.** Secure aggregation and differential privacy can be
  switched on independently and combined, letting us match the privacy protection to
  the threat model — from simply keeping data local, to hiding individual updates from
  the server, to a provable guarantee against inference from the final model.
- **Reproducible experimentation.** The Flower framework lets every configuration be
  described in one file and launched with one command, and its simulation engine
  reproduces a four-organisation deployment on a single machine.

---

## 3.4.11 Challenges

Federated learning also introduces genuine difficulties, several of which are central
to this study:

- **Heterogeneous (non-IID) data.** Our four datasets come from different networks
  with different attacks, so their data distributions differ markedly. This is the
  hardest problem in federated learning: naive averaging can struggle when clients pull
  in conflicting directions. It is precisely why we study FedProx (to reduce client
  drift) and the personalisation methods FedBN and FedRep (to let each client adapt
  locally).
- **The privacy–utility trade-off.** Every layer of privacy has a cost. Secure
  aggregation is nearly free, but differential-privacy noise directly reduces
  detection accuracy, and finding an acceptable operating point requires careful
  calibration and a full privacy–utility sweep.
- **Communication and coordination cost.** Training proceeds over many rounds (150 in
  our runs), each requiring the model to be broadcast and updates collected. In a real
  deployment this network overhead, and tolerance to clients that drop out mid-round,
  become significant engineering concerns (SecAgg+'s secret-sharing scheme is partly a
  response to the latter).
- **No global view for model selection.** Because no party may look across all clients'
  data, conveniences we take for granted in centralised training — early stopping on a
  validation set, picking the best epoch — are awkward under federation. We deliberately
  side-step this by running a fixed round budget and evaluating offline, at some cost in
  tuning flexibility.
- **Systems complexity.** Running four clients that share one GPU, coordinating them
  through a server, and layering cryptographic masking and noise on top all add
  moving parts and failure modes that a single centralised training script does not
  have.

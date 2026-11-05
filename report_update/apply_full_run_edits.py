"""Add the full-budget latent-steering results (Section 5.4) to report.html."""
from pathlib import Path

p = Path(__file__).resolve().parent / "report.html"
s = p.read_text()

M = lambda x: x  # readability marker for inline MathML snippets
BETA3 = "<math><mi>β</mi><mo>=</mo><mn>3</mn></math>"
TEN5 = "<math><msup><mn>10</mn><mn>5</mn></msup></math>"
TEN4 = "<math><msup><mn>10</mn><mn>4</mn></msup></math>"

SEC54 = f"""
<h3>5.4&ensp;Full-budget latent steering on Lift, Can, and Square</h3>
<p>After the pilot, we ran the full latent-steering budget of Section 4 on all three tasks with {BETA3} and one training seed: 100 BC warmup episodes, then {TEN5} online steps at 20 updates per step (2,001,500 updates). The three processes ran concurrently, Lift on one GPU and Can and Square sharing the other, and finished in 13.6, 17.6, and 19.6 hours, with mean update times of 22–28&nbsp;ms. The frozen flow weights were unchanged at the end of every run. Because these flows were retrained on the GPU workstation, BC was re-measured on the same 50 evaluation seeds (40000–40049) used for the steered policy. It succeeds in 23/50 episodes on Lift, 26/50 on Can, and 4/50 on Square (Table 5), within the intervals of Table 3.</p>
<figure>
  <figcaption>Table 5: Full-budget latent steering, one training seed, <math><mi>β</mi><mo>=</mo><mn>3</mn></math>, 50 paired evaluation episodes every {TEN4} online steps. Intervals are 95% Wilson intervals. “Steered only” and “BC only” count episodes at the final evaluation solved by one policy and not the other.</figcaption>
  <table class="tb">
    <thead><tr><th>Task</th><th>Warmup wins</th><th>BC</th><th>Steered, <math><msup><mn>10</mn><mn>4</mn></msup></math> steps</th><th>Steered, final</th><th>Best</th><th>Steered only</th><th>BC only</th></tr></thead>
    <tbody>
      <tr><td>Lift</td><td>51/100</td><td>23/50 [0.33, 0.60]</td><td>43/50</td><td><b>46/50</b> [0.81, 0.97]</td><td>48/50</td><td>23</td><td>0</td></tr>
      <tr><td>Can</td><td>61/100</td><td>26/50 [0.39, 0.65]</td><td>39/50</td><td><b>44/50</b> [0.76, 0.94]</td><td>44/50</td><td>18</td><td>0</td></tr>
      <tr><td>Square</td><td>9/100</td><td>4/50 [0.03, 0.19]</td><td>5/50</td><td>0/50 [0.00, 0.07]</td><td>5/50</td><td>0</td><td>4</td></tr>
    </tbody>
  </table>
</figure>
<p>On Lift and Can, latent steering improves substantially on BC. Steered success exceeds BC at all ten evaluations on both tasks, and the final 95% intervals do not overlap BC’s. Most of the gain appears within the first {TEN4} online steps (86% on Lift, 78% on Can), followed by smaller changes (Figure 2a). At the final evaluation, the steered policy solves 23 Lift and 18 Can episodes that BC fails, and loses none that BC solves. It also finishes faster: successful episodes average 42 instead of 55 steps on Lift and 111 instead of 119 on Can. Online training episodes show the same trend, with 90 of the last 100 successful on Lift and 94 on Can. These are still single-seed results, so variation across training seeds remains unmeasured.</p>
<p>Square fails. With only 9 successful warmup episodes among about 47,000 transitions, the steered policy stays near BC (0–5/50, against BC’s 4/50) for the first 60,000 online steps, then solves no evaluation episode from 70,000 steps onward. The learner logs indicate critic divergence rather than slow learning. Each step earns reward <math><mo>−</mo><mn>1</mn></math>, or 4 on success, which ends the episode, so every true value lies in <math><mo stretchy="false">[</mo><mo>−</mo><mn>100</mn><mo>,</mo><mn>4</mn><mo stretchy="false">]</mo></math>. Square’s mean TD target leaves that range at 1.28 million updates (about 64,000 online steps), falls below <math><mo>−</mo><mn>300</mn></math> by 71,000 steps, and ends near <math><mo>−</mo><msup><mn>10</mn><mn>3</mn></msup></math> (Figure 2b). Its late-training action-critic loss has a median near <math><msup><mn>10</mn><mn>4</mn></msup></math>, compared with 2.7 on Can. The steering policy follows the diverged critic away from the prior. Its median late KL is 2.4, compared with 0.014 on Can and 0.10 on Lift, its latent means reach the <math><mo>±</mo><mn>2</mn></math> bound, and its smallest latent standard deviation falls to 0.007. Lift shows a milder version of the same bias: its mean TD target settles near <math><mo>−</mo><mn>125</mn></math>, below the reachable minimum, without harming success. Sparse successes combined with 20 updates per step appear to make the critic unstable. Bounding TD targets to the reachable range, lowering the update ratio, and adding demonstrations or more warmup episodes to replay are the natural fixes.</p>
<figure class="fig">
  <img src="fig_dsrl_full.svg" alt="Full-budget latent steering: success per task and TD-target magnitude">
  <figcaption>Figure 2: Full-budget latent steering, one training seed. (a) Steered (solid) and frozen BC (dashed) success at each evaluation, against online environment steps after warmup. (b) Magnitude of the mean TD target, logged every 100 updates; the dotted line marks the largest magnitude any true value can reach.</figcaption>
</figure>
"""

edits = [
    # abstract
    ("In a single-seed pilot on Can, latent steering raises success from 60% to 75% after 5,000 online steps.",
     "In single-seed full-budget runs, latent steering raises success from 46% to 92% on Lift and from 52% to 88% on Can, measured against BC on the same 50 evaluation episodes. On Square, where only 9 of 100 warmup episodes succeed, its critic diverges and success falls to zero."),
    # baselines
    ("Its Can pilot is reported in Section 5.3, and a full-budget run on all three tasks is in progress. It will be compared with B and C once their interaction budgets are matched.",
     "Its Can pilot is reported in Section 5.3 and full-budget runs on all three tasks in Section 5.4. It will be compared with B and C once their interaction budgets are matched."),
    # section 5 intro
    ("We report three completed pieces of evidence: closed-loop performance of the frozen Flow BC policies, numerical checks of the flow interface, and a staged pilot of latent steering on Can.",
     "We report four completed pieces of evidence: closed-loop performance of the frozen Flow BC policies, numerical checks of the flow interface, a staged pilot of latent steering on Can, and full-budget latent-steering runs on all three tasks."),
    # 5.3 closing
    ("At that rate, a full-budget run takes an estimated 24–28 hours per task. That run was launched on all three tasks after the pilot, and its results will appear in the final report.</p>",
     "At that rate, we estimated 24–28 hours per full-budget task. The runs in Section 5.4 took 13.6–19.6 hours.</p>"),
    # remaining plan
    ("The full-budget latent-steering run on Lift, Can, and Square is under way. Because its flows were retrained on the GPU workstation, the BC reference for each task will be re-measured on the same evaluation seeds rather than taken from Table 3. Further training seeds will follow for any setting that improves on BC, and the same GPU setup will host the B/C runs.",
     "Full-budget latent steering improved on BC for Lift and Can with one training seed (Section 5.4). We will add at least two further seeds for both tasks. For Square, we will bound TD targets to the reachable return range, lower the update-to-data ratio, and seed replay with demonstrations before rerunning. The same GPU setup will host the B/C runs."),
    ("Full-budget runs, additional seeds, and <math><mi>β</mi></math> sensitivity", "Additional seeds, Square stabilization, and <math><mi>β</mi></math> sensitivity"),
    ("Table 5 lists the remaining work packages", "Table 6 lists the remaining work packages"),
    ("Table 5: Remaining work and responsibility assignments.", "Table 6: Remaining work and responsibility assignments."),
]
for old, new in edits:
    assert s.count(old) == 1, ("not found exactly once", old[:70])
    s = s.replace(old, new)

anchor = "<h2>6&ensp;Remaining plan</h2>"
assert s.count(anchor) == 1
s = s.replace(anchor, SEC54.lstrip("\n") + "\n" + anchor)
p.write_text(s)
print("report.html updated")

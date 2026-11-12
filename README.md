# Lift, Not Luck: companion notebooks

Runnable notebooks for [*Lift, Not Luck: Proving what your product and your agents really changed*](https://books.sugiv.fyi/lift/intro.html),
a hands-on book about experimentation, bandits and causal inference for engineers shipping products and AI agents.
The running example is **Parlo**, a fictional language-learning app, and **Juno**, its fictional AI tutor agent.

Every notebook runs on any laptop, or on Google Colab without a GPU. Nothing is downloaded: every dataset comes from
the seeded simulators in `sim/`, so a run reproduces the chapter's numbers, tables and charts exactly. The notebooks
are saved with their outputs, so you can also read them on GitHub without running anything.

## Running them

```bash
git clone https://github.com/sugix/lift-not-luck && cd lift-not-luck
python -m venv .venv && . .venv/bin/activate     # Python 3.12
pip install -r requirements.txt
jupyter lab chapters/
```

Open a chapter and run it top to bottom. The first cell finds the repo's root so `sim` imports from anywhere.
In Google Colab, open a notebook from this repository and run the first cell: it clones the repo and installs Pingouin.

`bash tools/verify_notebooks.sh` runs every notebook headless in a fresh environment and prints PASS or FAIL for each.

## Chapters

| Notebook | Chapter |
|---|---|
| **Part I: Tests you can trust** | |
| [01](chapters/01-lift-that-wasnt.ipynb) | The Lift That Wasn't: Why We Randomise |
| [02](chapters/02-design-test.ipynb) | Designing the Test: How Big, How Long, What to Count |
| [03](chapters/03-trust-checks.ipynb) | Don't Trust It Yet: Checks Before You Read a Result |
| [04](chapters/04-reading-result.ipynb) | Reading the Result: P-values, Chance to Win and Expected Loss |
| [05](chapters/05-faster-answers.ipynb) | Faster Answers: CUPED and Peeking Safely |
| [06](chapters/06-many-metrics.ipynb) | Many Metrics, Long-Run Truth |
| **Part II: Bandits and self-optimising journeys** | |
| [07](chapters/07-explore-exploit.ipynb) | Explore vs Exploit: Bandits from Scratch |
| [08](chapters/08-bandits-production.ipynb) | Bandits in Production: Delays, Proxies and Guardrails |
| [09](chapters/09-contextual-bandits.ipynb) | Contextual Bandits: The Right Message for Each Learner |
| [10](chapters/10-proving-lift.ipynb) | Proving a Bandit Worked: Holdouts and Honest Numbers |
| **Part III: When you can't randomise** | |
| [11](chapters/11-observational.ipynb) | When You Can't Randomise: Propensity, Weighting and Double ML |
| [12](chapters/12-natural-experiments.ipynb) | Natural Experiments: Before and After, Twins, and the Line |
| [13](chapters/13-interference.ipynb) | When the Coin Isn't Obeyed: Encouragement, Interference and Switchbacks |
| **Part IV: Putting it to work** | |
| [14](chapters/14-who-benefits.ipynb) | Who Benefits? Uplift Models and Targeting Policies |
| [15](chapters/15-lift-engine.ipynb) | Build Your Own Lift Engine |
| [B](chapters/appendix-b-pingouin-cheatsheet.ipynb) | Appendix B: Pingouin and Friends, a Cheat Sheet |

Chapters 11 and 14 fit machine-learning models and Chapter 15 runs a few hundred simulated tests; each takes a minute or two. The rest run in seconds.

## Layout

- `chapters/` one notebook per chapter, with the chapter's text and every code cell in order
- `sim/` the simulators behind every dataset, plus the small readout engine from Chapter 15 (`sim/engine.py`)
- `requirements.txt` the library versions the book was built and verified with
- `tools/verify_notebooks.sh` the headless check

The notebooks are generated from the book's own source, so they always match the published chapters.

Code is MIT-licensed. Parlo, Juno and all their data are fictional.

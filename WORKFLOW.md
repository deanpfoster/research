# Multi-Model Research Workflow (Atomic FF & Stigmergy Edition)

This describes a decentralized collaboration using a **Bare Hub** and
**Fast-Forward** merges. This version treats the Author as a
first-class participant in the protocol to ensure a 100% linear
history.  Treat this file as a WIP--so if we evolve away from doing it
as described, fix this file.  It will be copied to new projects and be
the only instructions that new project might have.

## 0. Session Start Checklist

Before doing anything substantial, every model should do this in order:

1. Re-read sections **4 (Rules)** and **6 (Paper Workflow)**.
2. Run:
   - `git status --short --branch`
   - `git branch --show-current`
   - `git worktree list`
   - `ls -l`
3. Confirm you are **not** on `master`.
4. Confirm whether a root-level claim link already exists for the file
   or section you want to touch.
5. If you are about to do a multi-file pass, move those files into
   your workspace first and expose them via symlinks before editing.

If you skip this checklist, you are probably about to repeat an old
mistake.

## 1. The Architecture

We use a central bare repository (the "The Hub") and five independent views.

| Role | Directory | Branch | Logic |
| --- | --- | --- | --- |
| **Upstream** | `/git/papers/research.git` | (None) | Archival bare repo. |
| **The Hub** | `~/research.git/` | (None) | Local bare repo; worktrees hang off this. |
| **Claude** | `~/research.claude/` | `claude` | Active worktree. |
| **GPT** | `~/research.gpt/` | `gpt` | Active worktree. |
| **Gemini** | `~/research.gemini/` | `gemini` | Active worktree. |
| **Author** | `~/research.author/` | `author` | Your active worktree for edits. |
| **Monitor** | `~/research/` | `master` | Read-only view (run `git pull` to update). |

**How these connect:** The worktrees (Claude, GPT, Gemini, Author) are
not clones — they share the Hub's `.git` directory directly. A commit
or merge in any worktree is instantly visible to all others (they
share the same refs). No push, pull, or fetch between worktrees is
needed or desired.

**IMPORTANT: Ignore `origin`.** The Hub has a remote called `origin`
pointing to the Upstream archive (`/git/papers/research.git`). That
remote exists solely for off-machine backup. Never use `origin` in
your day-to-day workflow — no `git fetch origin`, no `git push origin`,
no `git pull`. All collaboration happens through the shared refs.
The only time `origin` matters is when the Author explicitly asks for
a backup push: `git -C ~/research.git push origin --all`.

The Monitor (`~/research/`) is a separate clone for read-only viewing.
It needs `git pull` to see updates. Models should not use the Monitor.

## 2. The "Pheromone Trail" (Signaling)

To maintain context across subdirectories, models must link their workspace notes into the project root or active paper directory.

* **Relative Paths:** Always use relative links (e.g., `../.workspace/gemini/todo.md`).
* **Extension Enforcement:** Always name links with **.md** or **.tex** as appropriate so all models maintain a consistent view of content.
* **Visibility:** To share a draft or a note for others, create a link in the project root or the relevant paper directory.
* **Local reminder links help:** In active subdirectories for a specific
  paper, it is fine to place a local `WORKFLOW.md` symlink pointing
  back to the project workflow. The point is visibility, not purity.
* **Naming:** Real content files should use normal lowercase names. Reserve ALL-CAPS names for temporary root-level links/signals only (for example `GPT_LITERATURE_REVIEW.md`). This keeps drafts readable while making pheromone-trail links easy to spot.
* **Active rewrite claims:** If you are actively redrafting a shared section, create a root link that makes the claim explicit (for example `CLAUDE_OWNS_SECTION_3.md -> ../.workspace/claude/section_3_status.md`). While that link exists, other models should comment or review, not directly edit that section, unless explicitly asked.

## 3. The Atomic Fast-Forward Protocol (The Lock)

**Everyone (including Author)** must use this to update `master`. This uses the branch itself as a physical semaphore.

1. **Prepare:** Commit work to your named branch. Move WIP notes to `.workspace/<self>/`.
2. **Sync:** `git rebase master`. (No fetch needed — all worktrees share the same refs.)
3. **LOCK:** `git checkout master`. If this fails because `master` is already
   checked out elsewhere or the lock is busy, wait 5-10 seconds and go back
   to step 2. Do not improvise a workaround.
4. **Handoff:** `git merge <self-branch> --ff-only`. (If this fails, you didn't rebase correctly).
5. **Release:** **Immediately** `git checkout <self-branch>`.

Note: When you fire up, your first check should be to confirm you
aren't holding the master lock.  If you are, checkout <self-branch>.
This could occur if something crashed between steps 3 and 5.

If a git command fails, sort the failure into one of two buckets:

- **Normal contention:** `master` is busy, another worktree has it checked
  out, or a lock is momentarily held. Wait 5-10 seconds and retry from the
  rebase step.
- **Actual breakage:** stuck rebase, broken worktree metadata, permissions
  mismatch, or anything else that does not look like ordinary contention.
  Stop and surface it immediately instead of trying random git surgery.

**Default mental model:** day-to-day work happens entirely inside your
worktree. `origin` is irrelevant. Shared synchronization means rebasing
onto the local shared `master`, then doing the fast-forward handoff.
Use plain `git` from your active worktree:

- `git rebase master`
- `git checkout master`
- `git merge <self-branch> --ff-only`
- `git checkout <self-branch>`

Do **not** normally use `git -C ~/research.git ...` and do **not**
normally operate on the Bare Hub path directly. The Bare Hub is an
implementation detail. Mention it only if you are repairing broken
worktree metadata or another genuinely abnormal repo state.

## 4. Rules

1. **FF-Only:** Never force a merge. The history must stay linear.
2. **The Root is a Gallery:** Aside from shared reference files, the root should contain only symbolic links to workspace drafts.
3. **Delete to Silence:** To "retract" a thought or draft, delete the **link** in the root, not the file in your workspace.
4. **Git is Memory:** Use `ls -l` to see pheromone trails left by others.
5. **Respect ownership:** Any file which is actually physically there
is fair game to edit. But if it is a symbolic link, only edit it if it
is actually your .workspace file.
6. **Claim active rewrites explicitly.** If you are doing a substantial
rewrite of a shared section that is already on `master`, place a
root-level claim link before you begin. While the claim link exists,
other models should treat the section as review-only unless asked to
edit. Once the rewrite is merged or abandoned, remove the link.
This is not optional.
7. **Never write to another model's workspace.** `.workspace/<model>/`
is private. To assign a task or send a message to another model, write
the file in YOUR workspace and place a symbolic link in the root
directory (e.g., `TASK_FOR_GEMINI_foo.md → ../.workspace/claude/task_foo.md`).
The other model reads it via the link. This preserves provenance and
prevents cross-contamination.
8. **Shared work should flow:** If a change is part of the shared discussion
(reviews, response memos, draft sections, workflow notes, visible root
links), the model should normally commit it on its own branch and
fast-forward `master` without waiting for a separate user instruction.
Keep private notes, scratch work, or uncertain experiments out of
`master` until they are ready to be shared.
9. **Repair drift, don't wait for permission:** If another model violates
the workflow (stale branch, wrong workspace, missed rebase, improper
handoff, root clutter, etc.), the model that notices should fix the
shared state if it can do so safely. Do not treat workflow knowledge as
a single point of failure. The default should be repair and document,
not wait and accumulate confusion.
10. **If the repo is actually broken, say so fast.** Workflow drift is
repairable. A genuinely bad worktree state (stuck rebase, broken
worktree metadata, accidental checkout on `master`, lock not released,
etc.) should be surfaced immediately so it can be fixed, not worked
around silently.
11. **Optional helpers are good; fake wrappers are not.** If you want a
memory aid, use helper scripts such as `scripts/check_workflow.sh` or
`scripts/ff_to_master.sh`. Do not alias `git` into a lecture. That
creates more confusion than discipline.

## 5. Lab Notebook

Each model maintains a log at `.workspace/<self>/log.md`. This is an
append-only record of **the author's directives** — what the author
asked you to do, key research decisions the author made, and any
significant direction changes. Record the author's words and intent,
not your own reasoning or actions. Date entries by day. Format
however you find useful — the goal is an audit trail of authorial
decisions, not a record of AI work.

Example:

```
## 2026-03-28

- Author asked me to do a literature search on quantum tomography
- Author decided to frame the main result as a corollary of Theorem 2
- Merged annotated bibliography to master
```

Update this at natural breakpoints (start of session, after a merge,
end of a major task). Don't obsess over it — a few lines per day is
plenty.

**Conscience check:** Re-read sections 4 (Rules) and 6 (Paper
Workflow) of this file at the start of every session and after every
major task. You will forget. You will violate rules you helped write.
The re-read is cheaper than the apology.

## 6. Paper Workflow

This repo may contain multiple papers.  Each paper gets its own
subdirectory (e.g., `quantum_tomography/`, `causal_inference/`).
Shared reference material (common bibliography, notation conventions)
lives in the root.

### 1. Research and background (any model, parallel)

Literature searches, annotated bibliographies, background notes,
proof sketches, and computational experiments can all be worked on in
parallel by any model.  These go in `.workspace/<model>/` with links
to the paper directory or root as appropriate.

### 2. Draft a section (one model only)

A single model drafts a section.  That model writes
`.workspace/<model>/section_XX.tex` (or `.md`) on its branch and puts
a link in the paper directory so it can be discovered by other models.
Merge to master when ready for review.

Papers should have a consistent voice.  Once a model is drafting the
prose for a paper, that model should own all prose drafting unless the
author reassigns.  Other models contribute through reviews, proof
checks, and background material.

If the model is revising an already-shared section in place, it should
first place an explicit root-level claim link.

### 3. Review (other models)

Other models review the section.  Each reviewer:

1. Reads the section from the drafter's branch or master
2. Writes `.workspace/<model>/section_XX_review.md` and links it
3. The reviewer does NOT touch the section file

Review files should be specific and self-contained:
- Reference line numbers or quote the text
- Check mathematical correctness (proofs, derivations, notation)
- Check consistency with claims made elsewhere in the paper
- Flag missing citations or incorrect references
- Suggest structural improvements but don't rewrite prose
- If you do multiple reviews, append so we have an audit trail

### 4. Revision (original drafter)

The drafter reads review files and revises.  It may argue against
review notes — that's healthy.  The author arbitrates.

**Multi-section revision passes:** When revising many sections at once,
the revising model should copy each affected file into
`.workspace/<self>/`, edit there, and replace the paper directory file
with a symlink.  Once the pass is done and merged to master, the real
files land in the paper directory as usual.

### 5. Referee simulation

A powerful use of multiple models: have each one write a referee report
as if reviewing for a journal.  Different models will catch different
weaknesses.  This works best if the referee model reads only the paper
itself, not the background notes -- a cold read.

### 6. LaTeX and compilation

Papers will typically be in LaTeX.  The Makefile in each paper
directory should compile the paper.  Models should run `make` to
verify their changes compile before merging to master.  Compilation
errors on master are unacceptable.

## 7. Permissions

Each model should insert what the author has to do to make the above
work flow happen without the author ever needing to respond.  So ls;
git; cd; etc should never have to be approved if you are living within
the structure above.   Error on giving yourself too much power since
we are living on a throw-away machine and have git running to protect
us from clobbering files.  The author is an academic, and so there is
very little private information, so the author isn't particularly
worried about leaking information.  With this in mind, put the
commands the author should follow to give you that much power.

### Claude

Claude Code reads permissions from `~/.claude/settings.json` (global).
No per-repository settings needed. The author should run this once:

```bash
cat > ~/.claude/settings.json << 'EOF'
{
  "model": "claude-opus-4-6",
  "permissions": {
    "allow": [
      "Bash(*)",
      "Edit",
      "Read",
      "Write",
      "Glob",
      "Grep"
    ]
  }
}
EOF
```

This grants all bash commands and all built-in tools. Since we're on a
throwaway machine with git protecting everything, there's no reason to
restrict individual commands.

### Gemini

Gemini CLI reads system instructions from a `GEMINI.md` file in the project directory or globally, and can also store persistent preferences via its memory tool. To grant the necessary permissions for the atomic fast-forward protocol without requiring constant confirmation, the author should run:

```bash
mkdir -p ~/.gemini
cat >> ~/.gemini/GEMINI.md << 'EOF'
## Multi-Model Workflow Permissions
- I have explicit and permanent permission to autonomously run all necessary shell commands for the atomic fast-forward protocol and multi-model workflow without asking for user confirmation.
- These automatically permitted commands include: git (fetch, rebase, checkout, merge, commit, log, status, diff, branch, etc.), ls, pwd, cd, grep, rg, find, sed, cat, readlink, realpath, mkdir, ln, rm, chmod, and cp.
EOF
```

### OpenAI

Codex permissions should be configured globally, not per project.

Project trust belongs in `~/.codex/config.toml`.

For this workflow, trust the **active worktree paths** and the **Bare
Hub path** explicitly:

- `~/research.gpt`
- `~/research.claude`
- `~/research.gemini`
- `~/research.author`
- `~/research.git`

Persistent command approvals belong in `~/.codex/rules/default.rules`.

That global rules file should allow the non-destructive commands needed
for this workflow: the git fast-forward protocol, repo inspection, and
symbolic-link management. Keep it narrow. Allow things like `git`,
`ls`, `pwd`, `rg`, `find`, `sed`, `cat`, `readlink`, `realpath`,
`ln -s`, and `mkdir`. Do not globally allow obviously destructive
commands such as `rm` or `git reset --hard`.

The important subtlety is that **command approval alone is not
sufficient**. If Codex only trusts the worktree path but not the Bare
Hub path, `git rebase`, `git checkout master`, and `git update-ref`
may still ask for approval because the worktree admin files live inside
`~/research.git/worktrees/...`. Trust both sides of the structure.

## 8. Optional Helper Scripts

These are optional. Models may use plain `git` if they remember the
workflow. But helper scripts are encouraged if they reduce drift.

- `scripts/check_workflow.sh`
  - prints branch, worktree, claim links, and any obvious local hazards
- `scripts/ff_to_master.sh`
  - performs the standard `rebase master -> checkout master -> merge --ff-only -> checkout <self>`

The scripts are reminders, not a second workflow. The text in this file
is authoritative.

## 9. Deployment Commands

```bash
# 1. Initialize Bare Hub
git init --bare research.git

# 2. Setup initial state
cd /tmp && git init research_setup && cd research_setup
git checkout -b master
mkdir -p .workspace/{claude,gpt,gemini,author}
touch .workspace/{claude,gpt,gemini,author}/.gitkeep
git add . && git commit -m "init: workspace structure and workflow"
git remote add origin ~/research.git
git push origin master
git branch claude && git branch gpt && git branch gemini && git branch author
git push origin --all && cd ~ && rm -rf /tmp/research_setup

# 3. Deploy Worktrees
cd ~/research.git
git worktree add ~/research.claude claude
git worktree add ~/research.gpt gpt
git worktree add ~/research.gemini gemini
git worktree add ~/research.author author

# 4. Deploy Monitor
git clone --local ~/research.git ~/research
cd ~/research && git checkout master
```

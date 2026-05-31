# Branch Protection Setup

After pushing this repo, configure branch protection in GitHub:

**Settings -> Branches -> Add branch protection rule**

For `main`:

- [x] Require a pull request before merging
  - [x] Require approvals: 1
  - [x] Dismiss stale pull request approvals when new commits are pushed
  - [x] Require review from Code Owners
- [x] Require status checks to pass before merging
  - [x] Require branches to be up to date before merging
  - Status checks: `CI Success` (the single summary job from ci.yml)
- [x] Require conversation resolution before merging
- [x] Require signed commits (optional but recommended)
- [x] Do not allow bypassing the above settings
- [x] Restrict who can push to matching branches (admins only)

## Why this is FAANG-style

1. **No direct push to main** — every change goes through PR review
2. **Required status checks** — CI must pass before merge
3. **CODEOWNERS review** — sensitive paths require owner approval
4. **Up-to-date branches** — prevents merging stale code with broken interactions
5. **Single CI Success check** — one required check instead of N — easier to manage as workflow evolves

---
name: job-runner
description: Prepare and manage jobs on rented machines with the job-runner CLI. Use when writing a job-runner spec.json, checking prerequisites or offers, or starting, monitoring, resuming, or stopping a run.
---

# job-runner

Use `job-runner --help` for setup and options. To use the CLI directly from a checkout, run `./bin/job-runner`.

## Prepare a run

1. Give each new run its own directory, `name` (the machine label), and `store.prefix`. Put `spec.json` and the code archive identified by `code.path` in that directory. For the schema, see `job_runner/spec.py` in the job-runner checkout.
2. Define the machine requirements under `provider` and the commands in `steps`, in execution order. Declare each step's inputs and outputs. Inputs must come from `files` already in the store or from earlier steps. Review `policy.budget`, the spending threshold in USD for starting shutdown, and `policy.wait`, how many seconds a run without a machine waits for a matching offer.
3. Run `job-runner check DIRECTORY` to check the archive, input files, storage credential lifetime, and provider access without renting or writing data. Resolve any reported problems. Where the provider supports it, `job-runner offers DIRECTORY` shows matching offers before renting.

## Run and monitor

Start the controller with `job-runner run DIRECTORY`. It can rent and renew billable machines. From another shell, use `job-runner status DIRECTORY` to see completed steps, machines still in use or being returned, spending, and recent events.

If `run` exits with code `75`, renew the storage credential and run the command again with the same directory to resume.

## Stop and confirm return

Use `job-runner stop DIRECTORY` to request a stop. Keep the controller running so it can handle the request and return the machines. Use `job-runner status DIRECTORY` to confirm `over` is `true` and read the `outcome`. For scripts, `job-runner wait DIRECTORY` waits for `over` and exits successfully when the outcome is `complete`.

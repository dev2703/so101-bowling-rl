@AGENTS.md

## Claude-Specific Rules
- Always use the `Explore` subagent for broad codebase searches, directory listings, and log inspection instead of reading dozens of files directly in the main context.
- If you are running as a delegated worker thread, execute only the scoped task brief, run the verification commands, report exact modified files and test results, and stop. Do not spawn nested sub-delegations unless explicitly asked.
*Render Unto Caesar (RUC)* plugs into your AI agent and gives it the ability to write and run snippets of code on an as-needed basis &mdash; snippets of code that call right back to the AI agent during their execution. It effectively melds LLMs with traditional software, allowing each part of a task to be handled by the architecture that suits it best. The result is inference, judgment, and creativity that nonetheless executes methodically and reliably across long operations, large datasets, and complex processes.

![screenshot](./assets/RenderUntoCaesar-perdiem.gif)

## Deterministic execution + LLM judgment in a single integrated workflow

> **If you've built a machine whose whole purpose is to be less machine-like, but you still ask it to do machine-like things, you're going to have a bad time.**
>
> Foundation model developers keep training LLMs to follow user instructions more faithfully. They expend untold gigajoules of energy and eons of CPU/GPU time trying to get neural networks &mdash; fundamentally fickle probabilistic beasts by their very nature &mdash; to execute steps in the order they're given, and to refrain from getting distracted, prematurely declaring completion, or outright inventing completely new un-asked-for operations.
>
> This is a silly thing to optimize for.
>
> **We already have machines that follow instructions exactly, repeatably, and tirelessly. They're called *computers*.**
>
> Neural networks make bad computers. Yes, neural networks *run on* computers, but that fact alone doesn't magically grant them the ability to work *as* computers &mdash; any more so than the fact that you're made of cells automatically makes you a microbiologist.
>
> Neural nets are not bad computers because they need more reinforcement learning. They are bad computers because *they are not computers*. They are pattern-completion engines: powerful, flexible, fuzzy, associative, and astonishingly useful &mdash; but fundamentally ill-suited to long procedural execution.
>
> This is not even unique to *artificial* neural nets. Human brains are bad at it too. That's why we have to use a pencil to perform long division. That is why we use checklists, calendars, recipe cards, laminated flowcharts, and sticky notes. It's called &ldquo;externalized cognition&rdquo;. It's a superpower that's mostly unique to our species: we're smart enough to know what we're not smart about, and to use tools to patch the holes in our smarts. Externalized cognition exists precisely because neural cognition is not reliable procedure.
>
> RUC is built around that distinction: **Let the neural network handle interpretation, judgment, and creativity. Let the computer handle execution.**

LLMs make a great interface for requesting tasks, but a poor engine for executing them. They are good at writing verbiage and assessing messy or ambiguous information, but poor at carrying out long, exact procedures. When an operation requires both the fuzziness of LLMs and the methodical rigor of traditional code, **Render Unto Caesar (RUC)** bridges the gap. RUC separates the work into the parts that need interpretation and the parts that need machinery, and makes them interoperate to give you the best of both worlds.

Modern LLM apps make it natural to ask for complex work in plain English. The problem is that plain-English requests often mix together things LLMs are good at with things they are famously bad at. A user might ask ChatGPT, Claude, or Cursor to review support tickets, impute missing data in a spreadsheet, or brainstorm a series of ads. Those tasks require judgment and creativity, but they also require loops, counts, state, validation, consistency, and auditability &mdash; the stuff traditional code is built for.

Most users, even engineers, don’t naturally separate those two layers when they ask for the task. They describe the outcome they want, not the architecture needed to produce it reliably. RUC does that separation for them. It identifies which parts require deterministic execution — iteration, arithmetic, validation, state, and auditability — and which parts require classification, summarization, ambiguity resolution, or freeform writing. It defines the interfaces between those parts, specifying how code-shaped work should pass data into LLM-shaped work, and how LLM-shaped results should flow back into procedural execution. That lets both kinds of work operate in concert: code provides structure, continuity, and rigor, while the LLM handles nuance where rigid rules would break down.

The result is a system that can do LLM-shaped work with the control, structure, and repeatability of procedural software.

### Core idea

Render Unto Caesar is built around a simple split:

- **Code handles** loops, arithmetic, validation, retries, files, progress tracking, aggregation, and state management.
- **LLMs handle** language comprehension, creative writing, classification, summarization, ambiguity resolution, fuzzy matching, and other semantic decisions.

The important part is not merely that both are available. The important part is that RUC defines the boundary between them, so procedural code and LLM judgment can safely interoperate inside one workflow.

## Who this is for

RUC is built for business and engineering professionals who want to describe sequences of operations they want their computer to perform, and who expect the computer to carry out those operations *as a computer* rather than as some kind of Plinko game.

In practice, that usually means:

- Product leaders and PMs working with large CSV or spreadsheet exports.
- Analysts and operations teams who run repeated cleanup, classification, and normalization tasks.
- Domain owners and data stewards who are accountable for data quality outcomes.
- Engineers and AI power users supporting non-engineering teammates in VS Code.

### Example situations in which RUC proves useful

1. A PM needs to triage 1,700 support feedback tickets before a roadmap review.
2. A financial analyst has the transcripts of hundreds of earnings calls. The analyst needs to populate a spreadsheet detailing which company had the call, what date the call occurred on, who was on it, what key metrics were discussed, and what the values of those metrics were.
3. A regional sales manager needs to sort 7,000 free-text customer survey comments into bins based on what features of the product each customer described most prominently.
4. A national advertising executive wants to create custom copy for an ad to run on a county-by-county basis for all 3,144 counties in the United States.

## Requirements

- VS Code 1.100+
- Docker installed and running
- Access to pull the following Docker image from the GitHub Container Repository (GHCR): `ghcr.io/mightydatainc/ruc-mcp:latest`

## Development Guide

### Commands and Workflows

This project provides the following commands to developers who are working on this extension locally.

- Build the VSIX file locally: `npm run build`
- Install the extension into your VS Code: `npm run install`
  - Open or restart VS Code
  - Look in your list of Extensions
  - "Render Unto Caesar" should be there
- Publish to the VS Code Extension Marketplace: `npm run publish`
  - Need to have publisher token and credentials in place

### Developer

<img src="https://avatars.githubusercontent.com/u/83461102" alt="Mighty Data Inc logo" align="right" width="140" />

*Render Unto Caesar* was developed by Mikhail Voloshin at [Mighty Data, Inc.](https://www.mightydatainc.com/)  Copyright &copy; 2026. Licensed under MIT License.

Project repository on GitHub: https://github.com/mightydatainc/ruc-mcp

Please use the GitHub repository's issue tracker to report any problems or difficulties.

<br clear="right" />


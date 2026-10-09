"""Train-only full-gradient throughput check for the contextual controller."""

import argparse
import copy
import gc
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/l40s-tools-v5.json")
    parser.add_argument("--output", default="results/l40s-tools-v5/benchmark.json")
    sweep = parser.add_mutually_exclusive_group()
    sweep.add_argument("--padding-sweep", action="store_true")
    sweep.add_argument("--context-rerank-compare", action="store_true",
                       help="Compare legacy and contextual reranking from identical parent weights and Train cases")
    sweep.add_argument("--dialogue-state-compare", action="store_true",
                       help="Compare predicted dialogue state off/on using identical parent weights and Train cases")
    sweep.add_argument("--workflow-state-compare", action="store_true",
                       help="Compare existing state routed only to control versus shared with workflow and reranking")
    args = parser.parse_args()
    os.chdir(ROOT)
    import numpy as np
    import torch

    from newsvendor import corpus, structured_data
    from newsvendor.cli import provenance
    from newsvendor.io import digest, read, require, write
    from newsvendor.structured_labels import suite_targets
    from newsvendor.structured_train import (
        Router,
        action_weights,
        balance_cases,
        cases,
        conversation_successors,
        dataset_hashes,
        language_backward,
        language_view,
        load_backbone,
        replay_construction,
        seed,
        variant,
    )

    require(not Path(args.output).exists(), "Preserve prior benchmark measurements before retrying")
    require(torch.cuda.device_count() == 1 and torch.cuda.get_device_name() == "NVIDIA L40S", "Exactly one L40S required")
    torch.set_num_threads(4)
    config = variant(read(args.config), "base")
    require(not args.context_rerank_compare or (config["encoder"].get("structuredTools")
                                               and config["encoder"].get("dialogueController")),
            "Contextual rerank comparison requires structured tools and the controller")
    require(not args.dialogue_state_compare or config["encoder"].get("dialogueState"),
            "Dialogue-state comparison requires a model with the new state module")
    require(not args.workflow_state_compare or (config["encoder"].get("dialogueWorkflow") and config["encoder"].get("dialogueState")),
            "Workflow-state comparison requires a configured shared dialogue state")
    seed(config["seed"])
    rows, labels, collection = structured_data.load(config)
    counts = Counter(labels[r["id"]]["tool"] if labels[r["id"]]["action"] == "call_tool" else "speak"
                     for r in rows if r["split"] == "train" and r["component"] == "abcd")
    if config["encoder"].get("structuredTools"):
        config["training"]["actionWeights"] = action_weights(counts, config["training"].get("preserveCallPrior", False))
    if config["encoder"].get("workflowProgress"):
        from newsvendor.structured_procedures import annotations

        for key, annotation in annotations(rows, config["encoder"]["toolSchema"], True,
                                           config["training"].get("sourceRoleSupervision", False),
                                           config["encoder"].get("dialogueState", False)).items():
            labels[key] = labels[key] | annotation
    episodes = corpus.generate(read(config["researchConfig"]))
    train = balance_cases(cases(rows, episodes, "train") + replay_construction(config["constructionReplay"], episodes), config["training"])
    successors = conversation_successors(train)
    random = np.random.default_rng(config["seed"])
    order = random.permutation(len(train))[:64]
    tokenizer, encoder = load_backbone(config["encoder"])
    model = Router(encoder, config["encoder"]).cuda().train()
    parent = torch.load(config["warmStart"], map_location="cpu", weights_only=True, mmap=True)
    missing = model.load_state_dict(parent["weights"], strict=False)
    require(not missing.unexpected_keys and all(k.startswith("control.") for k in missing.missing_keys), "Unexpected warm-start mismatch")
    def prepare_cases():
        prepared, added_role_fields = [], 0
        for index in order:
            view, target = language_view(train[index], tokenizer, config, labels, collection)
            label = labels.get(train[index][1]["id"], {})
            if config["training"].get("sourceRoleSupervision") and label.get("argumentRoles"):
                plain = suite_targets(view, "abcd", {k: v for k, v in label.items() if k != "argumentRoles"})
                added_role_fields += sum("role" in new and "role" not in old for new, old in zip(target["fields"], plain["fields"], strict=True))
            target.update(caseIndex=int(index), retrievalRound=1)
            if int(index) in successors:
                target["nextIndex"] = successors[int(index)]
            prepared.append((view, target))
        if config["training"].get("sourceRoleSupervision"):
            require(added_role_fields > 0, "No added source-role targets in probe")
        return prepared, added_role_fields

    prepared, added_role_fields = prepare_cases()
    measurements = []
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    variants = [(None, 65000), (1.25, 65000), (1.25, 70000)] if args.padding_sweep else [(config["encoder"].get("paddingRatio"), config["encoder"].get("checkpointTokenLimit"))]
    contexts = (False, True) if args.context_rerank_compare else (config["encoder"].get("contextRerank", False),)
    histories = (False, True) if args.dialogue_state_compare else (config["encoder"].get("dialogueState", False),)
    workflows = (False, True) if args.workflow_state_compare else (config["encoder"].get("dialogueWorkflow", False),)
    variants = [(ratio, limit, contextual, history, workflow) for ratio, limit in variants for contextual in contexts for history in histories for workflow in workflows]
    configured = copy.deepcopy(config)
    report = {"scope": "Train-only numerical and throughput check, not effectiveness", "status": "running",
              "runnerHash": digest(Path(__file__).read_bytes()), "config": configured,
              "trainableParameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "encoderParameters": sum(p.numel() for p in model.encoder.parameters()),
              "warmStartMissingKeys": missing.missing_keys,
              "provenance": provenance(configured), "inputIds": [train[i][1]["id"] for i in order],
              "datasetHashes": dataset_hashes(configured),
              "initialInputHash": digest([view["batch"]["input_ids"].tolist() for view, _ in prepared]),
              "comparison": "Identical parent weights, Train source cases and seed. Self-predicted follow-up states can differ; inspect encodedTokens alongside elapsed time. Repeat zero includes cold compilation.",
              "parentHash": digest(Path(config["warmStart"]).read_bytes()), "measurements": measurements}
    write(args.output, report)
    for ratio, limit, contextual, history, workflow in variants:
        model.load_state_dict(initial)
        model.config.update(paddingRatio=ratio, checkpointTokenLimit=limit, contextRerank=contextual, dialogueState=history, dialogueWorkflow=workflow)
        config["encoder"].update(paddingRatio=ratio, checkpointTokenLimit=limit, contextRerank=contextual, dialogueState=history, dialogueWorkflow=workflow)
        prepared, added_role_fields = prepare_cases()
        model.zero_grad(set_to_none=True)
        seed(config["seed"])
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        optimizer = model.optimizer(config["training"])
        try:
            for repeat in range(2):
                torch.cuda.synchronize()
                start = time.perf_counter()
                costs, losses, rerank_gradients, stage_gradients, role_gradients = [], [], [], [], []
                history_gradients, past_losses = [], []
                for first in range(0, len(prepared), 32):
                    group = [(dict(view, contextRerank=contextual), target)
                             for view, target in prepared[first:first + 32]]
                    optimizer.zero_grad(set_to_none=True)
                    for lo, hi in model.training_batches([v for v, _ in group]):
                        values, head_losses, cost = language_backward(model, group[lo:hi], tokenizer, config, train, labels, collection, scale=1 / len(group))
                        losses.extend(values)
                        costs.append(cost)
                        past_losses.extend(m["pastTools"] for m in head_losses if "pastTools" in m)
                    if config["encoder"].get("rerankSupervision") or contextual:
                        gradient = sum(float(p.grad.detach().abs().sum()) for p in model.tools.rerank.parameters() if p.grad is not None)
                        require(gradient > 0, "Reranker did not receive gradients")
                        rerank_gradients.append(gradient)
                    if config["encoder"].get("workflowProgress"):
                        require(any("workflowNodes" in target for _, target in group), "No workflow supervision in probe group")
                        gradient = sum(float(p.grad.detach().abs().sum()) for p in model.tools.stage.parameters() if p.grad is not None)
                        require(gradient > 0, "Workflow head did not receive gradients")
                        stage_gradients.append(gradient)
                    if config["training"].get("sourceRoleSupervision"):
                        gradient = sum(float(p.grad.detach().abs().sum()) for p in model.tools.role.parameters() if p.grad is not None)
                        require(gradient > 0, "Role head did not receive gradients")
                        role_gradients.append(gradient)
                    if history:
                        history_gradients.append({name: sum(float(p.grad.detach().abs().sum()) for p in module.parameters() if p.grad is not None)
                                                  for name, module in (("tool", model.control.history.tool), ("sequence", model.control.history.sequence), ("project", model.control.history.project))})
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                    optimizer.step()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                measurements.append({"paddingRatio": ratio, "tokenLimit": limit, "contextRerank": contextual, "dialogueState": history, "dialogueWorkflow": workflow, "repeat": repeat, "cases": len(prepared), "seconds": elapsed,
                                     "casesPerSecond": len(prepared) / elapsed, "loss": float(np.mean(losses)),
                                     "encodedTokens": sum(c["tokens"] for c in costs), "peakBytes": torch.cuda.max_memory_allocated()})
                if history:
                    require(past_losses and all(sum(g[name] for g in history_gradients) > 0 for name in ("tool", "sequence", "project")), "Dialogue-state gradients missing")
                    changes = {name: any(not torch.equal(initial[k], v.detach().cpu()) for k, v in model.state_dict().items() if k.startswith("control.history." + name + "."))
                               for name in ("tool", "sequence", "project")}
                    require(all(changes.values()), "Dialogue-state weights did not update")
                    measurements[-1].update(historyGradients=history_gradients, historyUpdated=changes, pastToolLosses=past_losses,
                                            pastToolTargets=sum(len(t.get("pastTools", [])) for _, t in prepared))
                if rerank_gradients:
                    changed = any(not torch.equal(initial[k], v.detach().cpu()) for k, v in model.state_dict().items() if k.startswith("tools.rerank."))
                    require(changed, "Reranker parameters did not update")
                    measurements[-1].update(rerankGradients=rerank_gradients, rerankUpdated=changed)
                if stage_gradients:
                    changed = any(not torch.equal(initial[k], v.detach().cpu()) for k, v in model.state_dict().items() if k.startswith("tools.stage."))
                    require(changed, "Workflow parameters did not update")
                    measurements[-1].update(workflowGradients=stage_gradients, workflowUpdated=changed,
                                            workflowTargets=sum("workflowNodes" in t for _, t in prepared))
                if role_gradients:
                    changed = any(not torch.equal(initial[k], v.detach().cpu()) for k, v in model.state_dict().items() if k.startswith("tools.role."))
                    require(changed, "Role parameters did not update")
                    measurements[-1].update(roleGradients=role_gradients, roleUpdated=changed, addedRoleTargets=added_role_fields)
                print(measurements[-1], flush=True)
                write(args.output, report)
        except torch.cuda.OutOfMemoryError:
            measurements.append({"paddingRatio": ratio, "tokenLimit": limit, "contextRerank": contextual, "dialogueState": history, "dialogueWorkflow": workflow, "error": "out_of_memory"})
            print(measurements[-1], flush=True)
            write(args.output, report)
        except Exception as error:
            report.update(status="failed", error=type(error).__name__ + ": " + str(error))
            write(args.output, report)
            raise
        del optimizer
    report["status"] = "completed_with_oom" if any("error" in row for row in measurements) else "complete"
    write(args.output, report)


if __name__ == "__main__":
    main()

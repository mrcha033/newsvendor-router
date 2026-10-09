"""Trainable token encoder and shared functional heads; no language generation/helper."""

import math

import torch
from torch import nn
from torch.nn import functional as fn
from transformers import AutoModel, AutoTokenizer

from .corpus import KINDS, STATUSES
from .heads import Head
from .io import digest, require
from .sequence import DemandEncoder, features
from .structured_inputs import DECISIONS, MODES, OPS, SCALES, best_span, span_value
from .structured_value import constrain

SCHEMA = "structured-router-v1"


class Router(nn.Module):
    def __init__(self, encoder, config):
        super().__init__()
        self.config = config
        require(not config.get("contextRerank") or (config.get("structuredTools") and config.get("dialogueController")),
                "Contextual reranking requires structured tools and the dialogue controller")
        require(not config.get("dialogueState") or (config.get("structuredTools") and config.get("dialogueController")),
                "Dialogue state requires structured tools and the dialogue controller")
        require(not config.get("dialogueWorkflow") or (config.get("dialogueState") and config.get("workflowProgress")),
                "Dialogue workflow coupling requires dialogue state and workflow progress")
        self.encoder = encoder.requires_grad_(True)
        if config.get("sharedQueries", False):
            require(
                all(getattr(encoder.config, name, 0) == 0 for name in ("embedding_dropout", "mlp_dropout", "attention_dropout")),
                "Shared query encoding requires deterministic dropout-free schema representations",
            )
        if config.get("compileLayers", False):
            torch._dynamo.config.recompile_limit = config.get("compileLimit", 64)
            for layer in self.encoder.layers:
                layer.forward = torch.compile(layer.forward, dynamic=True)
        # Only the projection shape differs across backbones. Its initialization must not
        # shift the RNG stream used by the identical fusion, demand and functional heads.
        with torch.random.fork_rng(devices=[]):
            self.project = nn.Linear(encoder.config.hidden_size, 256)
        layer = nn.TransformerDecoderLayer(
            256, 4, 512, dropout=0.0, batch_first=True, norm_first=True
        )
        self.fusion = nn.TransformerDecoder(layer, 2)
        self.demand = DemandEncoder()
        self.heads = nn.ModuleDict(
            {
                "kind": Head(256, 128, len(KINDS)),
                "state": Head(256, 128, len(STATUSES)),
                "mode": Head(256, 128, len(MODES)),
                "use": Head(256, 128, 2),
                "evidence": Head(256, 128, 2),
                "relation": Head(256, 128, len(OPS)),
                "scale": Head(256, 128, len(SCALES)),
                "decision": Head(256, 128, len(DECISIONS)),
                "question": Head(256, 128, 1),
                "recovery": Head(256, 128, 1),
                "value": Head(256, 128, 2),
            }
        )
        self.pointers = nn.ModuleDict(
            {
                name: nn.Linear(256, 256, bias=False)
                for name in ("start", "end", "operand1", "operand2", "choice")
            }
        )
        if config.get("structuredTools"):
            from .structured_tool_heads import ToolHeads

            self.tools = ToolHeads()
        if config.get("dialogueController"):
            from .structured_control import Controller

            self.control = Controller(config.get("controllerResidual", False), config.get("controllerAuxiliary", False),
                                      config.get("dialogueState", False))

    def forward(self, view):
        multiple = isinstance(view, list)
        views = view if multiple else [view]
        device = next(self.parameters()).device
        with torch.autocast(
            device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and self.config.get("precision") == "bf16",
        ):
            encoded = self.encode_batch(views)
            if len(views) > 1 and self.config.get("batchFusion", False):
                outputs = self.finish_batch(views, encoded)
            else:
                outputs = [self.finish(v, *states) for v, states in zip(views, encoded, strict=True)]
        # Promote loss inputs and inference scores to FP32 after encoder/head computation.
        outputs = [{k: v.float() for k, v in output.items()} for output in outputs]
        return outputs if multiple else outputs[0]

    def encoder_sequences(self, views):
        sequences, owners, shared = [], {}, {}
        for case, view in enumerate(views):
            lengths = view["batch"]["attention_mask"].sum(-1).tolist()
            for seq, length in enumerate(lengths):
                key = None
                if self.config.get("sharedQueries", False) and seq >= view["queryStart"]:
                    key = tuple(view["batch"]["input_ids"][seq, :length].tolist())
                    if key in shared:
                        owners[shared[key]].append((case, seq))
                        continue
                    shared[key] = (case, seq)
                sequences.append((case, seq, length))
                owners[case, seq] = [(case, seq)]
        if self.config.get("bucketLengths", False):
            sequences.sort(key=lambda item: item[2], reverse=True)
        return sequences, owners

    def encoder_batches(self, sequences):
        size, multiple = self.config.get("encoderBatch", 4), self.config.get("padMultiple", 1)
        ratio = self.config.get("paddingRatio")
        require(size > 0 and multiple > 0, "Encoder batch and padding multiple must be positive")
        require(ratio is None or ratio >= 1, "Padding ratio must be at least one")
        batches, current, width, tokens = [], [], 0, 0
        for item in sequences:
            length = math.ceil(item[2] / multiple) * multiple
            proposed = max(width, length)
            if current and (len(current) == size or (
                ratio is not None and (len(current) + 1) * proposed > ratio * (tokens + length)
            )):
                batches.append((current, width))
                current, width, tokens = [], 0, 0
            current.append(item)
            width, tokens = max(width, length), tokens + length
        if current:
            batches.append((current, width))
        return batches

    def encode_batch(self, views):
        device = next(self.parameters()).device
        require(views, "Empty encoder batch")
        sequences, owners = self.encoder_sequences(views)
        token_map, query_map = {}, {}
        for case, view in enumerate(views):
            for i, (seq, pos) in enumerate(view["positions"]):
                token_map.setdefault((case, seq), []).append((i, pos))
            for i, (seq, lo, hi) in enumerate(view["queryPositions"]):
                query_map.setdefault((case, seq), []).append((i, lo, hi))
        groups = [[[], []] for _ in views]
        batches = self.encoder_batches(sequences)
        padded = sum(len(part) * width for part, width in batches)
        limit = self.checkpoint_limit() if self.config.get("checkpointTokenLimit") else None
        if self.training and self.config.get("checkpointTokenLimit"):
            enabled = padded > limit
            if enabled != self.encoder.is_gradient_checkpointing:
                (
                    self.encoder.gradient_checkpointing_enable
                    if enabled
                    else self.encoder.gradient_checkpointing_disable
                )()
        self.last_batch = {
            "cases": len(views),
            "tokens": sum(length for _, _, length in sequences),
            "sequences": len(sequences),
            "sharedSequences": sum(len(v) - 1 for v in owners.values()),
            "checkpointing": self.encoder.is_gradient_checkpointing,
            "encoderCalls": len(batches),
        }
        if self.config.get("checkpointTokenLimit"):
            self.last_batch.update(paddedTokens=padded, tokenLimit=limit)
        for current, width in batches:
            ids = torch.full(
                (len(current), width), self.encoder.config.pad_token_id, dtype=torch.long
            )
            mask = torch.zeros_like(ids)
            for row, (case, seq, length) in enumerate(current):
                ids[row, :length] = views[case]["batch"]["input_ids"][seq, :length]
                mask[row, :length] = 1
            hidden = self.project(
                self.encoder(
                    input_ids=ids.to(device), attention_mask=mask.to(device)
                ).last_hidden_state
            )
            prefix = (
                fn.pad(hidden.float().cumsum(1), (0, 0, 1, 0))
                if any(owner in query_map for case, seq, _ in current for owner in owners[case, seq])
                else None
            )
            for case in sorted({case_id for canonical, seq, _ in current for case_id, _ in owners[canonical, seq]}):
                locations, original, queries, query_order = [], [], [], []
                for row, (owner, seq, _) in enumerate(current):
                    for actual, actual_seq in owners[owner, seq]:
                        if actual != case:
                            continue
                        for index, pos in token_map.get((case, actual_seq), []):
                            locations.append((row, pos))
                            original.append(index)
                        for index, lo, hi in query_map.get((case, actual_seq), []):
                            queries.append((row, lo, hi))
                            query_order.append(index)
                if locations:
                    index = torch.tensor(locations, device=device)
                    groups[case][0].append((original, hidden[index[:, 0], index[:, 1]]))
                if queries:
                    index = torch.tensor(queries, device=device)
                    pooled = (
                        prefix[index[:, 0], index[:, 2]] - prefix[index[:, 0], index[:, 1]]
                    ) / (index[:, 2] - index[:, 1])[:, None]
                    groups[case][1].append((query_order, pooled.to(hidden.dtype)))
        encoded = []
        for tokens, queries in groups:
            ordered = []
            for pieces in (tokens, queries):
                ids = torch.tensor([i for indices, _ in pieces for i in indices], device=device)
                states = torch.cat([states for _, states in pieces])
                ordered.append(states[ids.argsort()])
            encoded.append(ordered)
        return encoded

    def checkpoint_limit(self):
        config = self.encoder.config
        return (
            self.config["checkpointTokenLimit"] * (22 * (768 + 2 * 1152))
            / (config.num_hidden_layers * (config.hidden_size + 2 * config.intermediate_size))
        )

    def training_batches(self, views):
        split = self.config.get("splitLongBatches", False)
        if split and len(views) > 1 and self.config.get("checkpointTokenLimit"):
            sequences, _ = self.encoder_sequences(views)
            padded = sum(len(part) * width for part, width in self.encoder_batches(sequences))
            if padded > self.checkpoint_limit():
                middle = len(views) // 2
                return self.training_batches(views[:middle]) + [
                    (lo + middle, hi + middle) for lo, hi in self.training_batches(views[middle:])
                ]
        return [(0, len(views))]

    def joined_queries(self, view, queries):
        count = len(view["fields"]) + len(view["actions"])
        joined = queries[:count]
        if view["public"]["observations"] and view.get("linked", False):
            seq, _ = features(view["public"]["observations"])
            history = self.demand([seq], [view.get("horizon", 7)])["state"]
            joined = torch.cat([joined, history])
        return joined

    def finish_batch(self, views, encoded):
        joined = [self.joined_queries(v, q) for v, (_, q) in zip(views, encoded, strict=True)]
        memory = [t for t, _ in encoded]
        queries = nn.utils.rnn.pad_sequence(joined, batch_first=True)
        tokens = nn.utils.rnn.pad_sequence(memory, batch_first=True)
        qlengths = torch.tensor([len(q) for q in joined], device=queries.device)
        lengths = torch.tensor([len(t) for t in memory], device=tokens.device)
        query_mask = torch.arange(queries.shape[1], device=queries.device)[None] >= qlengths[:, None]
        token_mask = torch.arange(tokens.shape[1], device=tokens.device)[None] >= lengths[:, None]
        fused = self.fusion(
            queries, tokens, tgt_key_padding_mask=query_mask, memory_key_padding_mask=token_mask
        )
        heads = {name: head(fused) for name, head in self.heads.items()}
        pointers = {name: pointer(fused) for name, pointer in self.pointers.items()}
        return [
            self.finish(
                view, *states,
                fused=(fused[i], {k: v[i] for k, v in heads.items()}, {k: v[i] for k, v in pointers.items()}),
            )
            for i, (view, states) in enumerate(zip(views, encoded, strict=True))
        ]

    def finish(self, view, tokens, queries, fused=None):
        device = tokens.device
        count = len(view["fields"])
        action_count = len(view["actions"])
        joined = (
            fused[0] if fused is not None
            else self.fusion(self.joined_queries(view, queries).unsqueeze(0), tokens.unsqueeze(0))[0]
        )
        fields, actions = joined[:count], joined[count : count + action_count]
        output = {}
        for name, head in self.heads.items():
            on_actions = name in ("value", "recovery")
            output[name] = (
                fused[1][name][count : count + action_count] if on_actions else fused[1][name][:count]
            ) if fused is not None else head(actions if on_actions else fields)
        output["type"] = output.pop("kind")
        output["fieldState"] = fields
        output["actionState"] = actions
        for name in ("start", "end"):
            projected = fused[2][name][:count] if fused is not None else self.pointers[name](fields)
            output[name] = projected @ tokens.T / math.sqrt(256)
        if view["atoms"]:
            prefix = fn.pad(tokens.float().cumsum(0), (0, 0, 1, 0))
            bounds = torch.tensor(
                [(a["tokens"][0], a["tokens"][-1] + 1) for a in view["atoms"]], device=device
            )
            atoms = (
                (prefix[bounds[:, 1]] - prefix[bounds[:, 0]])
                / (bounds[:, 1] - bounds[:, 0])[:, None]
            ).to(tokens.dtype)
            for name in ("operand1", "operand2"):
                projected = fused[2][name][:count] if fused is not None else self.pointers[name](fields)
                output[name] = projected @ atoms.T / math.sqrt(256)
        if view["choices"]:
            choice = queries[count + action_count :][view["choiceQueries"]]
            projected = fused[2]["choice"][:count] if fused is not None else self.pointers["choice"](fields)
            output["choice"] = projected @ choice.T / math.sqrt(256)
            allowed = torch.tensor(view["choiceFields"], device=device)
            output["choice"] = output["choice"].masked_fill(
                torch.arange(count, device=device)[:, None] != allowed[None, :], -1e9
            )
        output["value"] = constrain(fn.softplus(output["value"].float()), view.get("valueCosts"))
        output["recovery"] = output["recovery"].flatten()
        allowed = torch.tensor(view["allowedActions"], device=device)
        output["recovery"] = output["recovery"].masked_fill(~allowed, -1e9)
        allowed_modes = torch.tensor(
            [["modes" not in field or mode in field["modes"] for mode in MODES] for field in view["fields"]],
            device=device,
        )
        output["mode"] = output["mode"].masked_fill(~allowed_modes, -1e9)
        if view.get("dialogueWorkflow") and view.get("controllerTurns"):
            start = view["controllerStart"]
            output["dialogueState"] = self.control.history(view, queries, queries[start:start + action_count], output)
        if hasattr(self, "tools"):
            self.tools(view, fields, actions, tokens, queries, output)
        if hasattr(self, "control") and view.get("controllerStart") is not None:
            self.control(view, queries, output)
            if view.get("contextRerank"):
                self.tools.rerank_context(view, actions, queries, output)
        return output

    def optimizer(self, config):
        backbone = list(self.encoder.parameters())
        other = [p for name, p in self.named_parameters() if not name.startswith("encoder.")]
        return torch.optim.AdamW(
            [
                {"params": backbone, "lr": config["encoderLr"]},
                {"params": other, "lr": config["headLr"]},
            ],
            weight_decay=config.get("weightDecay", 0.01),
            fused=config.get("fusedOptimizer", False),
        )


def load_backbone(config):
    require(
        len(config["revision"]) == 40 and all(c in "0123456789abcdef" for c in config["revision"]),
        "Pin encoder SHA",
    )
    common = {
        "revision": config["revision"],
        "cache_dir": ".cache/torch-models",
        "token": False,
        "trust_remote_code": False,
    }
    tokenizer = AutoTokenizer.from_pretrained(config["model"], **common)
    require(tokenizer.is_fast, "Offsets require a fast tokenizer")
    encoder = AutoModel.from_pretrained(
        config["model"],
        **common,
        use_safetensors=True,
        attn_implementation=config.get("attention", "sdpa"),
    )
    require(
        config["maxLength"] <= encoder.config.max_position_embeddings,
        "Configured sequence length exceeds encoder context",
    )
    # Full source indexing precedes bounded encoder chunks; do not truncate tokenizer offsets.
    tokenizer.model_max_length = 10**9
    if config.get("gradientCheckpointing", True):
        encoder.gradient_checkpointing_enable()
    return tokenizer, encoder


def configure_optimizer(optimizer, config):
    # AdamW's fused kernel expects step counters on the parameter device. Older
    # foreach checkpoints store those counters on CPU; moments and values stay intact.
    fused = config.get("fusedOptimizer", False)
    for group in optimizer.param_groups:
        group["fused"] = fused
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            if "step" in state:
                state["step"] = state["step"].to(parameter.device if fused else "cpu")


def compute(op, values):
    require(values and all(math.isfinite(v) for v in values), "Invalid operands")
    if op == "copy":
        return values[0]
    require(len(values) == 2, "Binary operation requires ordered operands")
    a, b = values
    if op in ("divide", "percent", "change"):
        require(b != 0, "Zero divisor")
    result = {
        "add": lambda: a + b,
        "subtract": lambda: a - b,
        "multiply": lambda: a * b,
        "divide": lambda: a / b,
        "average": lambda: (a + b) / 2,
        "percent": lambda: 100 * a / b,
        "change": lambda: 100 * (a - b) / b,
    }[op]()
    require(math.isfinite(result), "Nonfinite calculation")
    return result


@torch.inference_mode()
def extract(view, output):
    fields = []
    for i, field in enumerate(view["fields"]):
        mode = MODES[int(output["mode"][i].argmax())]
        status = STATUSES[int(output["state"][i].argmax())]
        result = {
            "field": field["id"],
            "name": field["name"],
            "mode": mode,
            "type": KINDS[int(output["type"][i].argmax())],
            "state": status,
            "value": None,
            "evidence": [],
            "scale": SCALES[int(output["scale"][i].argmax())],
            "use": bool(output["use"][i].argmax()) if "position" in field else True,
            "queryOnly": bool(field.get("questionField")),
            "tool": field.get("tool"),
        }
        if field.get("roles") and "role" in output:
            result["role"] = view["roles"][int(output["role"][i].argmax())]
        state_required = field.get("stateRequired", False)
        if mode in ("missing", "conflict") or (state_required and status in ("conflict", "unavailable")):
            result["reason"] = (
                "conflict" if mode == "conflict" or (state_required and status == "conflict") else "missing"
            )
        else:
            start, end = best_span(view, output["start"][i], output["end"][i])
            text, loc = span_value(view, start, end)
            result["evidence"] = [loc] if int(output["evidence"][i].argmax()) else []
            if mode == "choice" and field.get("choices"):
                selected = int(output["choice"][i].argmax())
                result["value"] = view["choiceValues"][selected]
                from .suite import canonical

                result["evidence"] = [loc] if canonical(text) == canonical(str(result["value"])) else []
                result["source"] = "observed" if result["evidence"] else "schema"
            elif mode == "compute" and view["atoms"]:
                op = OPS[int(output["relation"][i].argmax())]
                ids = [int(output["operand1"][i].argmax())]
                if op != "copy":
                    ids.append(int(output["operand2"][i].argmax()))
                atoms = [view["atoms"][j] for j in ids]
                try:
                    result["value"] = compute(op, [a["value"] for a in atoms])
                    result["expression"] = {"op": op, "operands": [a["location"] for a in atoms]}
                    result["evidence"] = [a["location"] for a in atoms]
                except ValueError:
                    result["reason"] = "invalid_calculation"
            elif mode == "entity" and "entity" in output:
                entity = view["entities"][int(output["entity"][i].argmax())]
                from .structured_tools import copy_agrees

                # A historical hypothesis cannot override the current field pointer.
                # Computed entities retain their separately revalidated expression.
                agrees = copy_agrees(entity, text, result.get("role"))
                if view.get("copyAgreement") and not agrees and "expression" not in entity:
                    result.update(value=text.strip(), evidence=[loc], mode="span",
                                  source="observed", copyRejected="current_field_disagrees")
                else:
                    result.update(value=entity["value"], evidence=entity["evidence"], source=entity["origin"])
                    if "expression" in entity:
                        result["expression"] = entity["expression"]
            elif mode == "span" or (mode == "entity" and not view.get("structuredTools")):
                result["value"] = text.strip()
                result["evidence"] = [loc]
            else:
                result["reason"] = "unsupported_value"
        if not state_required:
            result["state"] = (
                "conflict" if mode == "conflict" else "candidate" if result["value"] is not None else "unavailable"
            )
        fields.append(result)
    return fields


@torch.inference_mode()
def assemble(view, output, *, no_value=False, use_value=True):
    fields = extract(view, output)
    valid = []
    for i, action in enumerate(view["actions"]):
        if not view["allowedActions"][i]:
            continue
        if (
            action["id"] == "retrieve"
            and view["selectedChunks"] == view["indexedChunks"]
            and not view.get("allowReread")
        ):
            continue
        valid.append(i)
    require(valid, "No permitted action")
    selected = (
        max(valid, key=lambda i: float(output["recovery"][i]))
        if no_value or not use_value
        else min(valid, key=lambda i: float(output["value"][i].sum()))
    )
    if "callGate" in output and view.get("controllerEnabled", True) and (no_value or not use_value):
        from .structured_control import selection

        selected = selection(output, valid)
    action = view["actions"][selected]["id"]
    result = {
        "action": action,
        "fields": fields,
        "actionValues": [
            {
                "action": view["actions"][i]["id"],
                "residualLoss": float(output["value"][i, 0]),
                "requestCost": float(output["value"][i, 1]),
            }
            for i in valid
        ],
        "retrieved": view["retrieved"],
        "inputHash": view["inputHash"],
    }
    if no_value or not use_value:
        result.pop("actionValues")
    if "callGate" in output:
        result["callProbability"] = float(output["callGate"].softmax(-1)[1])
        result["controllerScores"] = [{"action": a["id"], "score": float(output["controlRecovery"][i])}
                                      for i, a in enumerate(view["actions"]) if view["allowedActions"][i]]
    if "pastTools" in output:
        probabilities = output["pastTools"].softmax(-1)
        result["observedActions"] = [
            {"historyIndex": turn["historyIndex"], "source": {"kind": "history", "id": str(turn["historyIndex"])},
             "tool": view["actions"][view["historyActions"][int(probabilities[i].argmax())]]["id"].removeprefix("call_tool:"),
             "probability": float(probabilities[i].max()), "partial": turn["partial"], "type": "prediction"}
            for i, turn in enumerate(view["controllerTurns"]) if turn["role"] == "tool"
        ]
    result["policyMode"] = "recovery" if no_value or not use_value else "value"
    if "procedure" in output:
        probabilities = output["procedure"].softmax(-1)
        indices = probabilities.topk(min(3, len(probabilities))).indices.tolist()
        result["procedureIds"] = [view["procedures"][i]["id"] for i in indices]
        stages = output["stage"].softmax(-1)
        node = int(stages[indices[0]].argmax()) if view.get("workflowProgress") else int(stages.argmax())
        result["progress"] = {"stage": node,
                              "procedureScores": [{"id": view["procedures"][i]["id"], "score": float(probabilities[i])} for i in indices]}
        if view.get("workflowProgress"):
            result["progress"]["stageMeaning"] = "next_public_tool_node_or_no_further_logged_tool"
            result["progress"]["nextNodes"] = []
            for i in indices:
                procedure = view["procedures"][i]
                node = int(stages[i].argmax())
                result["progress"]["nextNodes"].append({
                    "procedure": procedure["id"], "node": node,
                    "tool": procedure["steps"][node] if node < len(procedure["steps"]) else None,
                    "score": float(stages[i, node]),
                    "source": {"kind": "document", "id": procedure["document"],
                               "start": procedure["start"], "end": procedure["end"]},
                })
        elif view.get("procedureContext"):
            result["progress"]["stageMeaning"] = "observed_tool_count_capped_at_15"
    if "rerankDelta" in output:
        base = output["rerankPrior"] if "rerankPrior" in output else output["baseRecovery"]
        if "rerankPrior" in output:
            result["actionScoreSpace"] = "joint_call_and_conditional_tool"
        result["actionScores"] = [{"action": a["id"], "base": float(base[i]),
                                   "reranked": float(base[i] + output["rerankDelta"][i])}
                                  for i, a in enumerate(view["actions"]) if view["allowedActions"][i]]
    if action in ("ask", "confirm"):
        eligible = [
            f
            for f in fields
            if (f["value"] is not None if action == "confirm" else f["state"] != "verified")
        ]
        if action == "confirm" and not eligible:
            action = result["action"] = "ask"
        eligible = eligible or fields
        field = max(eligible, key=lambda f: float(output["question"][fields.index(f), 0]))
        spec = next(f for f in view["fields"] if f["id"] == field["field"])
        result["question"] = {
            "field": field["field"],
            "reason": field.get("reason", "unconfirmed"),
            "choices": spec.get("choices", []),
            "text": "Please provide or confirm " + spec["description"] + ".",
        }
        if action == "confirm":
            result["question"]["reason"] = "confirmation"
            result["question"]["value"] = field["value"]
    elif action.startswith("call_tool:"):
        tool = action.split(":", 1)[1]
        result["action"], result["tool"] = "call_tool", tool
        positional = [i for i, f in enumerate(view["fields"]) if f.get("tool") == tool and "position" in f]
        used = [i for i in positional if fields[i]["use"]]
        # Never compact a gap in ordered arguments into a different executable call.
        gaps = [i for i in positional[:positional.index(used[-1]) + 1] if not fields[i]["use"]] if used else []
        missing = [
            i
            for i, spec in enumerate(view["fields"])
            if spec.get("tool") == tool
            and (fields[i]["use"] if "position" in spec else spec.get("required", True))
            and fields[i]["value"] is None
        ]
        missing = list(dict.fromkeys(missing + gaps))
        if missing:
            index = max(missing, key=lambda i: float(output["question"][i, 0]))
            roles = {r["name"] for i in missing for r in view["fields"][i].get("roles", [])}
            roles.update(view["fields"][i]["name"] for i in missing if "position" not in view["fields"][i])
            named = [
                i
                for i, f in enumerate(view["fields"])
                if f.get("questionField") and f["name"] in roles
            ]
            if named:
                index = max(named, key=lambda i: float(output["question"][i, 0]))
            spec = view["fields"][index]
            result["action"] = "ask"
            result["question"] = {
                "field": spec["id"],
                "reason": fields[index].get("reason", "missing"),
                "choices": spec.get("choices", []),
                "text": "Please provide or confirm " + spec["description"] + ".",
            }
            return result
        arguments = {}
        for f, spec in zip(fields, view["fields"], strict=True):
            if spec.get("tool") == tool and f["value"] is not None and f["use"]:
                value = f["value"]
                try:
                    from .structured_tools import validate

                    role = next((r for r in spec.get("roles", []) if r["name"] == f.get("role")), None)
                    value = validate(value, spec, role)
                except (ValueError, TypeError):
                    result["action"] = "ask"
                    result["question"] = {
                        "field": spec["id"],
                        "reason": "invalid_type",
                        "choices": spec.get("choices", []),
                    }
                    break
                arguments[spec["name"]] = value
        if result["action"] == "call_tool":
            result["arguments"] = arguments
            tool_spec = next(t for t in view["public"]["tools"] if t["id"] == tool)
            if tool_spec.get("argumentStyle") == "positional":
                result["argumentFields"] = arguments
                result["arguments"] = [
                    arguments[f["name"]]
                    for f in view["fields"]
                    if f.get("tool") == tool and f["name"] in arguments
                ]
            if tool_spec.get("requiresConfirmation") and digest(
                {"tool": tool, "arguments": result["arguments"]}
            ) not in view.get("confirmations", []):
                result["action"] = "confirm"
                result["question"] = {
                    "field": tool,
                    "reason": "confirmation",
                    "choices": ["approve", "decline"],
                    "value": arguments,
                    "text": "Please confirm the proposed " + tool + " arguments.",
                }
    elif action == "answer":
        result["answer"] = fields[0]["value"]
        result["decision"] = DECISIONS[int(output["decision"][0].argmax())]
        result["evidence"] = [
            {"document": e["id"], "start": e["start"], "end": e["end"]}
            for e in fields[0]["evidence"]
            if e["kind"] == "document"
        ]
        result["scale"] = fields[0]["scale"]
    return result

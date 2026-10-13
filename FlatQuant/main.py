import transformers
import torch

import flatquant.utils as utils
import flatquant.args_utils as args_utils
import flatquant.model_utils as model_utils
import flatquant.data_utils as data_utils
import flatquant.eval_utils as eval_utils
import flatquant.train_utils as train_utils
import flatquant.flat_utils as flat_utils
import gptq_utils


def _parse_deep_layers(deep_layers: str, num_layers: int):
    value = deep_layers.strip().lower()
    if value == "last5":
        start = max(0, num_layers - 5)
        return tuple(range(start, num_layers))

    layers = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        layer_id = int(token)
        if layer_id < 0:
            raise ValueError(f"deep layer id must be >= 0, got: {layer_id}")
        if layer_id >= num_layers:
            raise ValueError(f"deep layer id out of range [0, {num_layers - 1}], got: {layer_id}")
        layers.append(layer_id)
    if not layers:
        raise ValueError("deep_layers must be 'last5' or comma-separated 0-based layer ids.")
    return tuple(sorted(set(layers)))


def _get_qmin_qmax(bits: int, sym: bool):
    if sym:
        q_max = torch.tensor(2 ** (bits - 1) - 1)
        q_min = -q_max - 1
    else:
        q_max = torch.tensor(2**bits - 1)
        q_min = torch.tensor(0)
    return q_max, q_min


def _apply_deep_activation_bits(model, args, logger):
    if not args.enable_deep:
        return

    num_layers = len(model.model.layers)
    deep_layers = set(_parse_deep_layers(args.deep_layers, num_layers))
    logger.info(
        "Applying DEEP activation bits: default_a_bits=%d, deep_act_bits=%d, deep_layers=%s",
        args.a_bits,
        args.deep_act_bits,
        sorted(deep_layers),
    )
    for layer_idx, layer in enumerate(model.model.layers):
        target_bits = args.deep_act_bits if layer_idx in deep_layers else args.a_bits
        for module in layer.modules():
            act_q = getattr(module, "act_quantizer", None)
            if act_q is not None and hasattr(act_q, "bits"):
                act_q.bits = int(target_bits)
                if hasattr(act_q, "q_max") and hasattr(act_q, "q_min"):
                    qmax, qmin = _get_qmin_qmax(target_bits, bool(getattr(act_q, "sym", True)))
                    qmax = qmax.to(act_q.q_max.device)
                    qmin = qmin.to(act_q.q_min.device)
                    act_q.q_max = qmax
                    act_q.q_min = qmin


@torch.no_grad()
def _compute_layer_jump_ratios(model, trainloader, logger):
    eps = 1e-6
    layers = model.model.layers
    sums = [0.0] * len(layers)
    counts = [0] * len(layers)

    def make_hook(idx):
        def hook_fn(module, inputs, output):
            hidden_in = inputs[0]
            hidden_out = output[0] if isinstance(output, tuple) else output
            hidden_in = hidden_in.detach().float()
            hidden_out = hidden_out.detach().float()
            delta = hidden_out - hidden_in
            jr = torch.linalg.norm(delta, ord=2, dim=-1) / (
                torch.linalg.norm(hidden_in, ord=2, dim=-1) + eps
            )
            sums[idx] += jr.sum().item()
            counts[idx] += jr.numel()
        return hook_fn

    device = next(model.parameters()).device
    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    for batch in trainloader:
        inp = batch[0].to(device)
        model(input_ids=inp)
    for handle in handles:
        handle.remove()
    scores = {i: sums[i] / max(counts[i], 1) for i in range(len(layers))}
    logger.info("Jump-ratio scores per layer: %s", {i: round(v, 4) for i, v in scores.items()})
    return scores


def _select_deep_layers_by_ratio(jr_scores, ratio, num_layers, logger, mode="top"):
    k = max(1, round(ratio * num_layers))
    ranked = sorted(jr_scores.items(), key=lambda kv: kv[1], reverse=(mode == "top"))
    selected = sorted(idx for idx, _ in ranked[:k])
    logger.info(
        "JR adaptive DEEP selection (mode=%s): ratio=%.3f -> K=%d/%d layers -> %s (ranking head: %s)",
        mode, ratio, k, num_layers, selected,
        [(i, round(v, 4)) for i, v in ranked[:k]],
    )
    return selected


def _select_deep_layers_random(ratio, num_layers, seed, logger):
    import random as _random

    k = max(1, round(ratio * num_layers))
    rng = _random.Random(seed)
    selected = sorted(rng.sample(range(num_layers), k))
    logger.info(
        "Random DEEP selection (ablation): ratio=%.3f seed=%d -> K=%d/%d layers -> %s",
        ratio, seed, k, num_layers, selected,
    )
    return selected


def main():
    args, logger = args_utils.parser_gen()
    utils.seed_everything(seed=args.seed)

    model, apply_flatquant_to_model = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)

    # get calibration data
    trainloader = data_utils.get_loaders(
        args, args.cali_dataset, tokenizer, 
        nsamples=args.nsamples, seqlen=model.seqlen, eval_mode=False, 
    )
    logger.info("Finished loading training data.")

    if args.quantize:
        if args.enable_deep and args.deep_select in ("ratio", "bottom"):
            model.to(utils.DEV)
            jr_scores = _compute_layer_jump_ratios(model, trainloader, logger)
            selected = _select_deep_layers_by_ratio(
                jr_scores, args.deep_ratio, len(model.model.layers), logger,
                mode=("top" if args.deep_select == "ratio" else "bottom"),
            )
            args.deep_layers = ",".join(str(i) for i in selected)
            model.to("cpu")
            torch.cuda.empty_cache()
        elif args.enable_deep and args.deep_select == "random":
            rand_seed = args.random_seed if args.random_seed is not None else args.seed
            selected = _select_deep_layers_random(
                args.deep_ratio, len(model.model.layers), rand_seed, logger
            )
            args.deep_layers = ",".join(str(i) for i in selected)
        model = apply_flatquant_to_model(args, model)
        logger.info("Finished applying FlatQuant to model.")
        _apply_deep_activation_bits(model, args, logger)
        if args.resume:
            flat_utils.load_flat_parameters(args, model)
        elif args.reload_matrix:
            flat_utils.load_flat_matrices(args, model, path=args.matrix_path)
        elif (args.cali_trans or args.add_diag or args.lwc or args.lac):
            train_utils.cali_flat_quant(args, model, trainloader, utils.DEV, logger=logger)
        if args.save_matrix and not args.reload_matrix:
            flat_utils.save_flat_matrices(args, model)
        flat_utils.reparameterize_model(model)
        logger.info("Finished reparameterize model.")

    if args.w_bits < 16:
        save_dict = {}
        if args.gptq: # GPTQ Weight Quantization
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, utils.DEV, args)
        else: # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)
        save_dict["w_quantizers"] = quantizers

    ## save quantized weight
    if args.quantized_save:
        flat_utils.save_quantized_weights_with_safetensors(args, model, quantizers)

    if args.distribute_model:
        utils.distribute_model(model)
    else:
        model.to(utils.DEV)
    
    # Evaluating PPL
    for eval_dataset in args.eval_datasets:
        logger.info(eval_dataset)
        testloader = data_utils.get_loaders(
                args,
                eval_dataset,
                tokenizer,
                seqlen=model.seqlen,
                eval_mode=True
            )
        dataset_ppl = eval_utils.ppl_eval(model, testloader)
        logger.info(dataset_ppl)


    if args.lm_eval:
        import lm_eval
        from lm_eval import utils as lm_eval_utils
        from lm_eval.models.huggingface import HFLM

        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)

        task_names = args.tasks

        results = {}
        for task_name in task_names:
            logger.info(f"Evaluating {task_name}...")
            result = lm_eval.simple_evaluate(hflm, tasks=[task_name], batch_size=args.lm_eval_batch_size)['results']
            result = result[task_name]
            acc = round(result.get('acc_norm,none', result['acc,none']) * 100, 2)
            results[task_name] = acc
            logger.info(f"acc: {acc}%")
        metric_vals = {task: result for task, result in results.items()}
        metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 2)
        logger.info(metric_vals)


if __name__ == '__main__':
    main()